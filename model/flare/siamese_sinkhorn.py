import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

from cross_modal_pretrainer import CrossModalFlarePretrainer

class SiameseSinkhornPredictor(nn.Module):
    def __init__(self, model_config, stage1_ckpt_path, num_motifs=15, meta_dim=78):
        super().__init__()
        
        # 1. Load the Pre-Trained Motif Extractor
        print(f"[*] Loading Stage 1 FLARE Weights from: {stage1_ckpt_path}")
        self.motif_extractor = CrossModalFlarePretrainer(model_config, num_motifs=num_motifs)
        self.motif_extractor.load_state_dict(torch.load(stage1_ckpt_path, map_location='cpu'))
        
        # We drop the spectral encoder entirely to save memory, we don't need it anymore
        del self.motif_extractor.spectral_encoder 
        
        # 2. Sinkhorn OT Parameters
        self.epsilon = 0.05 # Entropy regularization parameter
        self.max_iters = 20
        
        # 3. Final Classification Head
        # Inputs: [Sinkhorn Distance (1), Mass Diff (1), Metadata (meta_dim)]
        in_features = 2 + meta_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1) # Outputs raw logit for BCEWithLogitsLoss
        )

    def compute_sinkhorn_distance(self, Z_A, Z_B):
        """
        Computes the entropic regularized optimal transport distance between two sets of motifs.
        Z_A, Z_B shape: [Batch, Motifs, Dim]
        """
        B, K, D = Z_A.shape
        
        # 1. Cost Matrix: Cosine Distance between Motifs
        # Normalize for stable cosine similarity
        Z_A_norm = F.normalize(Z_A, p=2, dim=-1)
        Z_B_norm = F.normalize(Z_B, p=2, dim=-1)
        
        # Cosine Similarity -> Cosine Distance
        sim_matrix = torch.bmm(Z_A_norm, Z_B_norm.transpose(1, 2))
        C = 1.0 - sim_matrix # Shape: [Batch, K, K]
        
        # 2. Sinkhorn Iterations (Log-domain for numerical stability)
        # Uniform marginals: We assume each motif has equal weight (1/K)
        log_a = torch.full((B, K), -torch.log(torch.tensor(K, dtype=torch.float32)), device=Z_A.device)
        log_b = torch.full((B, K), -torch.log(torch.tensor(K, dtype=torch.float32)), device=Z_A.device)
        
        log_K_matrix = -C / self.epsilon
        
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        
        for _ in range(self.max_iters):
            # Update u
            u = log_a - torch.logsumexp(log_K_matrix + v.unsqueeze(1), dim=2)
            # Update v
            v = log_b - torch.logsumexp(log_K_matrix.transpose(1, 2) + u.unsqueeze(1), dim=2)
            
        # Optimal Transport Plan P
        log_P = log_K_matrix + u.unsqueeze(2) + v.unsqueeze(1)
        P = torch.exp(log_P)
        
        # Sinkhorn Distance = sum(P * Cost)
        distance = torch.sum(P * C, dim=(1, 2)) # Shape: [Batch]
        return distance.unsqueeze(1)

    def forward(self, batch_A, batch_B, A_brics_A, A_brics_B, mass_A, mass_B, meta_features):
        # 1. Extract Physics-Aware Motifs (Frozen or mildly fine-tuned)
        Z_A, S_A, _ = self.motif_extractor.forward_graph(batch_A, A_brics_A, tau=0.5)
        Z_B, S_B, _ = self.motif_extractor.forward_graph(batch_B, A_brics_B, tau=0.5)
        
        # 2. Compute the Structural Transformation Cost
        sinkhorn_dist = self.compute_sinkhorn_distance(Z_A, Z_B)
        
        # 3. Compile the Final Feature Vector
        mass_diff = torch.abs(mass_A - mass_B)
        
        # Concatenate: [Dist, MassDiff, Meta]
        final_features = torch.cat([sinkhorn_dist, mass_diff, meta_features], dim=1)
        
        # 4. Predict Similarity Logit
        logits = self.classifier(final_features)
        return logits.squeeze(-1)