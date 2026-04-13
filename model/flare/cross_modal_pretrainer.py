import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path

# Setup paths to import your existing Siamese components
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model', 'bifurcate'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src'))

try:
    from classifier_siamese_model import UnpooledMassFormerEncoder, DenseGCNConv
except ImportError as e:
    print(f"[-] Error importing Siamese components: {e}")
    sys.exit(1)


# =============================================================================
# 1. THE SPECTRAL ENCODER (FLARE-Compliant Dimensions)
# =============================================================================
class SpectralTransformerEncoder(nn.Module):
    """
    Encodes the sequence of raw MS/MS peaks into a strict, lightweight latent space,
    then projects it up to the 768-D Siamese latent space for cross-modal alignment.
    """
    def __init__(self, in_features=4, flare_dim=256, out_dim=768, num_heads=4, num_layers=4):
        super().__init__()
        
        # 1. Input Projection: Project the 4-D features into the strict 256-D space
        self.peak_proj = nn.Linear(in_features, flare_dim)
        
        # 2. The Native Transformer: Lightweight to prevent overfitting on sparse peaks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=flare_dim, 
            nhead=num_heads, 
            dim_feedforward=flare_dim * 2, 
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. The Bridge: Upscale the 256-D output to match the 768-D Graph Motifs
        self.final_proj = nn.Linear(flare_dim, out_dim)
        
    def forward(self, peaks, peak_mask):
        """
        peaks: Tensor of shape [Batch, Max_Peaks, 4]
        peak_mask: Boolean Tensor of shape [Batch, Max_Peaks] (True where padded)
        """
        # Step 1: [Batch, 60, 4] -> [Batch, 60, 256]
        x = F.leaky_relu(self.peak_proj(peaks), 0.1)
        
        # Step 2: Contextualize the peaks in the strict 256-D space
        z_flare = self.transformer(x, src_key_padding_mask=peak_mask)
        
        # Step 3: Project to Siamese Graph Space [Batch, 60, 256] -> [Batch, 60, 768]
        z_spec = self.final_proj(z_flare)
        
        return z_spec 


# =============================================================================
# 2. THE CROSS-MODAL ORCHESTRATOR
# =============================================================================
class CrossModalFlarePretrainer(nn.Module):
    def __init__(self, model_config, emb_dim=768, num_motifs=15):
        super().__init__()
        
        # --- GRAPH BRANCH (Your existing bifurcated Siamese encoder) ---
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        
        self.kinetic_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.gcn1 = DenseGCNConv(emb_dim, emb_dim)
        self.gcn2 = DenseGCNConv(emb_dim, emb_dim)
        
        self.assignment_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_motifs)
        )
        
        # --- SPECTRAL BRANCH ---
        self.spectral_encoder = SpectralTransformerEncoder(in_features=4, out_dim=emb_dim)
        
        # Learnable temperature scalar for the contrastive loss (initialized to 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward_graph(self, batched_data, A_brics, tau=1.0):
        """Extracts the K motif embeddings from the molecule."""
        with torch.no_grad(): # Keep the base MassFormer frozen
            X_frozen = self.graph_encoder({'gf_v2_data': batched_data})
            
        X_base = X_frozen[:, 1:, :] # Slice off the Virtual Node
        
        # Payload Branch (Sharp Kinetics)
        X_kin = X_base + self.kinetic_mlp(X_base)
        
        def parameter_free_seq_norm(x):
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True, unbiased=False) + 1e-5 
            return (x - mu) / sigma

        X_norm = parameter_free_seq_norm(X_kin)
        
        # Assignment Branch (Smoothed Topology)
        H = self.gcn1(X_norm, A_brics)
        H = self.gcn2(H, A_brics)
        X_pool = parameter_free_seq_norm(H)
        
        assign_logits = self.assignment_head(X_pool)
        S = F.softmax(assign_logits / tau, dim=-1)
        
        # Merge: Z_graph represents the K motifs [Batch, 15, 768]
        Z_graph = torch.bmm(S.transpose(1, 2), X_norm)
        
        return Z_graph, S, X_pool 
        
    def forward(self, batched_data, A_brics, peaks, peak_mask, tau=1.0):
        # 1. Get K Graph Motifs [Batch, 15, 768]
        Z_graph, S, X_pool = self.forward_graph(batched_data, A_brics, tau)
        
        # 2. Get M Spectral Peaks [Batch, 60, 768]
        Z_spec = self.spectral_encoder(peaks, peak_mask)
        
        # 3. L2 Normalize both for stable cosine similarity in the contrastive loss
        Z_graph = F.normalize(Z_graph, p=2, dim=-1)
        Z_spec = F.normalize(Z_spec, p=2, dim=-1)
        
        return Z_graph, Z_spec, S, X_pool


# =============================================================================
# 3. THE FLARE BIDIRECTIONAL MAX-SIM LOSS
# =============================================================================
def flare_contrastive_loss(Z_graph, Z_spec, peak_mask, logit_scale):
    """
    Computes the cross-modal contrastive loss by finding the maximum similarity 
    between graph motifs and spectral peaks, while avoiding padded noise.
    """
    B, K, D = Z_graph.shape
    _, M, _ = Z_spec.shape
    
    # 1. Compute similarity between ALL motifs and ALL peaks in the batch
    Z_graph_flat = Z_graph.view(B * K, D)
    Z_spec_flat = Z_spec.view(B * M, D)
    
    # sim_matrix: [B*15, B*60] -> Reshaped to [GraphBatch, 15, SpecBatch, 60]
    sim_matrix = logit_scale.exp() * torch.matmul(Z_graph_flat, Z_spec_flat.t())
    sim_matrix = sim_matrix.view(B, K, B, M) 
    
    # 2. Mask out padded peaks (set their similarity to a massive negative number)
    mask_expand = peak_mask.view(1, 1, B, M).expand(B, K, B, M)
    sim_matrix.masked_fill_(mask_expand, -1e4)
    
    # 3. Direction A: Graph -> Spec MaxSim
    # For every Motif, which Peak in the spectrum does it match best?
    max_sim_g2s = sim_matrix.max(dim=3)[0] # Shape: [GraphBatch, 15, SpecBatch]
    score_g2s = max_sim_g2s.mean(dim=1)    # Shape: [GraphBatch, SpecBatch]
    
    # 4. Direction B: Spec -> Graph MaxSim
    # For every valid Peak, which Motif in the graph generated it?
    max_sim_s2g = sim_matrix.max(dim=1)[0] # Shape: [GraphBatch, SpecBatch, 60]
    
    # Average only over the VALID peaks, ignoring the padding
    valid_peak_counts = (~peak_mask).sum(dim=1).float().view(1, B)
    max_sim_s2g.masked_fill_(peak_mask.view(1, B, M).expand(B, B, M), 0.0)
    score_s2g = max_sim_s2g.sum(dim=2) / torch.clamp(valid_peak_counts, min=1.0) 
    
    # 5. Final Bidirectional Score Matrix
    logits = (score_g2s + score_s2g) / 2.0 # Shape: [Batch, Batch]
    
    # 6. InfoNCE Cross-Entropy (The diagonal elements are the true matches)
    labels = torch.arange(B, device=logits.device)
    loss_g = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.t(), labels)
    
    return (loss_g + loss_s) / 2.0