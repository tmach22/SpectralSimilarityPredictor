import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

from cross_modal_pretrainer import CrossModalFlarePretrainer

class SymmetricCrossAttentionBridge(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        # batch_first=True makes our tensor shapes [Batch, Motifs, Embed_Dim]
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, Z_A, Z_B):
        # A attends to B (Query=A, Key=B, Value=B)
        attn_A_to_B, _ = self.mha(query=Z_A, key=Z_B, value=Z_B)
        out_A = self.layer_norm(Z_A + attn_A_to_B) # Residual connection + Norm
        
        # B attends to A (Query=B, Key=A, Value=A)
        attn_B_to_A, _ = self.mha(query=Z_B, key=Z_A, value=Z_A)
        out_B = self.layer_norm(Z_B + attn_B_to_A)
        
        # Pool the 15 motifs into a single rich representation per molecule
        h_A = out_A.mean(dim=1) # Shape: [Batch, 768]
        h_B = out_B.mean(dim=1)
        
        # Symmetric Feature Fusion
        # This mathematical combination ensures f(A, B) == f(B, A)
        abs_diff = torch.abs(h_A - h_B)
        mult = h_A * h_B
        
        # Concatenate into a unified 1536-dimensional structure vector
        return torch.cat([abs_diff, mult], dim=1)


class SiameseCrossAttentionPredictor(nn.Module):
    def __init__(self, model_config, stage1_ckpt_path, num_motifs=15, meta_dim=81):
        super().__init__()
        
        # 1. Load the Pre-Trained Motif Extractor
        print(f"[*] Loading Stage 1 FLARE Weights from: {stage1_ckpt_path}")
        self.motif_extractor = CrossModalFlarePretrainer(model_config, num_motifs=num_motifs)
        self.motif_extractor.load_state_dict(torch.load(stage1_ckpt_path, map_location='cpu'))
        del self.motif_extractor.spectral_encoder # Free up GPU memory
        
        # 2. Initialize the Cross-Attention Bridge
        embed_dim = model_config.get('hidden_dim', 768)
        self.attention_bridge = SymmetricCrossAttentionBridge(embed_dim=embed_dim)
        
        # 3. Final Classification Head
        # Inputs: [Fused Structure (1536) + Mass Diff (1) + Metadata (81)] = 1618
        in_features = (embed_dim * 2) + 1 + meta_dim
        
        # We use a slightly deeper classifier here because it has to interpret 
        # 1536 dimensions of chemistry rather than a single Sinkhorn distance
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 1) # Outputs raw logit for BCE / Focal Loss
        )

    def forward(self, batch_A, batch_B, A_brics_A, A_brics_B, mass_A, mass_B, meta_features):
        # 1. Extract Physics-Aware Motifs 
        Z_A, S_A, _ = self.motif_extractor.forward_graph(batch_A, A_brics_A, tau=0.5)
        Z_B, S_B, _ = self.motif_extractor.forward_graph(batch_B, A_brics_B, tau=0.5)
        
        # 2. Cross-Attention Structural Comparison
        fused_structure = self.attention_bridge(Z_A, Z_B)
        
        # 3. Compile the Final Feature Vector
        # Ensure mass_diff is explicitly shaped [Batch, 1] for concatenation
        mass_diff = torch.abs(mass_A - mass_B)
        if mass_diff.dim() == 1:
            mass_diff = mass_diff.unsqueeze(1)
            
        final_features = torch.cat([fused_structure, mass_diff, meta_features], dim=1)
        
        # 4. Predict Similarity Logit
        logits = self.classifier(final_features)
        return logits.squeeze(-1)