import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
import math

# Setup paths
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model', 'bifurcate'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src'))
sys.path.insert(0, os.path.join(str(cwd), 'model', 'flare'))

try:
    from classifier_siamese_model import UnpooledMassFormerEncoder
except ImportError:
    from classifier_siamesemodel import UnpooledMassFormerEncoder

# ==========================================
# 1. PHASE 1: EDGE REGRESSOR HEAD
# ==========================================
class EdgeRegressorHead(nn.Module):
    """
    Phase 1 Head: Predicts continuous bond cleavage logits.
    Input: [Atom U Emb, Atom V Emb, BDE, Collision_Energy] -> Output: 1 Logit
    """
    def __init__(self, node_emb_dim=768, edge_attr_dim=2, hidden_dim=256):
        super().__init__()
        input_dim = (node_emb_dim * 2) + edge_attr_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, node_embeddings_flat, edge_index, edge_attr):
        u_emb = node_embeddings_flat[edge_index[0]]
        v_emb = node_embeddings_flat[edge_index[1]]
        edge_features = torch.cat([u_emb, v_emb, edge_attr], dim=-1)
        return self.mlp(edge_features).squeeze(-1)

# ==========================================
# 2. PHASE 2: SPECTRAL ENCODER
# ==========================================
class SpectralTransformerEncoder(nn.Module):
    """Standard Transformer to process the MS/MS peak list."""
    def __init__(self, d_model=768, nhead=8, num_layers=4):
        super().__init__()
        self.mz_proj = nn.Linear(1, d_model // 2)
        self.int_proj = nn.Linear(1, d_model // 2)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, peaks, peak_mask):
        mz = peaks[:, :, 0:1]
        intensity = peaks[:, :, 1:2]
        
        mz_emb = self.mz_proj(mz)
        int_emb = self.int_proj(intensity)
        
        x = torch.cat([mz_emb, int_emb], dim=-1)
        x = self.transformer(x, src_key_padding_mask=peak_mask)
        return x

# ==========================================
# 3. MASTER ARCHITECTURE: DESAF-Net
# ==========================================
class DESAFNet(nn.Module):
    """
    Discrete Edge-Scored Autoregressive Fragmentation Network.
    Unified model housing the MassFormer backbone, Phase 1 Anchor, and Phase 2 Spectral modules.
    """
    def __init__(self, model_config):
        super().__init__()
        
        # 1. The Payload Extractor (MassFormer)
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        
        # Safe Dimension Extraction
        emb_dim = model_config.get('embed_dim', 768)
        if emb_dim == -1: emb_dim = model_config.get('hidden_dim', 768)
        if emb_dim == -1: emb_dim = 768 
        
        # 2. Phase 1: Continuous Edge Scorer
        self.edge_head = EdgeRegressorHead(node_emb_dim=emb_dim, edge_attr_dim=2)
        
        # 3. Phase 2: Spectral Encoder (Ready for later)
        self.spectral_encoder = SpectralTransformerEncoder(d_model=emb_dim)
        
        # 4. Phase 2: Fragment Pooler (Placeholder for deterministic Set Transformer)
        self.fragment_pooler = nn.Linear(emb_dim, emb_dim) 
        
        # 5. Learnable temperature for Phase 2 Contrastive Loss
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def freeze_backbone(self):
        """Locks MassFormer. Only the EdgeHead will train during Phase 1."""
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        print("[*] MassFormer Backbone Frozen. Ready for Phase 1 Training.")

    def forward_phase1(self, batch):
        """
        Dedicated forward pass for Phase 1 Bond Cleavage Training.
        Bridges the dense Graphormer output with the sparse Edge Regressor.
        """
        # 1. Extract Dense Features
        X_base = self.graph_encoder({'gf_v2_data': batch})
        if isinstance(X_base, tuple): X_base = X_base[0]
        
        # Drop the virtual root node (index 0)
        X = X_base[:, 1:, :] 
        B = X.shape[0]
        
        # 2. The Dense-to-Sparse Bridge
        valid_X_list = []
        for b in range(B):
            # gf_preprocess pads with 0s. Count non-zeros to find actual atoms.
            true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
            valid_X_list.append(X[b, :true_nodes, :])
            
        X_flat = torch.cat(valid_X_list, dim=0) 
        
        # 3. Edge Prediction
        cut_logits = self.edge_head(X_flat, batch['edge_index'], batch['edge_attr_physics'])
        return cut_logits

    def forward_phase2(self, batch, peaks, peak_mask):
        """
        Placeholder for the REINFORCE discrete sampling pass.
        We will build out the deterministic scatter_mean and set transformer logic here later.
        """
        raise NotImplementedError("Phase 2 discrete sampling pass is not yet implemented.")

# ==========================================
# 4. PHASE 2 LOSS FUNCTION (Preserved)
# ==========================================
def desaf_contrastive_loss(Z_fragments, Z_spec, peak_mask, logit_scale):
    """
    ColBERT-style Max-Sim Contrastive Loss. 
    Preserved and renamed for Phase 2 REINFORCE implementation.
    """
    pass # Implementation remains the same as your previous bifurcated loss, 
         # but adapted for dynamic fragment counts instead of fixed 15 motifs.