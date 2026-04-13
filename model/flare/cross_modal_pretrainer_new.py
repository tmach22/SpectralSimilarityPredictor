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
    # Assuming it might be named differently based on previous logs
    from classifier_siamesemodel import UnpooledMassFormerEncoder

class DenseGCN(nn.Module):
    """A clean, dependency-free Graph Convolutional Network for dense adjacency matrices."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [Batch, Nodes, Features], adj: [Batch, Nodes, Nodes]
        B, N, _ = x.shape
        
        # 1. Add Self-Loops (so atoms remember their own features)
        I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, N, N)
        A_hat = adj + I
        
        # 2. Row Normalize (prevent feature values from exploding for atoms with many bonds)
        deg_inv = 1.0 / (A_hat.sum(dim=-1, keepdim=True) + 1e-8)
        A_norm = A_hat * deg_inv
        
        # 3. Message Passing: Multiply Adjacency by Features
        h = torch.bmm(A_norm, x)
        
        # 4. Feature Transformation & Activation
        return F.relu(self.linear(h))

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
        
        # Combine m/z and intensity into a single token per peak
        x = torch.cat([mz_emb, int_emb], dim=-1)
        x = self.transformer(x, src_key_padding_mask=peak_mask)
        return x

class BifurcatedCrossModalPretrainer(nn.Module):
    def __init__(self, model_config, num_motifs=15):
        super().__init__()
        self.num_motifs = num_motifs
        
        # 1. The Payload Extractor (MassFormer)
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        
        # ==========================================
        # Safe Dimension Extraction
        # ==========================================
        emb_dim = model_config.get('embed_dim', 768)
        if emb_dim == -1:
            emb_dim = model_config.get('hidden_dim', 768)
        if emb_dim == -1:
            emb_dim = 768 # Standard MassFormer PCQM4Mv2 dimension
        
        # 2. The Routing Stream (MPNN + Softmax Head)
        self.gnn_router = nn.Sequential(
            DenseGCN(emb_dim, 256),
            DenseGCN(256, 256)
        )
        self.router_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_motifs)
        )
        
        # 3. The Spectral Encoder
        self.spectral_encoder = SpectralTransformerEncoder(d_model=emb_dim)
        
        # 4. Learnable temperature for Contrastive Loss
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, batched_data, A_brics, peaks, peak_mask, tau=1.0):
        # ==========================================
        # PHASE 1: Extract Chemistry
        # ==========================================
        X_base = self.graph_encoder({'gf_v2_data': batched_data})
        if isinstance(X_base, tuple): X_base = X_base[0]
        # Drop the virtual root node (index 0)
        X = X_base[:, 1:, :] 
        B, N, D = X.shape

        # ==========================================
        # PHASE 2: Bifurcation - The Payload Stream
        # ==========================================
        def parameter_free_seq_norm(x):
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            return (x - mu) / sigma
        
        X_payload = parameter_free_seq_norm(X)

        # ==========================================
        # PHASE 3: Bifurcation - The Routing Stream
        # ==========================================
        # GNN passes messages along A_brics. H shape: [B, N, 256]
        H_1 = self.gnn_router[0](X_payload, A_brics)
        H_2 = self.gnn_router[1](H_1, A_brics)
        
        # Pass topologically-aware features into the MLP Router
        router_in = H_2.reshape(B * N, 256)
        assign_logits = self.router_mlp(router_in).reshape(B, N, self.num_motifs)
        
        # S shape: [B, N, 15]
        S = F.softmax(assign_logits / tau, dim=-1)

        # ==========================================
        # PHASE 4: Motif Pooling
        # ==========================================
        # Multiply Payload by Router to compress atoms into 15 fragments
        Z_graph = torch.bmm(S.transpose(1, 2), X_payload) # [B, 15, D]

        # ==========================================
        # PHASE 5: Spectral Encoding
        # ==========================================
        Z_spec = self.spectral_encoder(peaks, peak_mask) # [B, M, D]

        return Z_graph, Z_spec, S

def bifurcated_contrastive_loss(Z_graph, Z_spec, peak_mask, logit_scale):
    """
    UPDATED: Set-to-Set InfoNCE Contrastive Loss.
    Correctly computes negatives across the batch to avoid representational collapse.
    """
    B, num_motifs, D = Z_graph.shape
    _, M, _ = Z_spec.shape
    
    # 1. L2 Normalize features along the dimension axis
    Z_graph = F.normalize(Z_graph, dim=-1)
    Z_spec = F.normalize(Z_spec, dim=-1)
    
    # 2. Reshape to calculate similarity between ALL graphs and ALL spectra in the batch
    # Z_graph: [B, 1, 15, D]
    # Z_spec:  [1, B, D, M] (Transposed for matmul)
    Z_graph_expand = Z_graph.unsqueeze(1) 
    Z_spec_expand = Z_spec.unsqueeze(0).transpose(2, 3) 
    
    # sim_matrix shape: [Batch_Graph, Batch_Spec, 15, M]
    sim_matrix = torch.matmul(Z_graph_expand, Z_spec_expand)
    
    # 3. Mask out the padding peaks from spectra
    # peak_mask is [B, M] -> Expand to match sim_matrix
    mask_expanded = peak_mask.unsqueeze(0).unsqueeze(2).expand(B, B, num_motifs, M)
    sim_matrix.masked_fill_(mask_expanded, -1e4)
    
    # 4. ColBERT-Style Max-Sim
    # Each motif chooses its absolute best matching peak: [Batch_Graph, Batch_Spec, 15]
    max_sim_g2s, _ = sim_matrix.max(dim=3) 
    
    # Average across the 15 motifs to get one single score per Graph-Spec pair
    # score_matrix shape: [Batch_Graph, Batch_Spec]
    score_matrix = max_sim_g2s.mean(dim=2)
    
    # 5. Scale by temperature
    logits = score_matrix * logit_scale.exp()
    
    # 6. Apply InfoNCE (Symmetric Cross-Entropy)
    labels = torch.arange(B, device=logits.device)
    loss_g2s = F.cross_entropy(logits, labels)
    loss_s2g = F.cross_entropy(logits.T, labels)
    
    return (loss_g2s + loss_s2g) / 2.0