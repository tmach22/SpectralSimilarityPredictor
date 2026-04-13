import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from pathlib import Path

# Setup paths to find MassFormer
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from model import Predictor
    from gf_model import GFv2Embedder
except ImportError:
    pass 

# --- 1. THE DUAL-DUSTBIN SINKHORN LAYER ---
class UnbalancedSinkhornOT(nn.Module):
    def __init__(self, epsilon=0.1, max_iters=20, init_penalty=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.dustbin_penalty = nn.Parameter(torch.tensor([init_penalty], dtype=torch.float32))

    def forward(self, X_A, X_B):
        B, K_dim, D = X_A.shape
        _, K_dim_B, _ = X_B.shape

        # Global Scaling to preserve absolute structural severity
        C = torch.cdist(X_A, X_B, p=2.0)
        C_scaled = C / 25.0 

        # The Dual Dustbin: Manual Padding 
        col_pad = self.dustbin_penalty.view(1, 1, 1).expand(B, K_dim, 1)
        C_step1 = torch.cat([C_scaled, col_pad], dim=2) 
        row_pad = self.dustbin_penalty.view(1, 1, 1).expand(B, 1, K_dim_B + 1)
        C_padded = torch.cat([C_step1, row_pad], dim=1) 

        # Sinkhorn-Knopp Iterations 
        K_mat = torch.exp(-C_padded / self.epsilon)
        
        u = torch.ones(B, K_dim + 1, device=X_A.device) / (K_dim + 1)
        v = torch.ones(B, K_dim_B + 1, device=X_B.device) / (K_dim_B + 1)

        for _ in range(self.max_iters):
            u = 1.0 / (torch.bmm(K_mat, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = 1.0 / (torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

        P_padded = u.unsqueeze(-1) * K_mat * v.unsqueeze(1)
        total_transport_cost = torch.sum(P_padded * C_padded, dim=(1, 2)).unsqueeze(-1)

        P_valid = P_padded[:, :K_dim, :K_dim_B]
        P_norm = P_valid / (P_valid.sum(dim=-1, keepdim=True) + 1e-8)

        X_B_aligned = torch.bmm(P_norm, X_B)

        return X_B_aligned, P_valid, total_transport_cost

# --- 2. THE UNPOOLED MASSFORMER ENCODER ---
class UnpooledMassFormerEncoder(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str = None):
        super().__init__()
        dim_d = {"g_dim": 10, "o_dim": 1000}
        full_model = Predictor(dim_d, **model_config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading encoder weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            weights_to_load = state_dict.get('best_model_sd', state_dict)
                
            if 'encoder.encoder.graph_encoder.layers.0.fc1.bias' in weights_to_load:
                new_state_dict = OrderedDict()
                for k, v in weights_to_load.items():
                    if k.startswith('encoder.encoder.graph_encoder'):
                        new_key = k.replace('encoder.encoder.graph_encoder', 'embedders.0.encoder.graph_encoder', 1)
                        new_state_dict[new_key] = v
                    elif k.startswith('encoder.encoder.'):
                        new_key = k.replace('encoder.encoder.', 'embedders.0.encoder.', 1)
                        new_state_dict[new_key] = v
                full_model.load_state_dict(new_state_dict, strict=False)
            else:
                full_model.load_state_dict(weights_to_load, strict=False)
        
        self.encoder = None
        for embedder in full_model.embedders:
            if isinstance(embedder, GFv2Embedder):
                self.encoder = embedder
                break
        if self.encoder is None: raise RuntimeError("GFv2Embedder not found")

        self.unpooled_nodes = None
        
        def hook_fn(module, input, output):
            x = output[0] if isinstance(output, tuple) else output
            self.unpooled_nodes = x

        self.encoder.encoder.graph_encoder.layers[-1].register_forward_hook(hook_fn)

    def forward(self, batched_data: dict) -> torch.Tensor:
        _ = self.encoder(batched_data)
        x = self.unpooled_nodes
        if x.dim() == 3: 
            x = x.transpose(0, 1)
        return x

# --- 3. THE PHASE 3 OVERARCHING MODEL ---
class OptimalTransportSiameseModel(nn.Module):
    def __init__(self, model_config: dict, stage1_checkpoint: str, emb_dim: int = 768, spec_meta_dim: int = 81, num_motifs: int = 15):
        super().__init__()
        self.num_motifs = num_motifs 
        
        self.encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        
        print("\n" + "="*50)
        print(f"STAGE 3: MOTIF-POOLING SINKHORN ALIGNMENT (K={num_motifs})")
        
        if stage1_checkpoint and os.path.exists(stage1_checkpoint):
            checkpoint = torch.load(stage1_checkpoint, map_location="cpu")
            self.encoder.load_state_dict(checkpoint, strict=False)
        
        # Phase 1 Strict Freezing
        for param in self.encoder.parameters():
            param.requires_grad = False

        # --- THE KINETIC PROJECTION HEAD ---
        self.kinetic_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Zero-Initialize the residual branch to preserve physics at Epoch 0
        nn.init.zeros_(self.kinetic_mlp[-1].weight)
        nn.init.zeros_(self.kinetic_mlp[-1].bias)

        # --- THE MOTIF ASSIGNMENT HEAD ---
        self.assignment_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_motifs)
        )

        self.ot_layer = UnbalancedSinkhornOT(epsilon=0.1, max_iters=20)
        
        input_dim = (emb_dim * 2) + 1 + 1 + spec_meta_dim
        
        # Inspector MLP
        self.head = nn.Sequential(
            nn.BatchNorm1d(input_dim), # Tames the massive Sinkhorn transport costs
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),         # Prevents dead neurons
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        print(f" -> OT Inspector MLP Dimension: {input_dim}")
        print("="*50 + "\n")

    # EXPERT FIX: Added 'tau' parameter to support exponential temperature decay
    def forward(self, molecule_A_data, molecule_B_data, spec_meta, mass_A, mass_B, tau=0.5):
        with torch.no_grad():
            X_A_frozen = self.encoder({'gf_v2_data': molecule_A_data})
            X_B_frozen = self.encoder({'gf_v2_data': molecule_B_data})
            
        max_len = max(X_A_frozen.size(1), X_B_frozen.size(1))
        X_A_padded = F.pad(X_A_frozen, (0, 0, 0, max_len - X_A_frozen.size(1)))
        X_B_padded = F.pad(X_B_frozen, (0, 0, 0, max_len - X_B_frozen.size(1)))

        if mass_A.dim() == 1: mass_A = mass_A.unsqueeze(1)
        if mass_B.dim() == 1: mass_B = mass_B.unsqueeze(1)
        
        is_A_heavy = (mass_A >= mass_B).unsqueeze(-1) 
        X_heavy_base = torch.where(is_A_heavy, X_A_padded, X_B_padded)
        X_light_base = torch.where(is_A_heavy, X_B_padded, X_A_padded)
        
        # ====================================================================
        # EXPERT FIX: SLICE OFF THE VIRTUAL NODE
        # Drops the [VNode] at index 0 to align sizes with RDKit and DMoN matrices
        # ====================================================================
        X_heavy_base = X_heavy_base[:, 1:, :]
        X_light_base = X_light_base[:, 1:, :]
        
        # Generate Binary Masks for DMoN Loss
        mask_heavy = (X_heavy_base.abs().sum(dim=-1) > 1e-5).float()
        mask_light = (X_light_base.abs().sum(dim=-1) > 1e-5).float()
        
        # 1. Rotates ground-state into kinetic probability
        X_heavy_kin = X_heavy_base + self.kinetic_mlp(X_heavy_base)
        X_light_kin = X_light_base + self.kinetic_mlp(X_light_base)
        
        # 2. Parameter-Free Sequence Standardization
        def parameter_free_seq_norm(x):
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True, unbiased=False) + 1e-5 
            return (x - mu) / sigma

        X_heavy_norm = parameter_free_seq_norm(X_heavy_kin)
        X_light_norm = parameter_free_seq_norm(X_light_kin)
        
        # 3. Motif Assignment
        assign_logits_heavy = self.assignment_head(X_heavy_norm)
        assign_logits_light = self.assignment_head(X_light_norm)
        
        if self.training:
            assign_logits_heavy += torch.randn_like(assign_logits_heavy) * 0.1
            assign_logits_light += torch.randn_like(assign_logits_light) * 0.1
            
        # EXPERT FIX: Dynamic Temperature injected by the training loop
        S_heavy = F.softmax(assign_logits_heavy / tau, dim=-1)
        S_light = F.softmax(assign_logits_light / tau, dim=-1)
        
        # ====================================================================
        # EXPERT FIX: THE NORMALIZATION CAGE
        # Pooling using the NORMALIZED features (X_heavy_norm) instead of 
        # X_heavy_kin prevents the optimizer from cheating the Sinkhorn 
        # transport cost by scaling the embeddings to infinity.
        # ====================================================================
        Z_heavy = torch.bmm(S_heavy.transpose(1, 2), X_heavy_norm)
        Z_light = torch.bmm(S_light.transpose(1, 2), X_light_norm)
        
        # 4. Sinkhorn Optimal Transport Alignment
        Z_light_aligned, _, transport_cost = self.ot_layer(Z_heavy, Z_light)
        
        delta_motifs = torch.abs(Z_heavy - Z_light_aligned)
        
        valid_mask = (Z_heavy.abs().sum(dim=-1) > 1e-5).float().unsqueeze(-1)
        delta_motifs = delta_motifs * valid_mask
        
        pooled_max = torch.max(delta_motifs, dim=1)[0] 
        pooled_sum = torch.sum(delta_motifs, dim=1)    
        
        mass_diff = torch.abs(mass_A - mass_B) / 100.0
        chem_features = torch.cat((pooled_max, pooled_sum, transport_cost, mass_diff, spec_meta), dim=1)
        
        final_logits = self.head(chem_features)
        
        # Return S matrices and masks so the training loop can calculate DMoN
        return final_logits, S_heavy, S_light, mask_heavy, mask_light