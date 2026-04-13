import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from pathlib import Path

# --- PEFT IMPORT FOR LORA ---
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Please install peft using: pip install peft")

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

# --- 1. THE EXPERT SINKHORN LAYER (Global Scale & Total Cost) ---
class UnbalancedSinkhornOT(nn.Module):
    def __init__(self, epsilon=0.1, max_iters=20, init_penalty=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.dustbin_penalty = nn.Parameter(torch.tensor([init_penalty], dtype=torch.float32))

    def forward(self, X_A, X_B):
        B, N, D = X_A.shape
        _, M, _ = X_B.shape

        # 1. Cost Matrix & Global Scaling
        C = torch.cdist(X_A, X_B, p=2.0)
        C_scaled = C / 25.0 

        # 2. The Dual Dustbin: Manual Padding 
        col_pad = self.dustbin_penalty.view(1, 1, 1).expand(B, N, 1)
        C_step1 = torch.cat([C_scaled, col_pad], dim=2) 
        row_pad = self.dustbin_penalty.view(1, 1, 1).expand(B, 1, M + 1)
        C_padded = torch.cat([C_step1, row_pad], dim=1) 

        # 3. Sinkhorn-Knopp Iterations 
        K = torch.exp(-C_padded / self.epsilon)
        
        u = torch.ones(B, N + 1, device=X_A.device) / (N + 1)
        v = torch.ones(B, M + 1, device=X_B.device) / (M + 1)

        for _ in range(self.max_iters):
            u = 1.0 / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            v = 1.0 / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # Transport Plan P 
        P_padded = u.unsqueeze(-1) * K * v.unsqueeze(1)

        # Calculate the Total Transport Cost (Work done to map the molecules)
        total_transport_cost = torch.sum(P_padded * C_padded, dim=(1, 2)).unsqueeze(-1)

        # 4. Extract valid alignments
        P_valid = P_padded[:, :N, :M]
        P_norm = P_valid / (P_valid.sum(dim=-1, keepdim=True) + 1e-8)

        # 5. Soft Alignment 
        X_B_aligned = torch.bmm(P_norm, X_B)

        return X_B_aligned, P_valid, total_transport_cost

# --- 2. THE UNPOOLED MASSFORMER ENCODER ---
class UnpooledMassFormerEncoder(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str = None):
        super().__init__()
        dim_d = {"g_dim": 10, "o_dim": 1000}
        full_model = Predictor(dim_d, **model_config)

        if checkpoint_path:
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

        # --- THE INTERCEPTOR (PyTorch Hook) ---
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

# --- 3. THE LORA-TUNED SIAMESE MODEL ---
class OptimalTransportSiameseModel(nn.Module):
    def __init__(self, model_config: dict, stage1_checkpoint: str, emb_dim: int = 768, spec_meta_dim: int = 81):
        super().__init__()
        
        self.encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        
        print("\n" + "="*50)
        print("STAGE 3: LORA-TUNED EXPERT OT SOFT ALIGNMENT")
        
        checkpoint = torch.load(stage1_checkpoint, map_location="cpu")
        self.encoder.load_state_dict(checkpoint, strict=False)
        
        # 1. Freeze the base encoder completely to protect PCQM4Mv2 knowledge
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 2. Inject LoRA adapters into the attention and feed-forward layers
        lora_config = LoraConfig(
            r=8,               
            lora_alpha=16,     
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], 
            lora_dropout=0.1,
            bias="none"
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        print(" -> LoRA Injection Complete:")
        self.encoder.print_trainable_parameters()

        # 3. Latent Space Adapter
        self.latent_adapter = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(emb_dim)
        )

        self.ot_layer = UnbalancedSinkhornOT(epsilon=0.1, max_iters=20)
        
        # 4. Expanded Bottleneck: Max (768) + Sum (768) + OT Cost (1) + Mass Diff (1) + Meta (81) = 1619
        input_dim = (emb_dim * 2) + 1 + 1 + spec_meta_dim
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        print(f" -> OT Inspector MLP Dimension: {input_dim}")
        print("="*50 + "\n")

    def forward(self, molecule_A_data, molecule_B_data, spec_meta, mass_A, mass_B):
        # 1. Extract Tokens (Passes through frozen weights + LoRA adapters)
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
        
        # 2. Warp into Fragmentation Space via Adapter
        X_heavy = self.latent_adapter(X_heavy_base)
        X_light = self.latent_adapter(X_light_base)
        
        # 3. Sinkhorn Optimal Transport Alignment 
        X_light_aligned, _, transport_cost = self.ot_layer(X_heavy, X_light)
        
        # 4. Calculate Masked Structural Delta
        delta_atoms = torch.abs(X_heavy - X_light_aligned)
        valid_mask = (X_heavy_base.abs().sum(dim=-1) > 1e-5).float().unsqueeze(-1)
        delta_atoms = delta_atoms * valid_mask
        
        # 5. Dual Pooling (Max + Sum)
        pooled_max = torch.max(delta_atoms, dim=1)[0] 
        pooled_sum = torch.sum(delta_atoms, dim=1)    
        
        # 6. Final Concatenation & Prediction
        mass_diff = torch.abs(mass_A - mass_B) / 100.0
        chem_features = torch.cat((pooled_max, pooled_sum, transport_cost, mass_diff, spec_meta), dim=1)
        
        final_logits = self.head(chem_features)
        
        return final_logits