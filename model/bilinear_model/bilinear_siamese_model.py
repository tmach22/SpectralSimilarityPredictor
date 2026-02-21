import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
import os
from pathlib import Path

# Setup paths
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from model import Predictor
    from gf_model import GFv2Embedder
except ImportError:
    pass

# --- 1. ENCODER (Same as before) ---
class MassFormerEncoder(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str):
        super().__init__()
        dim_d = {"g_dim": 10, "o_dim": 1000}
        full_model = Predictor(dim_d, **model_config)

        if checkpoint_path:
            print(f"Loading encoder weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            weights_to_load = state_dict.get('best_model_sd', state_dict)
            
            # Fix keys for fine-tuned models
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

    def forward(self, batched_data: dict) -> torch.Tensor:
        return self.encoder(batched_data)

# --- 2. NEW: LOW-RANK BILINEAR HEAD ---
class LowRankBilinearHead(nn.Module):
    def __init__(self, input_dim: int, rank: int = 128):
        super().__init__()
        # Matrix P: Projects input (1000) down to Rank (128)
        # s = (Ph_A)^T (Ph_B)
        self.P = nn.Linear(input_dim, rank, bias=False)
        
    def forward(self, h_A, h_B):
        # 1. Project A and B
        p_A = self.P(h_A) # [Batch, Rank]
        p_B = self.P(h_B) # [Batch, Rank]
        
        # 2. Dot Product (Row-wise)
        # Sum over the Rank dimension
        bilinear_sim = torch.sum(p_A * p_B, dim=1, keepdim=True) # [Batch, 1]
        
        return bilinear_sim

# --- 3. NEW: METADATA GATE ---
class MetadataGate(nn.Module):
    def __init__(self, meta_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Outputs [0, 1]
        )
        
    def forward(self, meta):
        return self.net(meta)

# --- 4. NEW: MASS PENALTY HEAD ---
class MassPenaltyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus() # Ensures Output > 0
        )
        
    def forward(self, mass_diff):
        return self.fc(mass_diff)

# --- 5. ASSEMBLED MODEL ---
class BilinearSiameseModel(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str, spec_meta_dim: int):
        super().__init__()
        
        # Encoder (Output Dim = 1000 usually)
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        self.embedding_dim = 768 # Hardcoded for MassFormer
        
        # 1. Bilinear Interaction (The "Graph Similarity")
        self.bilinear = LowRankBilinearHead(self.embedding_dim, rank=128)
        
        # 2. Metadata Gate (The "Control Signal")
        self.meta_gate = MetadataGate(spec_meta_dim)
        
        # 3. Mass Penalty (The "Scaler")
        self.mass_penalty = MassPenaltyHead()
        
        print("BilinearSiameseModel Initialized:")
        print(f" - Projection Rank: 128")
        print(f" - Metadata Gate Active (GLU-style)")
        print(f" - Mass Logit Scaling Active")

    def forward(self, mol_A, mol_B, spec_meta, mass_diff):
        # 1. Get Embeddings
        h_A = self.encoder({'gf_v2_data': mol_A})
        h_B = self.encoder({'gf_v2_data': mol_B})
        
        # 2. Bilinear Similarity
        # s = h_A^T P^T P h_B
        sim_score = self.bilinear(h_A, h_B)
        
        # 3. Metadata Control Signal
        # gate = Sigmoid(MLP(meta))
        gate_signal = self.meta_gate(spec_meta)
        
        # 4. Chemical Score (Gated)
        # z_chem = sim * gate
        z_chem = sim_score * gate_signal
        
        # 5. Mass Penalty
        # P_mass = Softplus(MLP(mass_diff))
        p_mass = self.mass_penalty(mass_diff)
        
        # 6. Final Logit Scaling
        # z_final = z_chem / (1 + P_mass)
        z_final = z_chem / (1.0 + p_mass)
        
        return z_final