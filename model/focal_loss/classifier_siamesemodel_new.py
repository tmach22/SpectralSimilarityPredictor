import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from pathlib import Path

cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from model import Predictor
    from gf_model import GFv2Embedder
except ImportError:
    pass 

class MassFormerEncoder(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str):
        super().__init__()
        dim_d = {"g_dim": 10, "o_dim": 1000}
        full_model = Predictor(dim_d, **model_config)

        if checkpoint_path:
            print(f"Loading encoder weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if 'best_model_sd' in state_dict:
                weights_to_load = state_dict['best_model_sd']
            else:
                weights_to_load = state_dict
                
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

class SimilarityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MassPenaltyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, mass_diff: torch.Tensor) -> torch.Tensor:
        # Softplus ensures penalty is always positive
        return F.softplus(self.fc(mass_diff))

class SiameseSpectralSimilarityModel(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str, spec_meta_dim: int = 68):
        super().__init__()
        
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        
        # Branch 1: Chemical Similarity
        self.head_input_dim = 1 + spec_meta_dim
        self.head = SimilarityHead(input_dim=self.head_input_dim)
        
        # Branch 2: Mass Gating
        self.mass_gate = MassPenaltyHead()
        
        print(f"Siamese Model Initialized (Decoupled Architecture).")
        print(f"  - Head Input Dim: {self.head_input_dim}")
        print(f"  - Mass Gating Active")

    def forward(self, molecule_A_data, molecule_B_data, spec_meta, mass_diff):
        # 1. Structural Embeddings
        emb_A = self.encoder({'gf_v2_data': molecule_A_data})
        emb_B = self.encoder({'gf_v2_data': molecule_B_data})
        
        # 2. Cosine Similarity
        chem_sim = F.cosine_similarity(emb_A, emb_B, dim=1).unsqueeze(1)
        
        # 3. Branch 1: Chemistry Score
        chem_features = torch.cat((chem_sim, spec_meta), dim=1)
        chem_logits = self.head(chem_features)
        
        # 4. Branch 2: Mass Penalty
        penalty = self.mass_gate(mass_diff)
        
        # 5. Final Fusion
        final_logits = chem_logits - penalty
        
        # UPDATED: Return embeddings alongside logits for Loss Calculation
        return final_logits, emb_A, emb_B