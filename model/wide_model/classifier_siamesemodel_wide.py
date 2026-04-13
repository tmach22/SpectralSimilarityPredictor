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
    def __init__(self, emb_dim: int = 768, spec_meta_dim: int = 68, hidden_dim: int = 512):
        super().__init__()
        
        # Phase 1: Wide-Bottleneck Fusion
        # We concatenate: emb_A (768) + emb_B (768) + mass_diff (1) + spec_meta (68)
        # Total input_dim = 1605 (if emb_dim=768 and spec_meta_dim=68)
        input_dim = (emb_dim * 2) + 1 + spec_meta_dim
        
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


class SiameseSpectralSimilarityModel(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str, emb_dim: int = 768, spec_meta_dim: int = 68):
        super().__init__()
        
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        
        # The new Wide-Bottleneck MLP Head
        self.head = SimilarityHead(emb_dim=emb_dim, spec_meta_dim=spec_meta_dim)
        
        print(f"Siamese Model Initialized (Phase 1: Wide-Bottleneck Architecture).")
        print(f"  - Graph Embedding Dim: {emb_dim}")
        print(f"  - MLP Input Dim: {(emb_dim * 2) + 1 + spec_meta_dim}")

    def forward(self, molecule_A_data, molecule_B_data, spec_meta, mass_diff):
        # 1. Structural Embeddings 
        # GFv2Embedder returns the Readout Token: Shape [Batch, 768]
        emb_A = self.encoder({'gf_v2_data': molecule_A_data})
        emb_B = self.encoder({'gf_v2_data': molecule_B_data})
        
        # 2. Prepare Mass Difference for Injection
        # Ensure mass_diff is shaped [Batch, 1] for concatenation
        if mass_diff.dim() == 1:
            mass_diff = mass_diff.unsqueeze(1)
            
        # 3. Phase 1 Feature Engineering: Injection & Concatenation
        # By passing the raw embeddings and the mass difference directly into the MLP, 
        # the network can learn non-linear chemical rules (e.g., Halogen mass shifts)
        chem_features = torch.cat((emb_A, emb_B, mass_diff, spec_meta), dim=1)
        
        # 4. Final Prediction
        final_logits = self.head(chem_features)
        
        return final_logits