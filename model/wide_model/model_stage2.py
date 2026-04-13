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
    def __init__(self, model_config: dict, checkpoint_path: str = None):
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
    def __init__(self, emb_dim: int = 768, spec_meta_dim: int = 81, hidden_dim: int = 512):
        super().__init__()
        
        # Phase 1: Wide-Bottleneck Fusion
        # Input: h_heavy (768) + h_light (768) + mass_diff (1) + spec_meta (81) = 1618
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


class StageTwoClassificationModel(nn.Module):
    def __init__(self, model_config: dict, stage1_checkpoint: str, emb_dim: int = 768, spec_meta_dim: int = 81):
        super().__init__()
        
        # 1. Initialize the base encoder (do not load the original pcqm4mv2 weights here)
        self.encoder = MassFormerEncoder(model_config, checkpoint_path=None)
        
        # 2. Initialize the Wide-Bottleneck MLP Head
        self.head = SimilarityHead(emb_dim=emb_dim, spec_meta_dim=spec_meta_dim)
        
        # 3. Load the Fine-Tuned Stage 1 Weights
        print("\n" + "="*50)
        print("STAGE 2: CLASSIFICATION HEAD TRAINING")
        print(f" -> Loading fine-tuned geometry from: {stage1_checkpoint}")
        
        # Load state dict (strict=False because Stage 1 dict won't contain the new 'head' weights)
        checkpoint = torch.load(stage1_checkpoint, map_location="cpu")
        self.load_state_dict(checkpoint, strict=False)
        
        # 4. FREEZE THE ENCODER
        # Lock the geometry so MLP gradients don't ruin the Order Embeddings
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        print(" -> ✓ Encoder frozen. Only the MLP head will be trained.")
        print(f" -> MLP Input Dimension: {(emb_dim * 2) + 1 + spec_meta_dim}")
        print("="*50 + "\n")

    def forward(self, molecule_A_data, molecule_B_data, spec_meta, mass_A, mass_B):
        # 1. Structural Embeddings (Frozen Forward Pass)
        with torch.no_grad():
            emb_A = self.encoder({'gf_v2_data': molecule_A_data})
            emb_B = self.encoder({'gf_v2_data': molecule_B_data})
            
            # Enforce positive orthant
            emb_A = F.relu(emb_A)
            emb_B = F.relu(emb_B)
            
        # 2. Geometry Fix: Dynamic Sorting
        # Ensure mass tensors are [Batch, 1] for broadcasting
        if mass_A.dim() == 1: mass_A = mass_A.unsqueeze(1)
        if mass_B.dim() == 1: mass_B = mass_B.unsqueeze(1)
        
        is_A_heavy = (mass_A >= mass_B)
        
        # h_heavy is always the parent graph, h_light is always the subgraph
        h_heavy = torch.where(is_A_heavy, emb_A, emb_B)
        h_light = torch.where(is_A_heavy, emb_B, emb_A)
        
        # Calculate normalized mass diff dynamically
        mass_diff = torch.abs(mass_A - mass_B) / 100.0
            
        # 3. Concatenate (The Wide Bottleneck)
        chem_features = torch.cat((h_heavy, h_light, mass_diff, spec_meta), dim=1)
        
        # 4. Final Prediction (Trainable Pass)
        final_logits = self.head(chem_features)
        
        return final_logits