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

# --- 1. MassFormerEncoder (Unchanged) ---
class MassFormerEncoder(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str):
        super().__init__()
        dim_d = {"g_dim": 10, "o_dim": 1000}
        full_model = Predictor(dim_d, **model_config)

        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        if 'best_model_sd' in state_dict:
            weights_to_load = state_dict['best_model_sd']
        else:
            weights_to_load = state_dict
            
        if 'encoder.encoder.graph_encoder.layers.0.fc1.bias' in weights_to_load:
            print("Detected fine-tuned checkpoint. Remapping keys...")
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
            print("Detected original checkpoint. Loading directly...")
            full_model.load_state_dict(weights_to_load, strict=False)
        
        print("Base weights loaded successfully.")
        self.encoder = None
        for embedder in full_model.embedders:
            if isinstance(embedder, GFv2Embedder):
                self.encoder = embedder
                break
        if self.encoder is None:
            raise RuntimeError("Could not find GFv2Embedder in the loaded model.")

    def forward(self, batched_data: dict) -> torch.Tensor:
        return self.encoder(batched_data)

# --- 2. SimilarityHead (For Spectral Class) ---
class SimilarityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- 3. TanimotoHead (NEW: For Structural Regression) ---
class TanimotoHead(nn.Module):
    """
    Predicts Tanimoto Similarity from the two embeddings.
    Input: Concatenation of [Emb_A, Emb_B]
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Tanimoto is always between 0 and 1
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        # Concatenate embeddings
        combined = torch.cat((emb_a, emb_b), dim=1)
        return self.model(combined)

# --- 4. SiameseSpectralSimilarityModel (Updated) ---
class SiameseSpectralSimilarityModel(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str, spec_meta_dim: int, num_classes: int = 3):
        super().__init__()
        
        # 1. Encoder
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        encoder_embedding_dim = self.encoder.encoder.get_embed_dim() # usually 768
        
        # 2. Spectral Head (Metric Constrained)
        similarity_head_input_dim = 1 + spec_meta_dim
        self.similarity_head = SimilarityHead(
            input_dim=similarity_head_input_dim, 
            output_dim=num_classes
        )
        
        # 3. Structural Head (Auxiliary)
        self.tanimoto_head = TanimotoHead(embedding_dim=encoder_embedding_dim)
        
        print(f"Multi-Task Siamese model initialized.")
        print(f"  - Main Head: Spectral Classification ({num_classes} classes)")
        print(f"  - Aux Head: Structural Regression (Tanimoto)")

    def forward(self, molecule_A_data: dict, molecule_B_data: dict, spec_meta: torch.Tensor):
        # 1. Generate Embeddings
        embedding_A = self.encoder({'gf_v2_data': molecule_A_data})
        embedding_B = self.encoder({'gf_v2_data': molecule_B_data})

        # --- Task A: Spectral Classification ---
        # Metric Constraint Logic
        sim_metric = F.cosine_similarity(embedding_A, embedding_B, dim=1).unsqueeze(1)
        combined_vector = torch.cat((sim_metric, spec_meta), dim=1)
        spectral_logits = self.similarity_head(combined_vector)
        
        # --- Task B: Structural Regression ---
        # Predict Tanimoto directly from embeddings
        tanimoto_pred = self.tanimoto_head(embedding_A, embedding_B)
        
        return spectral_logits, tanimoto_pred