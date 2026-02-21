import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
import os
from pathlib import Path

# Setup paths to find MassFormer source
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

        print(f"Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        weights_to_load = state_dict.get('best_model_sd', state_dict)
            
        # Key Remapping Logic for Fine-Tuned checkpoints
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
        
        if self.encoder is None:
            raise RuntimeError("Could not find GFv2Embedder.")

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
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SiameseSpectralSimilarityModel(nn.Module):
    def __init__(self, model_config: dict, checkpoint_path: str, spec_meta_dim: int):
        super().__init__()
        
        # 1. Encoder
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)

        # 2. Define Head Input
        # 1 (Cosine) + Metadata (includes mass diff)
        similarity_head_input_dim = 1 + spec_meta_dim
        
        # 3. Head
        self.similarity_head = SimilarityHead(input_dim=similarity_head_input_dim)
        
        print(f"Siamese Model Initialized.")
        print(f" - Head Input Dim: {similarity_head_input_dim} (1 Metric + {spec_meta_dim} Meta)")

    def forward(self, molecule_A_data: dict, molecule_B_data: dict, spec_meta: torch.Tensor) -> torch.Tensor:
        # 1. Embeddings
        embedding_A = self.encoder({'gf_v2_data': molecule_A_data})
        embedding_B = self.encoder({'gf_v2_data': molecule_B_data})

        # 2. Cosine Similarity
        sim_metric = F.cosine_similarity(embedding_A, embedding_B, dim=1).unsqueeze(1)
        
        # 3. Concat (Metric + Meta + MassDiff)
        combined_vector = torch.cat((sim_metric, spec_meta), dim=1)
        
        # 4. Classify
        logits = self.similarity_head(combined_vector)
        
        return logits