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

# ==============================================================================
# 1. MassFormerEncoder (Unchanged)
# ==============================================================================
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
            
        # Key Remapping Logic
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

# ==============================================================================
# 2. SimilarityHead (Updated for Multiclass Support)
# ==============================================================================
class SimilarityHead(nn.Module):
    """
    MLP head for classification.
    UPDATED: Added 'output_dim' to support Multiclass (e.g., 4 classes).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, output_dim), # Output N logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# ==============================================================================
# 3. SiameseSpectralSimilarityModel (Updated)
# ==============================================================================
class SiameseSpectralSimilarityModel(nn.Module):
    """
    Siamese network using Structural Metric Constraint.
    """
    def __init__(self, model_config: dict, checkpoint_path: str, spec_meta_dim: int, num_classes: int = 1):
        super().__init__()
        
        # 1. Encoder
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        
        # 2. Define Input Dimension
        # Metric Constraint: [1 (Cosine Sim) + Metadata]
        similarity_head_input_dim = 1 + spec_meta_dim
        
        # 3. Create Head
        # Pass num_classes as the output dimension
        self.similarity_head = SimilarityHead(
            input_dim=similarity_head_input_dim, 
            output_dim=num_classes
        )
        
        print(f"Siamese model initialized.")
        print(f"  - Encoder: MassFormer")
        print(f"  - Head Input Dim: {similarity_head_input_dim} (1 Metric + {spec_meta_dim} Meta)")
        print(f"  - Head Output Dim: {num_classes} classes")

    def forward(self, molecule_A_data: dict, molecule_B_data: dict, spec_meta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass enforcing the metric constraint.
        """
        # 1. Generate Embeddings
        embedding_A = self.encoder({'gf_v2_data': molecule_A_data})
        embedding_B = self.encoder({'gf_v2_data': molecule_B_data})

        # 2. Calculate Core Metric (Cosine Similarity)
        # shape: [batch_size]
        sim_metric = F.cosine_similarity(embedding_A, embedding_B, dim=1)
        
        # Reshape to [batch_size, 1] for concatenation
        sim_metric = sim_metric.unsqueeze(1)
        
        # 3. Concatenate Metric with Metadata
        combined_vector = torch.cat((sim_metric, spec_meta), dim=1)
        
        # 4. Classify
        # Returns logits: shape [batch_size, num_classes]
        logits = self.similarity_head(combined_vector)
        
        return logits