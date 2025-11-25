import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import OrderedDict
from pathlib import Path
import os
import sys

# Setup paths to find MassFormer source
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

# Import MassFormer classes
try:
    from model import Predictor
    from gf_model import GFv2Embedder
except ImportError:
    pass # Allow import errors if running purely for code check

# ==============================================================================
# 1. MassFormerEncoder (Same as before - handles fine-tuned loading)
# ==============================================================================
class MassFormerEncoder(nn.Module):
    """
    Isolates the encoder and handles loading of both original and fine-tuned weights.
    """
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
# 2. SimilarityHead (Updated for Logits)
# ==============================================================================
class SimilarityHead(nn.Module):
    """
    MLP head for classification.
    UPDATED: Removed Sigmoid for use with BCEWithLogitsLoss.
    """
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
            # nn.Sigmoid()  <-- REMOVED. Output is now raw logits.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# ==============================================================================
# 3. SiameseSpectralSimilarityModel (Updated for Metric Constraint)
# ==============================================================================
class SiameseSpectralSimilarityModel(nn.Module):
    """
    Siamese network that classifies based on explicit Cosine Similarity + Metadata.
    """
    def __init__(self, model_config: dict, checkpoint_path: str, spec_meta_dim: int, custom_encoder_weights_path: str = None):
        super().__init__()
        
        # 1. Encoder
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)

        if custom_encoder_weights_path:
            print(f"Loading custom weights logic here... (omitted for brevity, handled by train script)")
        
        # 2. Define Input Dimension
        # OLD: (768 * 3) + 8 = 2312
        # NEW: 1 (Cosine Sim) + 8 (Metadata) = 9
        similarity_head_input_dim = 1 + spec_meta_dim
        
        # 3. Create Head
        self.similarity_head = SimilarityHead(input_dim=similarity_head_input_dim)
        
        print(f"Siamese model initialized.")
        print(f"  - Encoder: MassFormer (Frozen or Finetuned)")
        print(f"  - Head Input Dim: {similarity_head_input_dim} (1 Metric + {spec_meta_dim} Meta)")

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
        # shape: [batch_size, 1 + spec_meta_dim]
        combined_vector = torch.cat((sim_metric, spec_meta), dim=1)
        
        # 4. Classify
        # Returns logits (unbounded scores)
        logits = self.similarity_head(combined_vector)
        
        return logits