import torch
import torch.nn as nn
import argparse
from pathlib import Path

import os
import sys
cwd = Path.cwd()

parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)

# We import the necessary classes from the MassFormer codebase
from model import Predictor
from gf_model import GFv2Embedder, set_data_args, set_graphormer_base_architecture_args

from runner import Runner
from args import train_args

class MassFormerEncoder(nn.Module):
    """
    A wrapper class to isolate the pre-trained MassFormer encoder.
    
    This class performs a "head-ectomy" by loading the full pre-trained
    Predictor model but only using its internal GFv2Embedder for inference.
    """
    def __init__(self, model_config: dict, checkpoint_path: str):
        """
        Initializes the encoder.

        Args:
            model_config (dict): The 'model' section of the YAML config file.
            checkpoint_path (str): Path to the.pkl file (e.g., 'checkpoints/demo.pkl').
        """
        super().__init__()

        # We need to create a dummy 'dim_d' dictionary, as the original
        # Predictor class expects it. The values are not critical since
        # we are not using the prediction head.
        dim_d = {"g_dim": 20, "o_dim": 1000}

        # 1. Instantiate the full end-to-end Predictor model
        #    This creates the architecture with random weights.
        full_model = Predictor(dim_d, **model_config)

        # 2. Load the pre-trained weights from the checkpoint file
        print(f"Loading pre-trained weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # The weights are stored under the "best_model_sd" key
        full_model.load_state_dict(state_dict["best_model_sd"])
        print("Weights loaded successfully.")

        # 3. Isolate the GFv2Embedder
        #    We find the correct embedder from the model's 'embedders' list.
        self.encoder = None
        for embedder in full_model.embedders:
            if isinstance(embedder, GFv2Embedder):
                self.encoder = embedder
                break
        
        if self.encoder is None:
            raise RuntimeError("Could not find GFv2Embedder in the loaded model.")
            
        print("MassFormer encoder has been successfully isolated.")

    def forward(self, batched_data: dict) -> torch.Tensor:
        """
        Generates a molecular embedding from preprocessed graph data.

        Args:
            batched_data (dict): A dictionary containing the batched graph data
                                 as expected by the GFv2Embedder.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, embedding_dim]
                          containing the molecular embeddings.
        """
        # The forward pass simply calls the isolated encoder
        return self.encoder(batched_data)

class SimilarityHead(nn.Module):
    """
    An MLP that predicts a similarity score from combined molecular embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        """
        Args:
            input_dim (int): The dimension of the combined input vector.
            hidden_dim (int): The dimension of the hidden layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The combined embedding vector.

        Returns:
            torch.Tensor: A single-value tensor representing the similarity score.
        """
        return self.model(x)

class SiameseSpectralSimilarityModel(nn.Module):
    """
    The complete Siamese network for spectral similarity prediction.
    """
    def __init__(self, model_config: dict, checkpoint_path: str):
        super().__init__()
        
        # 1. Create the shared molecular encoder
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)
        
        # Get the embedding dimension from the encoder's config
        encoder_embedding_dim = self.encoder.encoder.get_embed_dim()
        
        # 2. The input to the similarity head will be the concatenation of:
        #    - embedding_A (size: encoder_embedding_dim)
        #    - embedding_B (size: encoder_embedding_dim)
        #    - |embedding_A - embedding_B| (size: encoder_embedding_dim)
        similarity_head_input_dim = 3 * encoder_embedding_dim
        
        # 3. Create the similarity prediction head
        self.similarity_head = SimilarityHead(input_dim=similarity_head_input_dim)
        
        print(f"Siamese model initialized. Encoder embedding dim: {encoder_embedding_dim}")

    def forward(self, molecule_A_data: dict, molecule_B_data: dict) -> torch.Tensor:
        """
        Performs a forward pass on a pair of molecules.

        Args:
            molecule_A_data (dict): Preprocessed graph data for the first molecule.
            molecule_B_data (dict): Preprocessed graph data for the second molecule.

        Returns:
            torch.Tensor: The predicted similarity score.
        """
        # Generate embeddings for each molecule using the *same* shared encoder
        embedding_A = self.encoder(molecule_A_data)
        embedding_B = self.encoder(molecule_B_data)
        
        # Calculate the element-wise absolute difference
        diff = torch.abs(embedding_A - embedding_B)
        
        # Combine the embeddings for the similarity head
        combined_vector = torch.cat(, dim=1)
        
        # Predict the final score
        score = self.similarity_head(combined_vector)
        
        return score