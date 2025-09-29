import torch
import torch.nn as nn
import argparse
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import os
import sys
cwd = Path.cwd()

data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)

# --- Assume your finalized DataLoader script is named 'data_loader.py' ---
from data_loader import SpectralPairDataset, collate_fn

# --- Step 1: Define the Actual MassFormer Encoder Architecture ---

class GraphormerLayer(nn.Module):
    """
    A single layer of the Graphormer/MassFormer encoder, containing the
    multi-head self-attention and feed-forward network components.
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x, attn_bias=None):
        # x shape: [batch_size, num_atoms, embedding_dim]
        # In a full implementation, attn_bias would be used to inject graph structure info
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_bias, need_weights=False)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x

class MassFormerEncoder(nn.Module):
    """
    The MassFormer encoder, based on the Graphormer architecture.
    This module converts a batch of molecular graphs into a batch of embedding vectors.
    """
    def __init__(self, num_layers=12, embedding_dim=768, num_heads=12,
                 atom_feature_dim=148, bond_feature_dim=21):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.atom_encoder = nn.Linear(atom_feature_dim, embedding_dim)
        self.edge_encoder = nn.Linear(bond_feature_dim, num_heads)
        self.readout_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.layers = nn.ModuleList([GraphormerLayer(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, graph_batch):
        batch_embeddings = []
        for graph in graph_batch:
            node_features = graph['node_features'].to(self.readout_embedding.device)
            atom_embeddings = self.atom_encoder(node_features)
            full_embeddings = torch.cat([self.readout_embedding.squeeze(0), atom_embeddings], dim=0)
            x = full_embeddings.unsqueeze(0)

            for layer in self.layers:
                x = layer(x)

            graph_embedding = x[:, 0, :] # Select the readout node's embedding
            batch_embeddings.append(graph_embedding)
            
        return torch.cat(batch_embeddings, dim=0)

def load_actual_pretrained_encoder(checkpoint_path: str):
    """
    Loads the MassFormer architecture and populates it with the official
    pre-trained weights from the demo.pkl checkpoint file.
    """
    print("Initializing MassFormer encoder architecture...")
    
    encoder = MassFormerEncoder(
        num_layers=12,
        embedding_dim=768,
        num_heads=12,
        atom_feature_dim=148,
        bond_feature_dim=21
    )

    print(f"Loading pre-trained weights from: {checkpoint_path}")
    try:
        with open(checkpoint_path, 'rb') as f:
            full_model_state_dict = pickle.load(f)
            print(f"Checkpoint keys: {list(full_model_state_dict.keys())[:5]} ...")
            
        # The original model's encoder is named 'transformer'. We need to extract
        # only the weights for this part and remove the prefix.
        encoder_state_dict = {
            k.replace('transformer.', ''): v 
            for k, v in full_model_state_dict.items() 
            if k.startswith('transformer.')
        }

        encoder.load_state_dict(encoder_state_dict)
        print("Successfully loaded pre-trained weights into the encoder.")
        
    except FileNotFoundError:
        print(f"WARNING: Checkpoint file not found at {checkpoint_path}. Using randomly initialized encoder.")
    except Exception as e:
        print(f"An error occurred while loading weights: {e}. Using randomly initialized encoder.")
        
    return encoder

# --- Step 2: The Siamese Network Model Architecture ---

class SiameseSimilarityPredictor(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder
        self.embedding_dim = self.encoder.embedding_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, graph_a_batch, graph_b_batch):
        embedding_a = self.encoder(graph_a_batch)
        embedding_b = self.encoder(graph_b_batch)
        
        diff = torch.abs(embedding_a - embedding_b)
        fused_vector = torch.cat([embedding_a, embedding_b, diff], dim=1)
        
        predicted_similarity = self.regressor(fused_vector).squeeze(-1)
        return predicted_similarity

# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and test the Siamese Network model with pre-trained weights.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the downloaded demo.pkl checkpoint file.")
    parser.add_argument("--test_split_path", type=str, required=True, help="Path to a data split feather file (e.g., validation_pairs.feather).")
    args = parser.parse_args()
    
    print(f"\n--- Loading Pre-trained Encoder from {os.path.basename(args.checkpoint_path)} ---")
    print(f"--- Testing with Data Split: {os.path.basename(args.test_split_path)} ---")

    # 1. Load the pre-trained encoder
    massformer_encoder = load_actual_pretrained_encoder(args.checkpoint_path)
    
    # # 2. Instantiate the full Siamese model
    # model = SiameseSimilarityPredictor(pretrained_encoder=massformer_encoder)
    
    # print("\n--- Model Ready for Training ---")
    # print(f"Total parameters in the Siamese model: {sum(p.numel() for p in model.parameters()):,}")
    
    # # 3. Create a DataLoader to get a real batch of data for testing
    # print(f"\n--- Testing Forward Pass with a real batch from {os.path.basename(args.test_split_path)} ---")
    # test_dataset = SpectralPairDataset(args.test_split_path, SPECTRA_FILE)
    # test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
    
    # try:
    #     graphs_a, graphs_b, ground_truth_scores = next(iter(test_loader))
        
    #     if graphs_a:
    #         with torch.no_grad():
    #             predictions = model(graphs_a, graphs_b)
            
    #         print(f"Input batch size: {len(graphs_a)}")
    #         print(f"Output predictions shape: {predictions.shape}")
    #         print(f"Example Predictions: {predictions.numpy()}")
    #         print(f"Ground Truth Scores: {ground_truth_scores.numpy()}")
    #     else:
    #         print("Could not retrieve a valid batch from the DataLoader.")
            
    # except StopIteration:
    #     print("Could not retrieve a batch. The DataLoader is empty.")