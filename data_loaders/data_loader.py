import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# Use the DataLoader from torch_geometric, as it knows how to handle graph objects
# from torch_geometric.loader import DataLoader
import argparse

from pathlib import Path
import os
import sys
cwd = Path.cwd()
print(f"Current working directory: {cwd}")
parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)
from gf_data_utils import gf_preprocess, collator

class SpectralSimilarityDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pairs of molecular graphs and their
    spectral similarity score. It uses a two-step lookup to map spectrum IDs
    to their corresponding graph objects.
    """
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path):
        super().__init__()
        print(f"Loading pairs data from {pairs_feather_path}...")
        # This DataFrame contains the instructions (ID_A, ID_B, similarity)
        pairs_df = pd.read_feather(pairs_feather_path)
        
        print(f"Loading spectral data from {spec_data_path}...")
        spec_df = pd.read_pickle(spec_data_path)
        # *** FIX PART 1: Create the first lookup table (spec_id -> mol_id) ***
        self.spec_to_mol_lookup = spec_df.set_index('spec_id')

        print(f"Loading molecular graph data from {mol_data_path}...")
        mol_df = pd.read_pickle(mol_data_path)
        # *** FIX PART 2: Create the second lookup table (mol_id -> graph_obj) ***
        self.mol_lookup = mol_df.set_index('mol_id')

        print("Filtering pairs to match available processed spectra...")
        initial_pair_count = len(pairs_df)
        valid_spec_ids = set(self.spec_to_mol_lookup.index)
        self.pairs_df = pairs_df[
            pairs_df['name_main'].isin(valid_spec_ids) &
            pairs_df['name_sub'].isin(valid_spec_ids)
        ].reset_index(drop=True)
        final_pair_count = len(self.pairs_df)
        print(f"Filtering complete. Kept {final_pair_count} of {initial_pair_count} pairs.")

    def __len__(self):
        # The total number of training examples is the number of pairs.
        return len(self.pairs_df)

    def __getitem__(self, idx):
        # 1. Get the instruction for this specific training example.
        pair_info = self.pairs_df.iloc[idx]
        id_A = pair_info['name_main']
        id_B = pair_info['name_sub']
        similarity = torch.tensor(pair_info['cosine_similarity'], dtype=torch.float32)

        # 2. *** PERFORM THE TWO-STEP MAPPING ***
        # Step A: Use spec_id to find the mol_id for each molecule in the pair.
        mol_id_A = self.spec_to_mol_lookup.loc[id_A, 'mol_id']
        mol_id_B = self.spec_to_mol_lookup.loc[id_B, 'mol_id']

        # 1. Retrieve the RDKit molecule object from the 'mol' column using the mol_id.
        mol_A = self.mol_lookup.loc[mol_id_A, 'mol']
        mol_B = self.mol_lookup.loc[mol_id_B, 'mol']
        
        # 2. Convert the molecule object to a fully-featured graph using gf_preprocess.
        # We pass the dataset index `idx` as required by the function.
        graph_A = gf_preprocess(mol_A, idx)
        graph_B = gf_preprocess(mol_B, idx)
        
        # 3. Return the complete triplet.
        return graph_A, graph_B, similarity

# =============================================================================
# Testing Block: Run this script directly to test the DataLoader
# =============================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test the GNN data loader with rich features.")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to the feather file.")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to the pre-built graph data.")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to the spectrum to molecule mapping data.")
    args = parser.parse_args()
    print(f"Args: {args}")
    
    # # --- 1. Define paths to your data files ---
    # PAIRS_FILE = Path('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/data_splits/val_pairs.feather')
    # # *** ADDED THE PATH TO spec_df.pkl ***
    # SPEC_DATA_FILE = Path('data/proc/spec_df.pkl') # Assumes you run this from the massformer root
    # MOL_DATA_FILE = Path('data/proc/mol_df.pkl')  # Assumes you run this from the massformer root

    print("--- Initializing Dataset ---")
    # Create an instance of the dataset with all three required files
    dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path
    )
    print(f"Dataset initialized successfully. Total number of pairs: {len(dataset)}\n")

    # --- 2. Test retrieving a single item ---
    print("--- Testing __getitem__ (retrieving one sample) ---")
    # Get the first sample from the dataset
    graph_A, graph_B, similarity_score = dataset[0]
    
    print(f"Sample 0 loaded.")
    print(f"  - Type of Graph A: {type(graph_A)}")
    print(f"  - Graph A attributes: {graph_A}")
    print(f"  - Type of Graph B: {type(graph_B)}")
    print(f"  - Graph B attributes: {graph_B}")
    print(f"  - Similarity Score: {similarity_score.item():.4f} (dtype: {similarity_score.dtype})\n")

    # --- 3. Test the DataLoader for batching ---
    print("--- Testing DataLoader (batching multiple samples) ---")
    
    # Create a simple collate function for our Siamese task
    def siamese_collate_fn(batch):
        graphs_A, graphs_B, similarities = zip(*batch)
        batch_A = collator(graphs_A)
        batch_B = collator(graphs_B)
        return batch_A, batch_B, torch.stack(similarities, 0)

    siamese_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=siamese_collate_fn)
    
    # Get the first batch from the new loader
    batch_A, batch_B, batch_similarity = next(iter(siamese_loader))
    
    print("First batch loaded successfully.")
    print("\n--- Batch A (a dictionary of padded tensors) ---")
    for key, value in batch_A.items():
        print(f"  - {key}: {value.shape}")
        
    print("\n--- Batch B (a dictionary of padded tensors) ---")
    for key, value in batch_B.items():
        print(f"  - {key}: {value.shape}")
        
    print(f"\n--- Similarity Tensor ---")
    print(f"  - Shape: {batch_similarity.shape}")
    print(f"  - Values: {batch_similarity.numpy().round(4)}")