import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import traceback

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

# --- Assuming the following are in your project structure ---
# These are the official MassFormer utilities for data processing
from gf_data_utils import gf_preprocess, collator

class ContrastiveDataset(Dataset):
    """Dataset for loading molecular pairs for contrastive learning."""
    def __init__(self, pairs_path, mol_df, subset_size=None):
        """
        Args:
            pairs_path (str): Path to the feather file containing molecule pairs and labels.
            mol_df (pd.DataFrame): DataFrame containing molecule IDs and RDKit molecule objects.
        """
        self.pairs_df = pd.read_feather(pairs_path)
        self.mol_df = mol_df
        
        # Pre-filter pairs to ensure both molecules exist in mol_df, which speeds up lookup
        mol_ids = set(self.mol_df['mol_id'].tolist())
        self.pairs_df = self.pairs_df[
            self.pairs_df['mol_id_a'].isin(mol_ids) & 
            self.pairs_df['mol_id_b'].isin(mol_ids)
        ].reset_index(drop=True)
        
        # Create a mapping from mol_id to its index in mol_df for fast lookup
        self.mol_id_to_idx = {mol_id: i for i, mol_id in enumerate(self.mol_df['mol_id'])}

        if subset_size is not None:
            print(f"--- Using a subset of the data: {subset_size} ---")
            if isinstance(subset_size, float) and 0 < subset_size <= 1.0:
                # Interpret as a fraction
                self.pairs_df = self.pairs_df.sample(frac=subset_size, random_state=42).reset_index(drop=True)
            elif isinstance(subset_size, int) and subset_size > 0:
                # Interpret as an absolute number of samples
                num_samples = min(subset_size, len(self.pairs_df))
                self.pairs_df = self.pairs_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
            else:
                raise ValueError("subset_size must be a float between 0 and 1, or a positive integer.")

        print(f"Loaded {len(self.pairs_df):,} valid pairs from {pairs_path}")

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        mol_id_a = int(row.mol_id_a)
        mol_id_b = int(row.mol_id_b)
        label = int(row.label)

        # Use the precomputed map to find the molecule objects
        mol_a = self.mol_df.loc[self.mol_id_to_idx[mol_id_a], 'mol']
        mol_b = self.mol_df.loc[self.mol_id_to_idx[mol_id_b], 'mol']

        graph_A = gf_preprocess(mol_a, idx)
        graph_B = gf_preprocess(mol_b, idx)
        
        return graph_A, graph_B, label

def siamese_collate_fn(batch):
    """
    Custom collate function to process batches of molecular pairs for the Siamese model.
    It uses the official MassFormer preprocessing and collating utilities.
    """
    graphs_a, graphs_b, labels = zip(*batch)
    
    # Use the official MassFormer collator to create a padded batch
    collated_a = collator(graphs_a)
    collated_b = collator(graphs_b)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return collated_a, collated_b, labels

# --- Standalone Test Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the Contrastive Dataloader.")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to a contrastive pairs feather file for testing.")
    parser.add_argument("--mol_df_path", type=str, required=True, help="Path to mol_df.pkl.")
    parser.add_argument("--subset_size", type=float, default=1.0, help="Number of pairs to load for testing.")
    args = parser.parse_args()

    print("\n--- Running DataLoader Test ---")
    
    try:
        # 1. Load the molecule dataframe
        print(f"Loading molecule data from: {args.mol_df_path}")
        mol_df = pd.read_pickle(args.mol_df_path)
        mol_df['mol_id'] = mol_df['mol_id'].astype(int)
        print("Molecule data loaded successfully.")

        # 2. Create the Dataset
        print(f"Loading pairs data from: {args.pairs_path}")
        dataset = ContrastiveDataset(args.pairs_path, mol_df, args.subset_size)
        
        # 3. Create the DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=0, # Use 0 for simple testing to avoid multiprocessing issues
            collate_fn=siamese_collate_fn
        )
        print("DataLoader created successfully.")

        # 4. Fetch and inspect a few batches
        print("\n--- Inspecting Batches ---")
        for i, (batch_a, batch_b, labels) in enumerate(dataloader):
            if i >= 2: # Only check the first 2 batches
                break
            
            print(f"\n--- Batch {i+1} ---")
            print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
            print(f"Labels: {labels.numpy()}")
            
            print("\nBatch A (Molecule 1):")
            for key, tensor in batch_a.items():
                print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

            print("\nBatch B (Molecule 2):")
            for key, tensor in batch_b.items():
                print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        print("\n--- DataLoader Test Passed! ---")

    except Exception as e:
        print(f"\n--- DataLoader Test Failed ---")
        traceback.print_exc(file=sys.stdout)
        print(f"Error: {e}")