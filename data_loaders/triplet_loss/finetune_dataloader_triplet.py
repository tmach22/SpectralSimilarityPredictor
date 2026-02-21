import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from pathlib import Path
import argparse

# --- 1. SETUP SYS.PATH ---
# Adjust this logic if your folder structure is different
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from gf_data_utils import gf_preprocess, collator
except ImportError as e:
    print(f"Error: Could not import MassFormer utilities. {e}")
    # For debugging purpose, we might not want to exit if just checking syntax
    # sys.exit(1)

# =============================================================================
# TRIPLET DATASET (Batch-Hard Strategy)
# =============================================================================

class TripletOnlineDataset(Dataset):
    """
    Data Loader for Online Triplet Mining.
    
    Strategy:
    - We iterate over 'Positive Pairs' (Label=1) from the dataset.
    - Each item returns (Anchor_Graph, Positive_Graph, Class_Label).
    - Class_Label is an integer mapped from the Molecule ID.
    
    The Collate Function will:
    - Flatten Anchor and Positive graphs into a single large batch.
    - If batch_size=N, we get 2N graphs.
    - This structure allows the OnlineTripletLoss to mine hard negatives
      from the other pairs in the batch.
    """
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path):
        super().__init__()
        
        print(f"Loading pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        # FILTER: Keep only TRUE MATCHES (Label=1)
        # This guarantees that every item loaded provides a valid Positive pair.
        self.pos_pairs = self.pairs_df[self.pairs_df['label'] == 1].reset_index(drop=True)
        print(f"Filtered for Positive Pairs: {len(self.pos_pairs)} / {len(self.pairs_df)}")
        
        print(f"Loading spectral data...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)

        print(f"Loading molecular data...")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        self.valid_mol_ids = set(self.mol_lookup.index)
        
        # Map Mol IDs to Integers for Triplet Labeling
        # (Triplet Loss requires integer class labels to find matching/non-matching pairs)
        print("Mapping Molecule IDs to Integers...")
        unique_mols = self.mol_lookup.index.unique()
        self.mol_id_to_int = {m_id: i for i, m_id in enumerate(unique_mols)}
        print(f"Mapped {len(unique_mols)} unique molecules to class integers.")

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        try:
            pair_info = self.pos_pairs.iloc[idx]
            id_A, id_B = pair_info['name_main'], pair_info['name_sub']
            
            # 1. Validation checks
            if id_A not in self.valid_spec_ids or id_B not in self.valid_spec_ids:
                 return self.__getitem__((idx + 1) % len(self))
            
            spec_A = self.spec_lookup.loc[id_A]
            
            mol_id_A = spec_A['mol_id']
            # Note: For Label=1, mol_id_B is naturally the same as mol_id_A
            
            if mol_id_A not in self.valid_mol_ids:
                 return self.__getitem__((idx + 1) % len(self))
            
            # 2. Get Graphs
            mol_A = self.mol_lookup.loc[mol_id_A, 'mol']
            
            # Preprocess
            # We use 'idx' and 'idx+1' as seeds if randomization is used in preprocessing
            graph_A = gf_preprocess(mol_A, idx)
            graph_B = gf_preprocess(mol_A, idx + 1) 
            
            if graph_A is None or graph_B is None: 
                return self.__getitem__((idx + 1) % len(self))
                
            # 3. Get Class Label
            label = self.mol_id_to_int[mol_id_A]
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return graph_A, graph_B, label_tensor

        except Exception as e:
            # Fallback to next item on error
            return self.__getitem__((idx + 1) % len(self))

def triplet_collate_fn(batch):
    """
    Collates a list of (Graph_A, Graph_B, Label).
    Returns: 
        batch_graphs: A single MassFormer-collated batch containing [A1, B1, A2, B2, ...]
        batch_labels: A tensor of labels [L1, L1, L2, L2, ...]
    """
    graphs_A, graphs_B, labels = zip(*batch)
    
    # 1. Interleave graphs: A1, B1, A2, B2...
    # This keeps the positive pair together in the list, though order doesn't strictly matter
    # for the loss as long as labels match.
    combined_graphs = []
    combined_labels = []
    
    for gA, gB, lbl in zip(graphs_A, graphs_B, labels):
        combined_graphs.append(gA)
        combined_graphs.append(gB)
        combined_labels.append(lbl)
        combined_labels.append(lbl) # Repeat label for the second view
    
    # 2. Collate Graphs using MassFormer collator
    batch_graphs = collator(combined_graphs)
    
    # 3. Stack Labels
    batch_labels = torch.stack(combined_labels, dim=0)
    
    return batch_graphs, batch_labels

# =============================================================================
# TESTING BLOCK
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Triplet Online Data Loader")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to .feather pairs file")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to spec_df .pkl")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to mol_df .pkl")
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    print("--- 1. Testing Dataset Initialization ---")
    try:
        ds = TripletOnlineDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
        print(f"Dataset successfully created. Size: {len(ds)}")
        
        print("\n--- 2. Testing __getitem__ ---")
        gA, gB, label = ds[0]
        print(f"Graph A Type: {type(gA)}")
        print(f"Graph B Type: {type(gB)}")
        print(f"Label: {label.item()} (Type: {type(label)})")
        
        print("\n--- 3. Testing DataLoader & Collation ---")
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=triplet_collate_fn)
        
        # Fetch one batch
        batch_graphs, batch_labels = next(iter(loader))
        
        print(f"Batch Labels Shape: {batch_labels.shape}")
        print(f"Batch Labels Content: {batch_labels.tolist()}")
        
        # Verification
        expected_size = args.batch_size * 2
        assert batch_labels.shape[0] == expected_size, f"Expected {expected_size} labels, got {batch_labels.shape[0]}"
        assert batch_labels[0] == batch_labels[1], "First two labels should be identical (Positive Pair)"
        
        # Check Graph Keys (standard MassFormer keys)
        print(f"Graph Batch Keys: {batch_graphs.keys()}")
        if 'x' in batch_graphs:
            print(f"Node Features (x) Shape: {batch_graphs['x'].shape}")
            
        print("\n✅ Triplet DataLoader Test PASSED.")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()