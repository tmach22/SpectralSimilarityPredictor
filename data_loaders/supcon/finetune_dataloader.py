import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from collections import defaultdict
import argparse
from pathlib import Path

# --- 1. SETUP SYS.PATH ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from gf_data_utils import gf_preprocess, collator
    from misc_utils import np_one_hot, EPS 
except ImportError as e:
    print(f"Error: Could not import MassFormer utilities from {script_dir}")
    sys.exit(1)

# =============================================================================
# DATASET CLASS FOR SUPCON (PAIRS BASED)
# =============================================================================

class SupConPairsDataset(Dataset):
    """
    SupCon Dataset that iterates over a PAIRS file (Binary Label 0/1).
    
    Goal:
    - We want to use the 'mol_id' to define the "Class" for SupCon.
    - True Pairs (1) will have the SAME mol_id.
    - False Pairs (0) will have DIFFERENT mol_ids.
    
    Returns:
    - Graph_A, Graph_B (The pair)
    - Mol_ID_A, Mol_ID_B (The labels for SupCon)
    """
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path, max_nodes=128):
        super().__init__()
        self.max_nodes = max_nodes
        
        print(f"Loading pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        print(f"Loading spectral data from {spec_data_path}...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)

        print(f"Loading molecular data from {mol_data_path}...")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        self.valid_mol_ids = set(self.mol_lookup.index)
        
        # --- ID MAPPING ---
        # SupCon Loss requires Integer Labels (0, 1, 2...). 
        # Mol_IDs are often strings or hash codes. We map them to Ints.
        print("Mapping Molecule IDs to Integers...")
        unique_mols = self.mol_lookup.index.unique()
        self.mol_id_to_int = {m_id: i for i, m_id in enumerate(unique_mols)}
        print(f"Mapped {len(unique_mols)} unique molecules.")

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        try:
            pair_info = self.pairs_df.iloc[idx]
            id_A, id_B = pair_info['name_main'], pair_info['name_sub']
            
            # 1. Validation
            if id_A not in self.valid_spec_ids or id_B not in self.valid_spec_ids:
                 return self.__getitem__((idx + 1) % len(self))
            
            spec_A = self.spec_lookup.loc[id_A]
            spec_B = self.spec_lookup.loc[id_B]
            
            mol_id_A_raw = spec_A['mol_id']
            mol_id_B_raw = spec_B['mol_id']
            
            if mol_id_A_raw not in self.valid_mol_ids or mol_id_B_raw not in self.valid_mol_ids:
                 return self.__getitem__((idx + 1) % len(self))
            
            # 2. Get Graphs
            mol_A = self.mol_lookup.loc[mol_id_A_raw, 'mol']
            mol_B = self.mol_lookup.loc[mol_id_B_raw, 'mol']
            
            graph_A = gf_preprocess(mol_A, idx)
            graph_B = gf_preprocess(mol_B, idx) # Use same seed/idx logic
            
            if graph_A is None or graph_B is None: 
                return self.__getitem__((idx + 1) % len(self))
                
            # 3. Get Integer Labels (The "Class" for SupCon)
            # If it's a True Pair (Label 1), these ints will be identical.
            # If it's a False Pair (Label 0), these ints will be different.
            label_A = self.mol_id_to_int[mol_id_A_raw]
            label_B = self.mol_id_to_int[mol_id_B_raw]
            
            # Convert to Tensor
            tensor_label_A = torch.tensor(label_A, dtype=torch.long)
            tensor_label_B = torch.tensor(label_B, dtype=torch.long)
            
            return graph_A, graph_B, tensor_label_A, tensor_label_B

        except Exception as e:
            # Fallback
            return self.__getitem__((idx + 1) % len(self))

def supcon_pairs_collate_fn(batch):
    """
    Collates (Graph_A, Graph_B, Label_A, Label_B)
    """
    graphs_A, graphs_B, labels_A, labels_B = zip(*batch)
    
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    
    batch_labels_A = torch.stack(labels_A, 0)
    batch_labels_B = torch.stack(labels_B, 0)
    
    return batch_A, batch_B, batch_labels_A, batch_labels_B

# =============================================================================
# TEST BLOCK
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    print("--- Testing SupCon Pairs Loader ---")
    try:
        ds = SupConPairsDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
        print(f"Dataset Size: {len(ds)}")
        
        # Test Fetch
        gA, gB, lA, lB = ds[0]
        print(f"Sample 0: Label A={lA.item()}, Label B={lB.item()}")
        
        # If it's a positive pair, labels should match
        # If it's a negative pair, labels should differ
        
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=supcon_pairs_collate_fn)
        bA, bB, blA, blB = next(iter(loader))
        
        print(f"Batch Labels A: {blA}")
        print(f"Batch Labels B: {blB}")
        
        # For SupCon Training, you will concatenate these!
        # cat_labels = torch.cat([blA, blB], dim=0) 
        # This gives a batch of size 2*N
        
        print("Test PASSED.")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()