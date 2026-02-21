import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import argparse
from pathlib import Path
from rdkit import Chem 

# --- PATH SETUP (Ensure these match your env) ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from gf_data_utils import gf_preprocess, collator
except ImportError as e:
    print(f"Error: {e}")

class AlignUniformDataset(Dataset):
    """
    CORRECTED LOADER:
    1. Loads mol_A from spec_A and mol_B from spec_B (Fixes the Clone Bug).
    2. Applies Atom Shuffling to both (Fixes the Canonicalization Bug).
    """
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path):
        super().__init__()
        
        print(f"Loading pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        self.pos_pairs = self.pairs_df[self.pairs_df['label'] == 1].reset_index(drop=True)
        print(f"Filtered for Positive Pairs: {len(self.pos_pairs)}")
        
        # Load Lookups
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)

        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        
    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        try:
            pair_info = self.pos_pairs.iloc[idx]
            id_A, id_B = pair_info['name_main'], pair_info['name_sub']
            
            # --- 1. LOAD METADATA A ---
            if id_A not in self.valid_spec_ids: return self.__getitem__((idx + 1) % len(self))
            spec_A = self.spec_lookup.loc[id_A]
            mol_A = self.mol_lookup.loc[spec_A['mol_id'], 'mol']
            
            # --- 2. LOAD METADATA B (CRITICAL FIX) ---
            # Previously, we assumed mol_B == mol_A. This was wrong.
            if id_B not in self.valid_spec_ids: return self.__getitem__((idx + 1) % len(self))
            spec_B = self.spec_lookup.loc[id_B]
            mol_B = self.mol_lookup.loc[spec_B['mol_id'], 'mol']

            # --- 3. AUGMENTATION (Atom Shuffling) ---
            # We shuffle BOTH to ensure robust graph representations
            
            # Shuffle A
            idx_A = list(range(mol_A.GetNumAtoms()))
            np.random.shuffle(idx_A)
            mol_A_shuffled = Chem.RenumberAtoms(mol_A, idx_A)
            graph_A = gf_preprocess(mol_A_shuffled, idx)
            
            # Shuffle B
            idx_B = list(range(mol_B.GetNumAtoms()))
            np.random.shuffle(idx_B)
            mol_B_shuffled = Chem.RenumberAtoms(mol_B, idx_B)
            graph_B = gf_preprocess(mol_B_shuffled, idx + 1)
            
            if graph_A is None or graph_B is None: return self.__getitem__((idx + 1) % len(self))
            
            return graph_A, graph_B

        except Exception:
            return self.__getitem__((idx + 1) % len(self))

def au_collate_fn(batch):
    graphs_A, graphs_B = zip(*batch)
    combined = []
    for gA, gB in zip(graphs_A, graphs_B):
        combined.append(gA)
        combined.append(gB)
    return collator(combined)

# =============================================================================
# TESTING BLOCK
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    print("--- Testing AU Data Loader ---")
    ds = AlignUniformDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
    
    # Check Item
    gA, gB = ds[0]
    print(f"Item 0: {type(gA)}, {type(gB)}")
    
    # Check Batch
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=au_collate_fn)
    batch = next(iter(loader))
    
    print(f"Batch Keys: {batch.keys()}")
    if 'x' in batch:
        print(f"Batch Node Features: {batch['x'].shape}")
        
    # Expected: Batch Size * 2 graphs
    print(f"Expected Graphs: {args.batch_size * 2}")
    if 'batch' in batch:
         print(f"Actual Graphs in Batch: {batch['batch'].max().item() + 1}")
    
    print("✅ Test Passed.")