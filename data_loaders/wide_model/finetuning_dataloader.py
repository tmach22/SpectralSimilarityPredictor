import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import os
from pathlib import Path
import numpy as np
import argparse

# Setup sys.path to find massformer src
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from gf_data_utils import gf_preprocess, collator
except ImportError as e:
    print(f"Error importing MassFormer utils: {e}")
    sys.exit(1)

class StageOneDataset(Dataset):
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path, max_nodes=128, upsample_hard_positives=True):
        super().__init__()
        self.max_nodes = max_nodes
        
        print(f"Loading binary pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        print(f"Loading spectral data from {spec_data_path}...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)

        print(f"Loading molecular data from {mol_data_path}...")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        self.valid_mol_ids = set(self.mol_lookup.index)
        
        # --- HARD POSITIVE MINING ---
        if upsample_hard_positives:
            print("Running Hard Positive Mining for Stage 1...")
            mz_map = self.spec_lookup['prec_mz'].to_dict()
            mz_A = self.pairs_df['name_main'].map(mz_map)
            mz_B = self.pairs_df['name_sub'].map(mz_map)
            mass_diff = (mz_A - mz_B).abs()
            
            # Find true matches with >15 Da shift
            hard_mask = (self.pairs_df['label'] == 1) & (mass_diff > 15.0)
            hard_positives = self.pairs_df[hard_mask]
            
            if len(hard_positives) > 0:
                num_copies = 3
                self.pairs_df = pd.concat([self.pairs_df] + [hard_positives] * num_copies)
                self.pairs_df = self.pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                print(f" -> Upsampled {len(hard_positives)} Hard Positives ({num_copies}x copies).")
                print(f" -> New total dataset size: {len(self.pairs_df)}")
            else:
                print(" -> No Hard Positives found in this split.")

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        try:
            pair_info = self.pairs_df.iloc[idx]
            id_A, id_B = pair_info['name_main'], pair_info['name_sub']
            
            spec_A = self.spec_lookup.loc[id_A]
            spec_B = self.spec_lookup.loc[id_B]
            
            mol_A = self.mol_lookup.loc[spec_A['mol_id'], 'mol']
            mol_B = self.mol_lookup.loc[spec_B['mol_id'], 'mol']
            
            graph_A = gf_preprocess(mol_A, idx)
            graph_B = gf_preprocess(mol_B, idx)
            
            # Size checks
            if graph_A.x.size(0) > self.max_nodes or graph_B.x.size(0) > self.max_nodes:
                 raise ValueError(f"Graph too large (> {self.max_nodes})")
            
            # Extract raw masses for the Order Embeddings Loss
            mass_A = torch.tensor([float(spec_A.get("prec_mz", 0.0))], dtype=torch.float32)
            mass_B = torch.tensor([float(spec_B.get("prec_mz", 0.0))], dtype=torch.float32)
            label = torch.tensor(pair_info['label'], dtype=torch.float32)
            
            return graph_A, graph_B, mass_A, mass_B, label

        except Exception as e:
            # Fallback to random sample if a graph fails to process
            new_idx = np.random.randint(0, len(self.pairs_df))
            return self.__getitem__(new_idx)

def stage1_collate_fn(batch):
    graphs_A, graphs_B, mass_A, mass_B, labels = zip(*batch)
    
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    batch_mass_A = torch.stack(mass_A, 0) # [Batch, 1]
    batch_mass_B = torch.stack(mass_B, 0) # [Batch, 1]
    batch_labels = torch.stack(labels, 0)
    
    return batch_A, batch_B, batch_mass_A, batch_mass_B, batch_labels


# =============================================================================
# TESTING BLOCK
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Stage 1 Data Loader (Geometry Phase)")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    
    print("\n--- Testing Stage 1 DataLoader ---")
    try:
        ds = StageOneDataset(
            args.pairs_path, 
            args.spec_data_path, 
            args.mol_data_path, 
            upsample_hard_positives=True
        )
        
        # Test 1: Single Item Unpacking
        print("\n[Test 1: Single Item Extraction]")
        item = ds[0]
        if len(item) == 5:
            g_a, g_b, m_a, m_b, label = item
            print(" ✓ Successfully unpacked 5 items.")
            print(f"   Graph A shape: {g_a.x.shape} nodes/features")
            print(f"   Graph B shape: {g_b.x.shape} nodes/features")
            print(f"   Mass A: {m_a.item():.4f}")
            print(f"   Mass B: {m_b.item():.4f}")
            print(f"   Label:  {label.item()}")
        else:
            print(f" ✗ FAILED: Expected 5 items, got {len(item)}")
            
        # Test 2: Batch Collation
        print("\n[Test 2: Batch Collation]")
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=stage1_collate_fn)
        batch = next(iter(loader))
        b_a, b_b, b_ma, b_mb, b_labels = batch
        
        print(f" ✓ Successfully collated batch of size {args.batch_size}.")
        print(f"   Batch Graph A Keys: {list(b_a.keys())}")
        print(f"   Batch Mass A Tensor: {b_ma.shape} -> Expected [Batch, 1]")
        print(f"   Batch Mass B Tensor: {b_mb.shape} -> Expected [Batch, 1]")
        print(f"   Batch Labels Tensor: {b_labels.shape} -> Expected [Batch]")
        
    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        sys.exit(1)