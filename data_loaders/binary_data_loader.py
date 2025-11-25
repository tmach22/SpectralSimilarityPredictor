import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import os
from pathlib import Path
import argparse
import numpy as np

# Setup sys.path to find massformer src
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from gf_data_utils import gf_preprocess, collator
    from misc_utils import np_one_hot, EPS 
except ImportError as e:
    print(f"Error: Could not import MassFormer utilities from {script_dir}")
    print("Please check that the sys.path is correct.")
    print(f"Original error: {e}")
    sys.exit(1)

class BinaryClassificationDataset(Dataset):
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path):
        super().__init__()
        print(f"Loading binary pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        # Load lookups
        print(f"Loading spectral data from {spec_data_path}...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)

        print(f"Loading molecular data from {mol_data_path}...")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        
        # Setup Metadata Logic (Same as before)
        print("Setting up metadata encoders...")
        self._setup_metadata_maps(spec_df)

    def _setup_metadata_maps(self, spec_df):
        self.ce_key = "ace"
        self.preproc_ce = "normalize"
        self.inst_type_c2i = {s: i for i, s in enumerate(sorted(spec_df["inst_type"].unique()))}
        self.num_inst_type = len(self.inst_type_c2i)
        self.prec_type_c2i = {s: i for i, s in enumerate(sorted(spec_df["prec_type"].unique()))}
        self.num_prec_type = len(self.prec_type_c2i)
        self.frag_mode_c2i = {s: i for i, s in enumerate(sorted(spec_df["frag_mode"].unique()))}
        self.num_frag_mode = len(self.frag_mode_c2i)
        self.mean_ce = spec_df[self.ce_key].mean()
        self.std_ce = spec_df[self.ce_key].std()

    def _process_ce(self, col_energy):
        normalized_ce = (col_energy - self.mean_ce) / (self.std_ce + EPS)
        return torch.tensor([normalized_ce], dtype=torch.float32)

    def _get_spec_meta(self, spec_entry):
        col_energy_meta = self._process_ce(spec_entry[self.ce_key])
        inst_meta = torch.as_tensor(np_one_hot(self.inst_type_c2i[spec_entry["inst_type"]], num_classes=self.num_inst_type), dtype=torch.float32)
        prec_meta = torch.as_tensor(np_one_hot(self.prec_type_c2i[spec_entry["prec_type"]], num_classes=self.num_prec_type), dtype=torch.float32)
        frag_meta = torch.as_tensor(np_one_hot(self.frag_mode_c2i[spec_entry["frag_mode"]], num_classes=self.num_frag_mode), dtype=torch.float32)
        
        return torch.cat([col_energy_meta, inst_meta, prec_meta, frag_meta, col_energy_meta], dim=0).unsqueeze(0)

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        pair_info = self.pairs_df.iloc[idx]
        
        # 1. Get Graphs
        id_A, id_B = pair_info['name_main'], pair_info['name_sub']
        
        # Safety check
        if id_A not in self.valid_spec_ids or id_B not in self.valid_spec_ids:
            # Recursively get next item if invalid
            return self.__getitem__((idx + 1) % len(self))
            
        spec_A = self.spec_lookup.loc[id_A]
        spec_B = self.spec_lookup.loc[id_B]
        
        mol_A = self.mol_lookup.loc[spec_A['mol_id'], 'mol']
        mol_B = self.mol_lookup.loc[spec_B['mol_id'], 'mol']
        
        graph_A = gf_preprocess(mol_A, idx)
        graph_B = gf_preprocess(mol_B, idx)
        
        # 2. Get Metadata
        spec_meta = self._get_spec_meta(spec_A)
        
        # 3. Get Binary Label
        # Ensure it's a float for BCE loss (0.0 or 1.0)
        label = torch.tensor(pair_info['label'], dtype=torch.float32)
        
        return graph_A, graph_B, spec_meta, label

def binary_collate_fn(batch):
    graphs_A, graphs_B, spec_metas, labels = zip(*batch)
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    batch_meta = torch.cat(spec_metas, dim=0)
    batch_labels = torch.stack(labels, 0)
    return batch_A, batch_B, batch_meta, batch_labels

# =============================================================================
# ### NEW: Testing Block ###
# =============================================================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Test the BinaryClassificationDataset.")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to the binary .feather file.")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to spec_df_2.pkl")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to mol_df_2.pkl")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    print(f"--- Starting Binary Dataloader Test ---")
    
    # 1. Init
    try:
        dataset = BinaryClassificationDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
        print(f"Dataset initialized. Size: {len(dataset)}")
    except Exception as e:
        print(f"Dataset Init Failed: {e}")
        sys.exit(1)

    # 2. Get Item
    try:
        print("\n--- Testing __getitem__ ---")
        g_a, g_b, meta, label = dataset[0]
        print(f"Graph A Type: {type(g_a)}")
        print(f"Meta Shape: {meta.shape}")
        print(f"Label: {label.item()} (Type: {label.dtype})")
        
        # Label Sanity Check
        assert label.item() in [0.0, 1.0], f"Label must be 0.0 or 1.0, got {label.item()}"
        assert label.dtype == torch.float32, f"Label must be float32 for BCE loss, got {label.dtype}"
        
    except Exception as e:
        print(f"__getitem__ Failed: {e}")
        sys.exit(1)

    # 3. Collate
    try:
        print("\n--- Testing DataLoader/Collate ---")
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=binary_collate_fn)
        batch = next(iter(loader))
        b_a, b_b, b_meta, b_labels = batch
        
        print(f"Batch Meta Shape: {b_meta.shape} (Should be [{args.batch_size}, feat_dim])")
        print(f"Batch Labels Shape: {b_labels.shape} (Should be [{args.batch_size}])")
        print(f"Batch Labels: {b_labels}")
        
        assert b_labels.shape[0] == args.batch_size
        
    except Exception as e:
        print(f"Collate Failed: {e}")
        sys.exit(1)

    print("\n--- Test PASSED Successfully! ---")