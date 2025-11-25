# Save this as 'finetune_dataloader.py'
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
    from misc_utils import np_one_hot, EPS # For metadata
except ImportError as e:
    print(f"Error: Could not import MassFormer utilities from {script_dir}")
    print("Please check that the sys.path is correct.")
    print(f"Original error: {e}")
    sys.exit(1)


class TripletRankingDataset(Dataset):
    """
    Dataloader for "Ordinal Pair Ranking" with a DYNAMIC margin.
    
    Each item from this dataset consists of TWO pairs, sampled
    from two different similarity bins, and a calculated margin.
    """
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path, num_samples_per_epoch, margin_per_bin=0.02):
        super().__init__()
        print(f"Loading pairs data from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        print(f"Loading spectral data from {spec_data_path}...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')

        print(f"Loading molecular graph data from {mol_data_path}...")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')

        self.num_samples = num_samples_per_epoch
        self.margin_per_bin = margin_per_bin # Store the dynamic margin base
        print(f"Using a dynamic margin base of {self.margin_per_bin} per bin difference.")

        # --- Create the Bin-to-Index Lookup ---
        print("Creating 10-bin lookup for pair ranking...")
        self.bin_to_indices = defaultdict(list)
        for idx, sim in enumerate(self.pairs_df['cosine_similarity']):
            bin_idx = min(int(np.floor(sim * 10)), 9) 
            self.bin_to_indices[bin_idx].append(idx)
        
        self.available_bins = sorted(self.bin_to_indices.keys())
        print(f"Bin lookup created. {len(self.available_bins)} bins found.")
        self.valid_spec_ids = set(self.spec_lookup.index)
        
    def __len__(self):
        return self.num_samples

    def _get_graphs_from_pair(self, pair_row, idx_offset):
        """Helper function to get the two graph objects for a pair."""
        spec_id_1 = pair_row['name_main']
        spec_id_2 = pair_row['name_sub']
        
        if spec_id_1 not in self.valid_spec_ids or spec_id_2 not in self.valid_spec_ids:
            return None, None 

        mol_id_1 = self.spec_lookup.loc[spec_id_1, 'mol_id']
        mol_id_2 = self.spec_lookup.loc[spec_id_2, 'mol_id']
        
        mol_obj_1 = self.mol_lookup.loc[mol_id_1, 'mol']
        mol_obj_2 = self.mol_lookup.loc[mol_id_2, 'mol']
        
        graph_1 = gf_preprocess(mol_obj_1, idx_offset)
        graph_2 = gf_preprocess(mol_obj_2, idx_offset + 1)
        
        return graph_1, graph_2

    def __getitem__(self, idx):
        # 1. Randomly pick two *different* bins
        bin_i, bin_j = np.random.choice(self.available_bins, 2, replace=False)
        
        # 2. Get a random pair from each bin
        pair_index_i = np.random.choice(self.bin_to_indices[bin_i])
        pair_index_j = np.random.choice(self.bin_to_indices[bin_j])
        
        pair_i = self.pairs_df.iloc[pair_index_i]
        pair_j = self.pairs_df.iloc[pair_index_j]
        
        # 3. Get the 4 graph objects
        graph_i1, graph_i2 = self._get_graphs_from_pair(pair_i, idx * 4)
        graph_j1, graph_j2 = self._get_graphs_from_pair(pair_j, idx * 4 + 2)

        if graph_i1 is None or graph_j1 is None:
            return self.__getitem__((idx + 1) % self.__len__())
            
        # 4. *** NEW DYNAMIC MARGIN LOGIC ***
        #    Calculate the margin based on how far apart the bins are.
        bin_diff = abs(bin_i - bin_j)
        dynamic_margin = bin_diff * self.margin_per_bin 
        target_margin = torch.tensor(dynamic_margin, dtype=torch.float32)

        # 5. Set the target label
        #    We always return (low_sim_pair, high_sim_pair, target, margin)
        #    Target = 1.0 means we want: dist(low_sim) > dist(high_sim)
        
        if bin_i < bin_j: # bin_i is "low sim", bin_j is "high sim"
            target = torch.tensor(1.0, dtype=torch.float32)
            return graph_i1, graph_i2, graph_j1, graph_j2, target, target_margin
        else: # bin_j is "low sim", bin_i is "high sim"
            target = torch.tensor(1.0, dtype=torch.float32)
            return graph_j1, graph_j2, graph_i1, graph_i2, target, target_margin


def triplet_ranking_collate_fn(batch):
    """
    Collates a batch of (graph_Low1, graph_Low2, graph_High1, graph_High2, target, margin)
    """
    # Unzip the batch of 6-tuples
    g_l1_list, g_l2_list, g_h1_list, g_h2_list, target_list, margin_list = zip(*batch)
    
    # Collate the four graph batches separately
    batch_L1 = collator(g_l1_list)
    batch_L2 = collator(g_l2_list)
    batch_H1 = collator(g_h1_list)
    batch_H2 = collator(g_h2_list)
    
    # Stack the target and margin tensors
    batch_targets = torch.stack(target_list, 0)
    batch_margins = torch.stack(margin_list, 0)
    
    return batch_L1, batch_L2, batch_H1, batch_H2, batch_targets, batch_margins

# =============================================================================
# ### UPDATED: Testing Block ###
# =============================================================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Test the TripletRankingDataset and collate function.")
    
    # Required Paths
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to the .feather file of pairs.")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to the spectrum lookup .pkl file (spec_df_2.pkl)")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to the molecule graph .pkl file (mol_df_2.pkl)")
    
    # Test Parameters
    parser.add_argument("--epoch_size", type=int, default=100, help="Small number of samples for testing epoch.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing the loader.")
    parser.add_argument("--margin_per_bin", type=float, default=0.02, help="Dynamic margin for testing.")

    args = parser.parse_args()
    print(f"--- Starting Dataloader Test ---")
    print(f"Args: {args}")
    
    # --- 1. Test Dataset Instantiation ---
    print("\n--- 1. Testing Dataset Instantiation ---")
    try:
        dataset = TripletRankingDataset(
            pairs_feather_path=args.pairs_path,
            spec_data_path=args.spec_data_path,
            mol_data_path=args.mol_data_path,
            num_samples_per_epoch=args.epoch_size,
            margin_per_bin=args.margin_per_bin
        )
        print(f"Dataset initialization successful.")
        print(f"Dataset length (epoch size): {len(dataset)}")
    except Exception as e:
        print(f"\n!!! Dataset Instantiation FAILED !!!")
        print(e)
        sys.exit(1)

    # --- 2. Test __getitem__ ---
    print("\n--- 2. Testing __getitem__ (fetching one sample) ---")
    try:
        item = dataset[0]
        assert len(item) == 6, f"__getitem__ should return 6 items, but got {len(item)}"
        
        g_l1, g_l2, g_h1, g_h2, target, margin = item
        
        print("Successfully fetched one 6-tuple item.")
        print(f"  Type of graph_LowSim_1: {type(g_l1)}")
        print(f"  Type of graph_LowSim_2: {type(g_l2)}")
        print(f"  Type of graph_HighSim_1: {type(g_h1)}")
        print(f"  Type of graph_HighSim_2: {type(g_h2)}")
        print(f"  Type of target: {type(target)}")
        print(f"  Target value: {target.item()} (should be 1.0)")
        print(f"  Type of margin: {type(margin)}")
        print(f"  Dynamic Margin value: {margin.item()} (should be > 0)")
        assert target.item() == 1.0, "Target should always be 1.0"
        
    except Exception as e:
        print(f"\n!!! __getitem__ Test FAILED !!!")
        print(e)
        sys.exit(1)

    # --- 3. Test DataLoader and Collation ---
    print("\n--- 3. Testing DataLoader and Collation ---")
    try:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=triplet_ranking_collate_fn,
            num_workers=0 # Use 0 for simple test
        )
        
        batch = next(iter(loader))
        b_l1, b_l2, b_h1, b_h2, b_targets, b_margins = batch
        
        print("Successfully fetched one batch.")
        print("\n  Batch Contents (showing batch_LowSim_1):")
        for key, value in b_l1.items():
            print(f"    - {key}: {value.shape}")
        
        print(f"\n  Batch Targets Shape: {b_targets.shape} (should be [{args.batch_size}])")
        print(f"  Batch Targets Values: {b_targets.numpy()} (should be all 1.0)")
        print(f"\n  Batch Margins Shape: {b_margins.shape} (should be [{args.batch_size}])")
        print(f"  Batch Margins Values: {b_margins.numpy()} (should be various positive floats)")
        
        assert b_targets.shape == torch.Size([args.batch_size]), "Target batch shape is incorrect."
        assert b_margins.shape == torch.Size([args.batch_size]), "Margin batch shape is incorrect."

    except Exception as e:
        print(f"\n!!! DataLoader Collation Test FAILED !!!")
        print(e)
        sys.exit(1)

    print("\n--- Dataloader Test PASSED Successfully! ---")