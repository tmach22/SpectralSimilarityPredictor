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
    from misc_utils import np_one_hot, EPS 
except ImportError as e:
    print(f"Error: Could not import MassFormer utilities from {script_dir}")
    sys.exit(1)


class TripletRankingDataset(Dataset):
    """
    Dataloader for "Hierarchical/Zoom-In" Ordinal Pair Ranking.
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
        self.margin_per_bin = margin_per_bin
        print(f"Base dynamic margin: {self.margin_per_bin} per bin difference.")

        # --- 1. Create Bin Lookup ---
        print("Creating 10-bin lookup...")
        self.bin_to_indices = defaultdict(list)
        for idx, sim in enumerate(self.pairs_df['cosine_similarity']):
            bin_idx = min(int(np.floor(sim * 10)), 9) 
            self.bin_to_indices[bin_idx].append(idx)
        
        self.available_bins = sorted(self.bin_to_indices.keys())
        self.valid_spec_ids = set(self.spec_lookup.index)

        # --- 2. Define Importance Weights (The "Zoom-In" Logic) ---
        print("Configuring 'Zoom-In' Sampling Weights...")
        self.bin_weights = {}
        raw_probs = []
        
        for b in self.available_bins:
            if b >= 7:
                w = 3.0  # "High Sim" bins get 3x importance
            else:
                w = 1.0  # "Low/Med Sim" bins get 1x importance
            
            self.bin_weights[b] = w
            raw_probs.append(w)
            
        total_weight = sum(raw_probs)
        self.bin_probs = [w / total_weight for w in raw_probs]
        
    def __len__(self):
        return self.num_samples

    def _get_graphs_from_pair(self, pair_row, idx_offset):
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
        # 1. SKEWED SAMPLING
        bin_i, bin_j = np.random.choice(
            self.available_bins, 
            size=2, 
            replace=False, 
            p=self.bin_probs
        )
        
        # 2. Get Pairs
        pair_index_i = np.random.choice(self.bin_to_indices[bin_i])
        pair_index_j = np.random.choice(self.bin_to_indices[bin_j])
        
        pair_i = self.pairs_df.iloc[pair_index_i]
        pair_j = self.pairs_df.iloc[pair_index_j]
        
        # *** EXTRACT SIMILARITIES FOR VERIFICATION ***
        sim_i = torch.tensor(pair_i['cosine_similarity'], dtype=torch.float32)
        sim_j = torch.tensor(pair_j['cosine_similarity'], dtype=torch.float32)
        
        # 3. Get Graphs
        graph_i1, graph_i2 = self._get_graphs_from_pair(pair_i, idx * 4)
        graph_j1, graph_j2 = self._get_graphs_from_pair(pair_j, idx * 4 + 2)

        if graph_i1 is None or graph_j1 is None:
            return self.__getitem__((idx + 1) % self.__len__())
            
        # 4. PROGRESSIVE MARGIN CALCULATION
        bin_diff = abs(bin_i - bin_j)
        importance_multiplier = max(self.bin_weights[bin_i], self.bin_weights[bin_j])
        dynamic_margin = bin_diff * self.margin_per_bin * importance_multiplier
        target_margin = torch.tensor(dynamic_margin, dtype=torch.float32)

        # 5. Return Target (1.0 means "First Pair Distance > Second Pair Distance")
        #    We want the distance of the LOWER bin to be LARGER.
        #    So first pair passed should be the LOWER similarity pair.
        
        if bin_i < bin_j: # bin_i is "lower", bin_j is "higher"
            target = torch.tensor(1.0, dtype=torch.float32)
            # UPDATED: Returning 8 items
            return graph_i1, graph_i2, graph_j1, graph_j2, target, target_margin, sim_i, sim_j
        else: # bin_j is "lower", bin_i is "higher"
            target = torch.tensor(1.0, dtype=torch.float32)
            # Swap so the first pair is the lower bin (j), second is higher (i)
            return graph_j1, graph_j2, graph_i1, graph_i2, target, target_margin, sim_j, sim_i


def triplet_ranking_collate_fn(batch):
    # UPDATED: Unpack 8 items
    g_l1_list, g_l2_list, g_h1_list, g_h2_list, target_list, margin_list, sim_low_list, sim_high_list = zip(*batch)
    
    batch_L1 = collator(g_l1_list)
    batch_L2 = collator(g_l2_list)
    batch_H1 = collator(g_h1_list)
    batch_H2 = collator(g_h2_list)
    
    batch_targets = torch.stack(target_list, 0)
    batch_margins = torch.stack(margin_list, 0)
    batch_sim_low = torch.stack(sim_low_list, 0)
    batch_sim_high = torch.stack(sim_high_list, 0)
    
    # Return 8 items
    return batch_L1, batch_L2, batch_H1, batch_H2, batch_targets, batch_margins, batch_sim_low, batch_sim_high

# =============================================================================
# UPDATED: Testing Block
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    print(f"--- Testing Zoom-In Strategy with Detailed Output ---")
    
    # 1. Dataset Init
    dataset = TripletRankingDataset(args.pairs_path, args.spec_data_path, args.mol_data_path, 100)
    
    # 2. Dataloader Batch Test
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=triplet_ranking_collate_fn)
    batch = next(iter(loader))
    
    # Unpack all 8 items
    b_l1, b_l2, b_h1, b_h2, b_targets, b_margins, b_sim_low, b_sim_high = batch
    
    print(f"\n--- Batch Inspection (Size {args.batch_size}) ---")
    
    # A. Inspect Graph Content (Values and Shapes)
    print("\n1. Graph Content Check (Low Sim Pair - Graph 1):")
    # MassFormer graphs are dicts. Let's look at 'x' (node features) or 'attn_bias'
    # Note: Keys depend on your specific GFv2 implementation. 'x' is standard PyG, 
    # but MassFormer often uses 'attn_bias', 'edge_input', etc.
    sample_graph = b_l1
    for k, v in sample_graph.items():
        if isinstance(v, torch.Tensor):
            print(f"   Key: {k:<20} | Shape: {list(v.shape)}")
            # Print first few values of the first node feature/item if applicable
            if k in ['x', 'graph_node_feature', 'attn_bias'] and v.numel() > 0:
                flat_v = v.flatten()
                print(f"     -> First 3 vals: {flat_v[:3].tolist()}")

    # B. Inspect Ground Truth vs Margins
    print("\n2. Similarity & Margin Logic Check:")
    print(f"{'Idx':<4} | {'Low Sim':<10} | {'High Sim':<10} | {'Diff':<6} | {'Margin':<8} | {'Target':<6}")
    print("-" * 65)
    
    for i in range(args.batch_size):
        s_low = b_sim_low[i].item()
        s_high = b_sim_high[i].item()
        marg = b_margins[i].item()
        targ = b_targets[i].item()
        
        # Infer bins for display
        bin_low = int(np.floor(s_low * 10))
        bin_high = int(np.floor(s_high * 10))
        bin_diff = abs(bin_high - bin_low)
        
        print(f"{i:<4} | {s_low:<10.4f} | {s_high:<10.4f} | {bin_diff:<6} | {marg:<8.4f} | {targ:<6.1f}")
        
        # Sanity Assertions
        assert s_low < s_high, f"Error: Low Sim ({s_low}) >= High Sim ({s_high})"
        assert targ == 1.0, "Error: Target must be 1.0"
        
    print("\n--- Test Passed: Graphs have data, and Ranking Logic (Low < High) is correct. ---")