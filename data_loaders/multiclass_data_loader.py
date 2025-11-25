import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
from pathlib import Path
import argparse

# Setup sys.path
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from gf_data_utils import gf_preprocess, collator
    from misc_utils import np_one_hot, EPS 
except ImportError:
    pass

class MulticlassDataset(Dataset):
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path):
        super().__init__()
        print(f"Loading pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        print(f"Loading spectral/mol data...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)

        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        
        self._setup_metadata_maps(spec_df)

    def _setup_metadata_maps(self, spec_df):
        self.ce_key = "ace"
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
        
        id_A, id_B = pair_info['name_main'], pair_info['name_sub']
        if id_A not in self.valid_spec_ids or id_B not in self.valid_spec_ids:
            return self.__getitem__((idx + 1) % len(self))
            
        spec_A = self.spec_lookup.loc[id_A]
        spec_B = self.spec_lookup.loc[id_B]
        mol_A = self.mol_lookup.loc[spec_A['mol_id'], 'mol']
        mol_B = self.mol_lookup.loc[spec_B['mol_id'], 'mol']
        
        graph_A = gf_preprocess(mol_A, idx)
        graph_B = gf_preprocess(mol_B, idx)
        spec_meta = self._get_spec_meta(spec_A)
        
        # --- UPDATED 3-CLASS LOGIC ---
        sim = pair_info['cosine_similarity']
        if sim < 0.65:
            label = 0 # Low / Dissimilar
        elif sim < 0.85:
            label = 1 # Medium / Ambiguous
        else:
            label = 2 # High / Identity
            
        return graph_A, graph_B, spec_meta, torch.tensor(label, dtype=torch.long)

def multiclass_collate_fn(batch):
    graphs_A, graphs_B, spec_metas, labels = zip(*batch)
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    batch_meta = torch.cat(spec_metas, dim=0)
    batch_labels = torch.stack(labels, 0)
    return batch_A, batch_B, batch_meta, batch_labels

# Testing Block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the MulticlassDataset (3 Classes).")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to any pairs .feather file.")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to spec_df_2.pkl")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to mol_df_2.pkl")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    print(f"--- Starting Multiclass Dataloader Test (3 Classes) ---")
    
    # 1. Init
    try:
        dataset = MulticlassDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
        print(f"Dataset initialized. Size: {len(dataset)}")
    except Exception as e:
        print(f"Dataset Init Failed: {e}")
        sys.exit(1)

    # 2. Get Item & Logic Check
    try:
        print("\n--- Testing __getitem__ logic (Thresholds: 0.65, 0.85) ---")
        # We will loop until we find at least one example of each class to verify logic
        found_classes = set()
        max_search = 2000
        
        print(f"Scanning first {max_search} items to verify class mapping...")
        print(f"{'Sim':<10} | {'Label':<5} | {'Correct?'}")
        print("-" * 30)
        
        for i in range(min(len(dataset), max_search)):
            # Note: __getitem__ returns (graph_A, graph_B, spec_meta, label)
            # We need to access the internal dataframe to verify the similarity 'sim' 
            # because __getitem__ doesn't return it by default in this version.
            
            # Retrieve the item
            g_a, g_b, meta, label = dataset[i]
            l_val = label.item()
            
            # Retrieve true similarity from dataframe for verification
            true_sim = dataset.pairs_df.iloc[i]['cosine_similarity']
            
            # Verification Logic for 3 Classes
            is_correct = False
            if true_sim < 0.65:
                expected_label = 0
            elif true_sim < 0.85:
                expected_label = 1
            else:
                expected_label = 2
            
            if l_val == expected_label:
                is_correct = True
            
            # Print distinct examples
            if l_val not in found_classes:
                found_classes.add(l_val)
                print(f"{true_sim:<10.4f} | {l_val:<5} | {is_correct}")
                assert is_correct, f"Logic Error: Sim {true_sim} mapped to Label {l_val}, expected {expected_label}"
                
            if len(found_classes) == 3:
                break
                
        print(f"\nFound classes: {found_classes}")
        if len(found_classes) < 3:
            print("Warning: Did not find all 3 classes in the search range. Check your input file distribution.")

    except Exception as e:
        print(f"__getitem__ Failed: {e}")
        sys.exit(1)

    # 3. Collate
    try:
        print("\n--- Testing DataLoader/Collate ---")
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=multiclass_collate_fn)
        batch = next(iter(loader))
        b_a, b_b, b_meta, b_labels = batch
        
        print(f"Batch Meta Shape: {b_meta.shape} (Should be [{args.batch_size}, feat_dim])")
        print(f"Batch Labels Shape: {b_labels.shape} (Should be [{args.batch_size}])")
        print(f"Batch Labels: {b_labels}")
        print(f"Labels Data Type: {b_labels.dtype} (Should be int64/Long for CrossEntropy)")
        
        assert b_labels.shape[0] == args.batch_size
        assert b_labels.dtype == torch.int64
        
    except Exception as e:
        print(f"Collate Failed: {e}")
        sys.exit(1)

    print("\n--- Test PASSED Successfully! ---")