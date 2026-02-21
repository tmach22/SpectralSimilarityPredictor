import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
from pathlib import Path
import argparse

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("Error: RDKit not found. Please install it.")
    sys.exit(1)

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

    def _calculate_tanimoto(self, mol_a, mol_b):
        """Generates fingerprints and calculates Tanimoto similarity."""
        if mol_a is None or mol_b is None:
            return 0.0
        # Generate Morgan Fingerprints (Radius 2, 2048 bits)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

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
        
        # 1. Spectral Label (3 Classes)
        sim = pair_info['cosine_similarity']
        if sim < 0.65:
            label = 0 # Low / Dissimilar
        elif sim < 0.85:
            label = 1 # Medium / Ambiguous
        else:
            label = 2 # High / Identity
            
        # 2. Structural Target (Tanimoto)
        tanimoto = self._calculate_tanimoto(mol_A, mol_B)
        tanimoto_target = torch.tensor(tanimoto, dtype=torch.float32)
            
        # Return 5 items
        return graph_A, graph_B, spec_meta, torch.tensor(label, dtype=torch.long), tanimoto_target

def multiclass_collate_fn(batch):
    # Unpack 5 items
    graphs_A, graphs_B, spec_metas, labels, tanimotos = zip(*batch)
    
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    batch_meta = torch.cat(spec_metas, dim=0)
    batch_labels = torch.stack(labels, 0)
    batch_tanimotos = torch.stack(tanimotos, 0) # New batch tensor for Tanimoto
    
    return batch_A, batch_B, batch_meta, batch_labels, batch_tanimotos

# =============================================================================
# Testing Block
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the Multi-Task Dataset.")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    print(f"--- Testing Multi-Task Dataloader ---")
    try:
        dataset = MulticlassDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
        print(f"Dataset initialized. Size: {len(dataset)}")
    except Exception as e:
        print(f"Init Failed: {e}")
        sys.exit(1)
        
    try:
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=multiclass_collate_fn)
        batch = next(iter(loader))
        _, _, _, b_labels, b_tanimotos = batch
        
        print(f"Batch Labels: {b_labels}")
        print(f"Batch Tanimotos: {b_tanimotos}")
        
        assert b_tanimotos.shape[0] == args.batch_size
        assert b_tanimotos.min() >= 0.0 and b_tanimotos.max() <= 1.0
        
        print("Test Passed: Tanimoto scores are being generated correctly.")
    except Exception as e:
        print(f"Batch Test Failed: {e}")
        sys.exit(1)