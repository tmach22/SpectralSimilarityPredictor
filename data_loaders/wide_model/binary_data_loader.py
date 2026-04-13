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
    print(f"Error importing MassFormer utils: {e}")
    sys.exit(1)

# =============================================================================
# 1. FIXED VOCABULARY DEFINITIONS
# =============================================================================

FIXED_INSTRUMENTS = [
    "QTOF", "Orbitrap", "Triple Quad", "Ion Trap", "Thermo Q Exactive HF",
    "Unknown", "None", "Other"
]

FIXED_ADDUCTS = [
    # Positive Mode
    "[M+H]+", "[M+NA]+", "[M+NH4]+", "[M+K]+", "[M+H-H2O]+",
    "[2M+H]+", "[M+ACN+H]+", "[M]+", "[2M+NA]+",
    "[M-3H2O+H]+", "[M-2H2O+H]+", "[M+CA]2+", "[M-2H2O+NH4]+",
    "[2M+K]+", "[2M+NH4]+", "[2M+CA]2+", "[M-H+2NA]+", "[M+H+NA]2+",
    "[2M-H2O+H]+", "[2M-H+2NA]+", "[M-5H2O+H]+", "[M-4H2O+H]+",
    "[M+CH3]+", "[M+2H]+", "M+H", "[M+H-2H2O]+", "[M+H]", "[M+H+NA]+",
    "[M]+*", "[M+H-H2O]", "M+NA", "[M]++", "[M+2H]++", "M+K", "M+NH4",
    "M+2NA", "[M+H-SO3]+", "[M-NH4+2H]+", "[M+2H]2+", "[M-2H2O+2H]2+",
    "[M-3H2O+2H]2+", "[3M+NA]+", "[3M+CA-H]+", "[3M+CA]2+", "[4M+CA]2+",
    "[3M+K]+", "[2M-2H2O+H]+", "[5M+CA]2+", "[M+ACN+NH4]+", "[3M+NH4]+",
    "[M-H2O+H]", "[M+]", "M+", "[M+CH3OH+H]+", "[M+FA+H]+", "[M+H+CH3CN]+",
    "[M+NA+CH3CN]+", "[2M+H+CH3CN]+", "[M+H-NH3]+", "[M-C6H10O5+H]+",
    "M+CL", "[M+2H-NH4]+",
    # Negative Mode
    "[M-H]-", "[M-H2O-H]-", "[M+CL]-", "[M+HCOO]-", "[M+CH3COO]-",
    "M-H",
    # Special/Unknown
    "CAROTENOID", "CAROTENOIDS", "UNKNOWN", "Other"
]

class FixedMetadataEncoder:
    def __init__(self, vocab_instruments, vocab_adducts):
        self.inst_map = {k.upper(): i for i, k in enumerate(vocab_instruments)}
        self.adduct_map = {k.upper(): i for i, k in enumerate(vocab_adducts)}
        
        self.n_inst = len(vocab_instruments)
        self.n_adduct = len(vocab_adducts)
        
        self.other_inst_idx = self.inst_map.get("OTHER", len(vocab_instruments)-1)
        self.other_adduct_idx = self.adduct_map.get("OTHER", len(vocab_adducts)-1)

    def encode(self, ce_norm, inst_str, adduct_str):
        # 1. Instrument
        inst_vec = np.zeros(self.n_inst, dtype=np.float32)
        inst_key = str(inst_str).strip().upper()
        inst_vec[self.inst_map.get(inst_key, self.other_inst_idx)] = 1.0
        
        # 2. Adduct
        adduct_vec = np.zeros(self.n_adduct, dtype=np.float32)
        adduct_key = str(adduct_str).strip().upper()
        adduct_vec[self.adduct_map.get(adduct_key, self.other_adduct_idx)] = 1.0
        
        # Return: [NCE, Inst_OneHot, Adduct_OneHot]
        return np.concatenate(([ce_norm], inst_vec, adduct_vec))

# =============================================================================
# 2. DATASET CLASS
# =============================================================================

class BinaryClassificationDataset(Dataset):
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path, max_nodes=128, upsample_hard_positives=False):
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
        
        # --- NEW: HARD POSITIVE MINING ---
        if upsample_hard_positives:
            print("Running Hard Positive Mining...")
            # Fast vectorized mapping of precursor masses
            mz_map = spec_df.set_index('spec_id')['prec_mz'].to_dict()
            mz_A = self.pairs_df['name_main'].map(mz_map)
            mz_B = self.pairs_df['name_sub'].map(mz_map)
            mass_diff = (mz_A - mz_B).abs()
            
            # Find pairs that are TRUE matches (label==1) but structurally asymmetric (> 15 Da shift)
            hard_mask = (self.pairs_df['label'] == 1) & (mass_diff > 15.0)
            hard_positives = self.pairs_df[hard_mask]
            
            if len(hard_positives) > 0:
                # Oversample these difficult pairs by injecting extra copies
                num_copies = 3
                self.pairs_df = pd.concat([self.pairs_df] + [hard_positives] * num_copies)
                # Shuffle the dataset
                self.pairs_df = self.pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                print(f" -> Upsampled {len(hard_positives)} Hard Positives ({num_copies}x copies).")
                print(f" -> New total dataset size: {len(self.pairs_df)}")
            else:
                print(" -> No Hard Positives found in this split.")
        # ---------------------------------

        print("Initializing Fixed Metadata Encoder (Inst + Adduct)...")
        self.meta_encoder = FixedMetadataEncoder(FIXED_INSTRUMENTS, FIXED_ADDUCTS)
        
        # --- NCE Logic ---
        self.ce_key = "nce"
        
        # Calculate stats for normalization
        vals = spec_df[self.ce_key].dropna()
        self.mean_ce = vals.mean()
        self.std_ce = vals.std()
        if pd.isna(self.std_ce) or self.std_ce == 0: self.std_ce = 1.0
        print(f"NCE Normalization: Mean={self.mean_ce:.2f}, Std={self.std_ce:.2f}")

    def _process_ce(self, col_energy):
        val = float(col_energy) if pd.notna(col_energy) else self.mean_ce
        # Z-Score Normalization
        return (val - self.mean_ce) / (self.std_ce + EPS)

    def _get_spec_meta_fixed(self, spec_entry):
        ce_val = self._process_ce(spec_entry.get(self.ce_key))
        inst_str = spec_entry.get("inst_type", "Unknown")
        adduct_str = spec_entry.get("prec_type", "Unknown")
        
        meta_vec = self.meta_encoder.encode(ce_val, inst_str, adduct_str)
        return torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        try:
            pair_info = self.pairs_df.iloc[idx]
            id_A, id_B = pair_info['name_main'], pair_info['name_sub']
            
            if id_A not in self.valid_spec_ids or id_B not in self.valid_spec_ids:
                raise ValueError(f"Missing Spec ID")
                
            spec_A = self.spec_lookup.loc[id_A]
            spec_B = self.spec_lookup.loc[id_B]
            
            mol_id_A = spec_A['mol_id']
            mol_id_B = spec_B['mol_id']
            
            if mol_id_A not in self.valid_mol_ids or mol_id_B not in self.valid_mol_ids:
                raise ValueError(f"Missing Mol ID")
            
            mol_A = self.mol_lookup.loc[mol_id_A, 'mol']
            mol_B = self.mol_lookup.loc[mol_id_B, 'mol']
            
            if mol_A is None or mol_B is None: raise ValueError("RDKit Mol is None")
            
            graph_A = gf_preprocess(mol_A, idx)
            graph_B = gf_preprocess(mol_B, idx)
            
            if graph_A is None or graph_B is None: raise ValueError("Graph Preprocess Failed")
            
            # Size checks
            if graph_A.x.size(0) > self.max_nodes or graph_B.x.size(0) > self.max_nodes:
                 raise ValueError(f"Graph too large (> {self.max_nodes})")
            if graph_A.x.size(0) == 0 or graph_B.x.size(0) == 0:
                 raise ValueError("Graph is empty")

            spec_meta = self._get_spec_meta_fixed(spec_A)
            
            # Calculate Mass Difference
            mass_A = float(spec_A.get("prec_mz", 0.0))
            mass_B = float(spec_B.get("prec_mz", 0.0))
            mass_A_tensor = torch.tensor([mass_A], dtype=torch.float32)
            mass_B_tensor = torch.tensor([mass_B], dtype=torch.float32)
            
            # (Note: normalized mass diff is now dynamically calculated in the Stage 2 model forward pass)
            
            label = torch.tensor(pair_info['label'], dtype=torch.float32)
            
            # Return 6 items
            return graph_A, graph_B, spec_meta, mass_A_tensor, mass_B_tensor, label

        except Exception as e:
            # Fallback to random sample
            new_idx = np.random.randint(0, len(self.pairs_df))
            return self.__getitem__(new_idx)

def binary_collate_fn(batch):
    # Unpack 6 items
    graphs_A, graphs_B, spec_metas, mass_As, mass_Bs, labels = zip(*batch)
    
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    batch_meta = torch.cat(spec_metas, dim=0)
    batch_mass_A = torch.stack(mass_As, 0) # [Batch, 1]
    batch_mass_B = torch.stack(mass_Bs, 0) # [Batch, 1]
    batch_labels = torch.stack(labels, 0)
    
    return batch_A, batch_B, batch_meta, batch_mass_A, batch_mass_B, batch_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Baseline Data Loader")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    
    print("Testing with Hard Positive Mining ENABLED...")
    try:
        ds = BinaryClassificationDataset(
            args.pairs_path, 
            args.spec_data_path, 
            args.mol_data_path, 
            upsample_hard_positives=True  # Set to True here for testing
        )
        print(f"Dataset initialized. Size: {len(ds)}")
        
        # Test __getitem__
        item = ds[0]
        if len(item) == 6:
            g_a, g_b, meta, m_a, m_b, label = item
            print(f"Success! Unpacked 6 items.")
            print(f"Graph A Nodes: {g_a.x.shape[0]}, Graph B Nodes: {g_b.x.shape[0]}")
            print(f"Metadata Shape: {meta.shape}")
            print(f"Mass A Value: {m_a.item():.4f}")
            print(f"Mass B Value: {m_b.item():.4f}")
            print(f"Label: {label.item()}")
        else:
            print(f"FAILED: Expected 6 items, got {len(item)}")
            
        # Test Collate
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=binary_collate_fn)
        batch = next(iter(loader))
        b_a, b_b, b_meta, b_ma, b_mb, b_labels = batch
        print(f"Batch Graph A Keys: {b_a.keys()}")
        print(f"Batch Metadata Shape: {b_meta.shape}")
        print(f"Batch Mass A Shape: {b_ma.shape}")
        print(f"Batch Mass B Shape: {b_mb.shape}")
        print(f"Batch Labels Shape: {b_labels.shape}")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        sys.exit(1)