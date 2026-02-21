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
    from misc_utils import EPS 
except ImportError as e:
    print(f"Error importing MassFormer utils: {e}")
    sys.exit(1)

# =============================================================================
# 1. FIXED VOCABULARY DEFINITIONS (Same as standard loader)
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
# 2. IRM DATASET CLASS
# =============================================================================

class IRMBinaryDataset(Dataset):
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path, max_nodes=128):
        super().__init__()
        self.max_nodes = max_nodes
        
        print(f"Loading pairs from {pairs_feather_path}...")
        self.pairs_df = pd.read_feather(pairs_feather_path)
        
        # --- Pre-calculate Environments ---
        # Env 0: Isomers (Mass Diff < 0.01) - The "Bias" Anchor
        # Env 1: Isobaric (0.01 <= Mass Diff < 1.0) - The "Sparse" Anchor
        # Env 2: Everything else (Mass Diff >= 1.0) - The "Causal" Anchor
        
        print("Assigning IRM Environments...")
        
        # Ensure mass_difference exists
        if 'mass_difference' not in self.pairs_df.columns:
             print("Warning: 'mass_difference' column missing. Attempting to calculate or default to 2.")
             # Check if we can calculate it from metadata? For now, default to 2 (safe) if missing
             self.pairs_df['env_id'] = 2 
        else:
             self.pairs_df['env_id'] = self.pairs_df['mass_difference'].apply(self._assign_env)

        print(f"Environment Distribution (Rows per Env):")
        print(self.pairs_df['env_id'].value_counts().sort_index())

        print(f"Loading spectral data...")
        spec_df = pd.read_pickle(spec_data_path)
        self.spec_lookup = spec_df.set_index('spec_id')
        self.valid_spec_ids = set(self.spec_lookup.index)
        
        # NCE Logic
        vals = spec_df['nce'].dropna()
        self.mean_ce = vals.mean()
        self.std_ce = vals.std()
        if pd.isna(self.std_ce) or self.std_ce == 0: self.std_ce = 1.0
        
        print(f"Loading molecular data...")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        self.valid_mol_ids = set(self.mol_lookup.index)

        # Init Encoder
        self.meta_encoder = FixedMetadataEncoder(FIXED_INSTRUMENTS, FIXED_ADDUCTS)

    def _assign_env(self, md):
        # Handle bad values
        if md < 0: return 5 # Default to Huge if unknown
        
        # The 6 Regimes
        if md < 0.01: return 0  # 0. Exact Isomers
        if md < 1.0:  return 1  # 1. Isobaric
        if md < 10.0: return 2  # 2. Tiny (<10Da)
        if md < 50.0: return 3  # 3. Medium (<50Da)
        if md < 100.0: return 4 # 4. Large (<100Da)
        return 5                # 5. Huge (>100Da)                     # Standard (Causal)

    def _process_ce(self, col_energy):
        val = float(col_energy) if pd.notna(col_energy) else self.mean_ce
        return (val - self.mean_ce) / (self.std_ce + EPS)

    def _get_spec_meta_fixed(self, spec_entry):
        ce_val = self._process_ce(spec_entry.get("nce"))
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
            
            # --- Graph Loading (Standard) ---
            if id_A not in self.valid_spec_ids or id_B not in self.valid_spec_ids:
                 raise ValueError("Missing Spec ID")
            
            spec_A = self.spec_lookup.loc[id_A]
            spec_B = self.spec_lookup.loc[id_B]
            
            mol_id_A = spec_A['mol_id']
            mol_id_B = spec_B['mol_id']
            
            if mol_id_A not in self.valid_mol_ids or mol_id_B not in self.valid_mol_ids:
                raise ValueError("Missing Mol ID")

            mol_A = self.mol_lookup.loc[mol_id_A, 'mol']
            mol_B = self.mol_lookup.loc[mol_id_B, 'mol']
            
            if mol_A is None or mol_B is None: raise ValueError("RDKit Mol is None")
            
            graph_A = gf_preprocess(mol_A, idx)
            graph_B = gf_preprocess(mol_B, idx)
            
            if graph_A is None or graph_B is None: raise ValueError("Graph Preprocess Failed")
            if graph_A.x.size(0) == 0 or graph_B.x.size(0) == 0: raise ValueError("Empty Graph")

            spec_meta = self._get_spec_meta_fixed(spec_A)
            label = torch.tensor(pair_info['label'], dtype=torch.float32)
            
            # --- NEW: Return Environment ID ---
            env_id = torch.tensor(pair_info['env_id'], dtype=torch.long)
            
            # Return 5 Items
            return graph_A, graph_B, spec_meta, label, env_id

        except Exception as e:
            # Fallback to random sample
            new_idx = np.random.randint(0, len(self.pairs_df))
            return self.__getitem__(new_idx)

def irm_collate_fn(batch):
    # Unpack 5 items
    graphs_A, graphs_B, spec_metas, labels, env_ids = zip(*batch)
    
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    batch_meta = torch.cat(spec_metas, dim=0)
    batch_labels = torch.stack(labels, 0)
    
    # Stack Environments
    batch_envs = torch.stack(env_ids, 0)
    
    return batch_A, batch_B, batch_meta, batch_labels, batch_envs

# =============================================================================
# 3. TEST BLOCK (Runs when file is executed directly)
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test IRM Data Loader")
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    
    print("\n--- Testing IRM Data Loader ---")
    try:
        # 1. Initialize
        ds = IRMBinaryDataset(args.pairs_path, args.spec_data_path, args.mol_data_path)
        print(f"Dataset initialized successfully. Total Size: {len(ds)}")
        
        # 2. Test __getitem__
        print("\n--- Testing Single Item Fetch ---")
        item = ds[0]
        if len(item) == 5:
            g_a, g_b, meta, label, env = item
            print(f"Success! Unpacked 5 items.")
            print(f"Graph A Nodes: {g_a.x.shape}")
            print(f"Metadata Shape: {meta.shape}")
            print(f"Label: {label.item()}")
            print(f"Environment ID: {env.item()} (0=Isomer, 1=Isobar, 2=Other)")
        else:
            print(f"FAILED: Expected 5 items, got {len(item)}")
            sys.exit(1)
            
        # 3. Test Collate & Batching
        print(f"\n--- Testing Batch Loading (Batch Size: {args.batch_size}) ---")
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=irm_collate_fn, shuffle=True)
        batch = next(iter(loader))
        
        b_a, b_b, b_meta, b_labels, b_envs = batch
        
        print(f"Batch Graph A: {b_a.keys()}") # Should verify keys exist
        print(f"Batch Metadata: {b_meta.shape}") # [B, MetaDim]
        print(f"Batch Labels: {b_labels.shape}") # [B]
        print(f"Batch Environments: {b_envs.shape}") # [B]
        
        print(f"Environments found in this batch: {b_envs.tolist()}")
        
        unique_envs = torch.unique(b_envs)
        print(f"Unique Envs in batch: {unique_envs.tolist()}")
        
        if len(unique_envs) < 2:
            print("Note: This random batch only contained one environment type. "
                  "Ideally, IRM batches should contain mixed environments.")
        
        print("\nPASSED: IRM Data Loader is ready for training.")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)