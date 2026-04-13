import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import os
from pathlib import Path
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

# Setup sys.path to find massformer src
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

# BDE encoder (new)
_flare_model_dir = os.path.join(str(cwd), 'model', 'flare')
if _flare_model_dir not in sys.path:
    sys.path.insert(0, _flare_model_dir)

try:
    from gf_data_utils import gf_preprocess, collator
    from misc_utils import np_one_hot, EPS
except ImportError as e:
    print(f"Error importing MassFormer utils: {e}")
    sys.exit(1)

try:
    from bde_edge_encoder import compute_bde_adjacency
    _BDE_AVAILABLE = True
except ImportError:
    _BDE_AVAILABLE = False
    print("[!] Warning: bde_edge_encoder not found — BDE features will be zeros.")


# =============================================================================
# BRICS ADJACENCY (backward compat)
# =============================================================================
def get_brics_adjacency(mol):
    N = mol.GetNumAtoms()
    A = np.zeros((N, N), dtype=np.float32)
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    cleavage_pairs = set([tuple(sorted(b[0])) for b in brics_bonds])
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        w = 1.0 if tuple(sorted((u, v))) in cleavage_pairs else 10.0
        A[u, v] = w
        A[v, u] = w
    np.fill_diagonal(A, 10.0)
    return torch.tensor(A, dtype=torch.float32)


# =============================================================================
# SHIFT FRACTION ESTIMATOR
# Estimates what fraction of fragment ions would shift by Δm for a given pair.
# For a simple estimate: fraction of bonds that are "downstream" of the
# modification site. Without atom mapping, we use the mass-weighted proxy:
# f_shift ≈ (lighter molecule mass) / (heavier molecule mass)
# This is a conservative lower bound.
# =============================================================================
def estimate_shift_fraction(mass_A, mass_B):
    """
    Proxy estimate: the lighter molecule's precursor mass / heavier molecule's mass.
    Rationale: if molecule A is lighter by Δm, the fragments that retained the
    extra group (Δm worth) will shift. In the worst case (modification at the
    core), f ≈ 1.0; at the periphery, f ≈ Δm/total_mass.

    Returns shift_fraction in [0, 1].
    """
    m_heavy = max(mass_A, mass_B)
    m_light = min(mass_A, mass_B)
    delta = m_heavy - m_light
    if m_heavy < 1e-3:
        return 0.0
    # Fraction ≈ delta / heavy_mass, capped at 1
    return float(np.clip(delta / m_heavy, 0.0, 1.0))


# =============================================================================
# FIXED VOCABULARY DEFINITIONS
# =============================================================================
FIXED_INSTRUMENTS = [
    "QTOF", "Orbitrap", "Triple Quad", "Ion Trap", "Thermo Q Exactive HF",
    "Unknown", "None", "Other"
]

FIXED_ADDUCTS = [
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
    "[M-H]-", "[M-H2O-H]-", "[M+CL]-", "[M+HCOO]-", "[M+CH3COO]-",
    "M-H",
    "CAROTENOID", "CAROTENOIDS", "UNKNOWN", "Other"
]

# Common mass deltas for hard negative mining (Da)
SHIFT_HARD_NEG_DELTAS = [14.016, 15.995, 28.031, 42.011, 15.024]


class FixedMetadataEncoder:
    def __init__(self, vocab_instruments, vocab_adducts):
        self.inst_map = {k.upper(): i for i, k in enumerate(vocab_instruments)}
        self.adduct_map = {k.upper(): i for i, k in enumerate(vocab_adducts)}
        self.n_inst = len(vocab_instruments)
        self.n_adduct = len(vocab_adducts)
        self.other_inst_idx = self.inst_map.get("OTHER", len(vocab_instruments) - 1)
        self.other_adduct_idx = self.adduct_map.get("OTHER", len(vocab_adducts) - 1)

    def encode(self, ce_norm, inst_str, adduct_str):
        inst_vec = np.zeros(self.n_inst, dtype=np.float32)
        inst_key = str(inst_str).strip().upper()
        inst_vec[self.inst_map.get(inst_key, self.other_inst_idx)] = 1.0

        adduct_vec = np.zeros(self.n_adduct, dtype=np.float32)
        adduct_key = str(adduct_str).strip().upper()
        adduct_vec[self.adduct_map.get(adduct_key, self.other_adduct_idx)] = 1.0

        return np.concatenate(([ce_norm], inst_vec, adduct_vec))


# =============================================================================
# DATASET CLASS  (now also returns A_bde_norm, shift_fraction, ce_norm)
# =============================================================================
class BinaryClassificationDataset(Dataset):
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path,
                 max_nodes=128, upsample_hard_negatives=False):
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

        # Build mz map for hard negative mining
        mz_map = spec_df.set_index('spec_id')['prec_mz'].to_dict()

        if upsample_hard_negatives:
            print("Running Hard Negative Mining for OT Training...")
            mz_A = self.pairs_df['name_main'].map(mz_map)
            mz_B = self.pairs_df['name_sub'].map(mz_map)
            mass_diff = (mz_A - mz_B).abs()

            # Original isobaric hard negatives (≤1 Da)
            isobaric_mask = (self.pairs_df['label'] == 0) & (mass_diff <= 1.0)
            hard_negatives = self.pairs_df[isobaric_mask]
            copies = []
            if len(hard_negatives) > 0:
                copies.extend([hard_negatives] * 3)
                print(f" -> Upsampled {len(hard_negatives)} Isobaric Hard Negatives (3x copies).")

            # NEW: shift hard negatives — pairs near common modification deltas
            for delta in SHIFT_HARD_NEG_DELTAS:
                shift_mask = (
                    (self.pairs_df['label'] == 0) &
                    (mass_diff >= delta - 0.05) &
                    (mass_diff <= delta + 0.05)
                )
                shift_negs = self.pairs_df[shift_mask]
                if len(shift_negs) > 0:
                    copies.extend([shift_negs] * 2)
                    print(f" -> Upsampled {len(shift_negs)} Shift Hard Negatives "
                          f"(Δm≈{delta:.1f} Da, 2x copies).")

            if copies:
                self.pairs_df = pd.concat([self.pairs_df] + copies)
                self.pairs_df = self.pairs_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        print(f" -> Final dataset size: {len(self.pairs_df)}")

        print("Initializing Fixed Metadata Encoder (Inst + Adduct)...")
        self.meta_encoder = FixedMetadataEncoder(FIXED_INSTRUMENTS, FIXED_ADDUCTS)

        self.ce_key = "nce"
        vals = spec_df[self.ce_key].dropna()
        self.mean_ce = vals.mean()
        self.std_ce = vals.std()
        if pd.isna(self.std_ce) or self.std_ce == 0:
            self.std_ce = 1.0

    def _process_ce(self, col_energy):
        val = float(col_energy) if pd.notna(col_energy) else self.mean_ce
        return (val - self.mean_ce) / (self.std_ce + EPS)

    def _get_spec_meta_fixed(self, spec_entry):
        ce_val = self._process_ce(spec_entry.get(self.ce_key))
        inst_str = spec_entry.get("inst_type", "Unknown")
        adduct_str = spec_entry.get("prec_type", "Unknown")
        meta_vec = self.meta_encoder.encode(ce_val, inst_str, adduct_str)
        return torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0)

    def _get_ce_norm(self, spec_entry):
        """Return raw normalized CE as a [1] tensor (separate from the metadata vector)."""
        ce_val = self._process_ce(spec_entry.get(self.ce_key))
        return torch.tensor([ce_val], dtype=torch.float32)

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        try:
            pair_info = self.pairs_df.iloc[idx]
            id_A, id_B = pair_info['name_main'], pair_info['name_sub']

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

            if mol_A is None or mol_B is None:
                raise ValueError("RDKit Mol is None")

            graph_A = gf_preprocess(mol_A, idx)
            graph_B = gf_preprocess(mol_B, idx)

            if graph_A is None or graph_B is None:
                raise ValueError("Graph Preprocess Failed")

            if graph_A.x.size(0) > self.max_nodes or graph_B.x.size(0) > self.max_nodes:
                raise ValueError(f"Graph too large (> {self.max_nodes})")
            if graph_A.x.size(0) == 0 or graph_B.x.size(0) == 0:
                raise ValueError("Graph is empty")

            spec_meta = self._get_spec_meta_fixed(spec_A)
            ce_norm = self._get_ce_norm(spec_A)   # [1] — used for CE conditioning

            mass_A = float(spec_A.get("prec_mz", 0.0))
            mass_B = float(spec_B.get("prec_mz", 0.0))
            mass_A_tensor = torch.tensor([mass_A], dtype=torch.float32)
            mass_B_tensor = torch.tensor([mass_B], dtype=torch.float32)

            label = torch.tensor(pair_info['label'], dtype=torch.float32)

            # BRICS adjacency (original)
            A_brics_A = get_brics_adjacency(mol_A)
            A_brics_B = get_brics_adjacency(mol_B)

            # BDE adjacency (new)
            if _BDE_AVAILABLE:
                A_bde_A, _, _ = compute_bde_adjacency(mol_A)
                A_bde_B, _, _ = compute_bde_adjacency(mol_B)
            else:
                A_bde_A = torch.zeros_like(A_brics_A)
                A_bde_B = torch.zeros_like(A_brics_B)

            # Shift fraction (new)
            shift_frac = torch.tensor(
                [estimate_shift_fraction(mass_A, mass_B)], dtype=torch.float32
            )

            return (graph_A, graph_B, spec_meta, mass_A_tensor, mass_B_tensor,
                    label, A_brics_A, A_brics_B, A_bde_A, A_bde_B,
                    shift_frac, ce_norm)

        except Exception:
            new_idx = np.random.randint(0, len(self.pairs_df))
            return self.__getitem__(new_idx)


# =============================================================================
# COLLATE FUNCTION  (handles new fields: A_bde_A/B, shift_frac, ce_norm)
# =============================================================================
def binary_collate_fn(batch):
    (graphs_A, graphs_B, spec_metas, mass_As, mass_Bs, labels,
     A_brics_A_list, A_brics_B_list,
     A_bde_A_list, A_bde_B_list,
     shift_fracs, ce_norms) = zip(*batch)

    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)

    batch_meta = torch.cat(spec_metas, dim=0)
    batch_mass_A = torch.stack(mass_As, 0)
    batch_mass_B = torch.stack(mass_Bs, 0)
    batch_labels = torch.stack(labels, 0)
    batch_shift_frac = torch.stack(shift_fracs, 0)    # [B, 1]
    batch_ce_norm = torch.stack(ce_norms, 0)           # [B, 1]

    def _pad_adj_list(adj_list):
        max_len = max(a.size(0) for a in adj_list)
        padded = []
        for a in adj_list:
            n = a.size(0)
            pad = max_len - n
            pa = F.pad(a, (0, pad, 0, pad), value=0.0)
            if pad > 0:
                pa[n:, n:] = torch.eye(pad)
            padded.append(pa)
        return torch.stack(padded, dim=0)

    batch_A_brics_A = _pad_adj_list(A_brics_A_list)
    batch_A_brics_B = _pad_adj_list(A_brics_B_list)
    batch_A_bde_A = _pad_adj_list(A_bde_A_list)
    batch_A_bde_B = _pad_adj_list(A_bde_B_list)

    return (batch_A, batch_B, batch_meta, batch_mass_A, batch_mass_B,
            batch_labels, batch_A_brics_A, batch_A_brics_B,
            batch_A_bde_A, batch_A_bde_B,
            batch_shift_frac, batch_ce_norm)


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

    ds = BinaryClassificationDataset(
        pairs_feather_path=args.pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=True
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=binary_collate_fn, shuffle=True)
    batch = next(iter(loader))
    (b_a, b_b, b_meta, b_ma, b_mb, b_labels,
     b_brics_a, b_brics_b, b_bde_a, b_bde_b, b_shift, b_ce) = batch

    print(f"Labels: {b_labels.shape}")
    print(f"BRICS A: {b_brics_a.shape}, BDE A: {b_bde_a.shape}")
    print(f"Shift fractions: {b_shift.squeeze()}")
    print(f"CE norm: {b_ce.squeeze()}")
    print("[+] SUCCESS!")
