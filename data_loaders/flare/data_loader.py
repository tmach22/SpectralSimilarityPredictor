import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import BRICS

# =============================================================================
# SETUP PATHS
# =============================================================================
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
except ImportError as e:
    print(f"[-] Error importing MassFormer utils: {e}")
    sys.exit(1)

try:
    from bde_edge_encoder import compute_bde_adjacency
    _BDE_AVAILABLE = True
except ImportError:
    _BDE_AVAILABLE = False
    print("[!] Warning: bde_edge_encoder not found — BDE features will be zeros.")


# =============================================================================
# BRICS ADJACENCY (backward compat — returns plain BRICS weights)
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
# CROSS-MODAL DATASET  (now also returns A_bde_norm and ce_norm per spectrum)
# =============================================================================
class CrossModalPretrainDataset(Dataset):
    def __init__(self, spec_data_path, mol_data_path, max_peaks=60, parent_intensity=1.1):
        super().__init__()
        self.max_peaks = max_peaks
        self.parent_intensity = parent_intensity

        print(f"[*] Loading Spectral Data from: {spec_data_path}")
        self.spec_df = pd.read_pickle(spec_data_path).reset_index(drop=True)

        print(f"[*] Loading Molecular Data from: {mol_data_path}")
        mol_df = pd.read_pickle(mol_data_path)
        self.mol_lookup = mol_df.set_index('mol_id')
        self.valid_mol_ids = set(self.mol_lookup.index)

        # CE normalization stats
        nce_col = 'nce' if 'nce' in self.spec_df.columns else None
        if nce_col:
            vals = self.spec_df[nce_col].dropna()
            self.mean_ce = float(vals.mean())
            self.std_ce = float(vals.std()) if float(vals.std()) > 0 else 1.0
        else:
            self.mean_ce, self.std_ce = 40.0, 20.0

        self.spec_df = self.spec_df[
            self.spec_df['mol_id'].isin(self.valid_mol_ids)
        ].reset_index(drop=True)
        print(f"[+] Dataset initialized with {len(self.spec_df)} valid molecule-spectrum pairs.")

    def __len__(self):
        return len(self.spec_df)

    def _process_peaks(self, peaks_list, prec_mz):
        if not isinstance(peaks_list, list) or len(peaks_list) == 0:
            return torch.zeros((self.max_peaks, 4)), torch.ones(self.max_peaks, dtype=torch.bool)
        peaks_list = sorted(peaks_list, key=lambda x: x[1], reverse=True)[:self.max_peaks - 1]
        peak_features = []
        for mz, intensity in peaks_list:
            neutral_loss = prec_mz - mz
            relative_mz = mz / prec_mz if prec_mz > 0 else 0.0
            peak_features.append([mz / 1000.0, float(intensity), neutral_loss / 1000.0, relative_mz])
        peak_features.append([prec_mz / 1000.0, self.parent_intensity, 0.0, 1.0])
        features_tensor = torch.tensor(peak_features, dtype=torch.float32)
        num_actual = features_tensor.shape[0]
        pad_length = self.max_peaks - num_actual
        if pad_length > 0:
            features_tensor = torch.cat(
                [features_tensor, torch.zeros((pad_length, 4), dtype=torch.float32)], dim=0
            )
        mask = torch.zeros(self.max_peaks, dtype=torch.bool)
        if pad_length > 0:
            mask[-pad_length:] = True
        return features_tensor, mask

    def _get_ce_norm(self, row):
        """Return normalized CE as a [1] tensor."""
        ce_val = float(row.get('nce', self.mean_ce)) if hasattr(row, 'get') else self.mean_ce
        if np.isnan(ce_val):
            ce_val = self.mean_ce
        return torch.tensor([(ce_val - self.mean_ce) / (self.std_ce + 1e-8)], dtype=torch.float32)

    def __getitem__(self, idx):
        try:
            row = self.spec_df.iloc[idx]
            mol_id = row['mol_id']
            raw_peaks = row['peaks']
            prec_mz = float(row['prec_mz'])

            # 1. Process spectral peaks
            padded_peaks, peak_mask = self._process_peaks(raw_peaks, prec_mz)

            # 2. CE normalization
            ce_norm = self._get_ce_norm(row)

            # 3. Extract RDKit mol
            mol = self.mol_lookup.loc[mol_id, 'mol']
            if mol is None:
                raise ValueError("RDKit Mol is None")

            # 4. Graph preprocessing
            graph = gf_preprocess(mol, idx)
            if graph is None:
                raise ValueError("Graph Preprocess Failed")

            # 5. BRICS adjacency (original)
            A_brics = get_brics_adjacency(mol)

            # 6. BDE adjacency (new: A_bde_norm in [0,1])
            if _BDE_AVAILABLE:
                A_bde_norm, _, _ = compute_bde_adjacency(mol)
            else:
                A_bde_norm = torch.zeros_like(A_brics)

            # Returns: graph, A_brics, A_bde_norm, peaks, peak_mask, ce_norm
            return graph, A_brics, A_bde_norm, padded_peaks, peak_mask, ce_norm

        except Exception:
            new_idx = np.random.randint(0, len(self.spec_df))
            return self.__getitem__(new_idx)


# =============================================================================
# COLLATE FUNCTION  (handles new A_bde_norm and ce_norm fields)
# =============================================================================
def cross_modal_collate_fn(batch):
    valid_graphs = []
    valid_A_brics = []
    valid_A_bde = []
    valid_peaks = []
    valid_masks = []
    valid_ce = []

    for item in batch:
        if item is not None and item[0] is not None and item[1] is not None:
            num_atoms = item[1].size(0)
            if 0 < num_atoms <= 100:
                valid_graphs.append(item[0])
                valid_A_brics.append(item[1])
                valid_A_bde.append(item[2])
                valid_peaks.append(item[3])
                valid_masks.append(item[4])
                valid_ce.append(item[5])

    if len(valid_graphs) == 0:
        return None, None, None, None, None, None

    batched_graphs = collator(valid_graphs)
    target_len = max([a.size(0) for a in valid_A_brics])

    # Pad A_brics and A_bde to same target_len
    padded_A_brics = []
    padded_A_bde = []
    for a_b, a_d in zip(valid_A_brics, valid_A_bde):
        n = a_b.size(0)
        pad = target_len - n

        pa_b = F.pad(a_b, (0, pad, 0, pad), value=0.0)
        if pad > 0:
            pa_b[n:, n:] = torch.eye(pad)
        padded_A_brics.append(pa_b)

        pa_d = F.pad(a_d, (0, pad, 0, pad), value=0.0)
        if pad > 0:
            pa_d[n:, n:] = torch.eye(pad)   # self-loop BDE = 1.0 (max, no cut)
        padded_A_bde.append(pa_d)

    batch_A_brics = torch.stack(padded_A_brics, dim=0)
    batch_A_bde = torch.stack(padded_A_bde, dim=0)
    batch_peaks = torch.stack(valid_peaks, dim=0)
    batch_masks = torch.stack(valid_masks, dim=0)
    batch_ce = torch.stack(valid_ce, dim=0)   # [B, 1]

    return batched_graphs, batch_A_brics, batch_A_bde, batch_peaks, batch_masks, batch_ce


# =============================================================================
# TESTING BLOCK
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print("\n--- Testing CrossModalPretrainDataset (with BDE + CE) ---")
    dataset = CrossModalPretrainDataset(args.spec_data_path, args.mol_data_path)

    if len(dataset) > 0:
        print("\n--- Testing DataLoader & Collation ---")
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            collate_fn=cross_modal_collate_fn, shuffle=True
        )
        batch = next(iter(loader))
        b_graphs, b_A_brics, b_A_bde, b_peaks, b_masks, b_ce = batch

        print(f"Batch Graph Keys: {list(b_graphs.keys())}")
        print(f"Batch A_brics Shape: {b_A_brics.shape}")
        print(f"Batch A_bde Shape:   {b_A_bde.shape}")
        print(f"Batch Peaks Shape:   {b_peaks.shape}")
        print(f"Batch CE Shape:      {b_ce.shape}")
        print("[+] SUCCESS!")
