import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import multiprocessing as mp
from tqdm import tqdm
import os
import itertools

# ==========================================
# 1. BDE HEURISTIC CONSTANTS
# ==========================================
_BDE_LOOKUP = {
    ('C', 'C', 1): 83.0,  ('C', 'C', 2): 146.0, ('C', 'C', 3): 200.0, ('C', 'C', 1.5): 115.0,
    ('C', 'N', 1): 73.0,  ('C', 'N', 2): 147.0, ('C', 'N', 1.5): 100.0,
    ('C', 'O', 1): 86.0,  ('C', 'O', 2): 178.0,
    ('C', 'S', 1): 65.0,  ('C', 'S', 2): 107.0,
    ('C', 'F', 1): 116.0, ('C', 'Cl', 1): 81.0, ('C', 'Br', 1): 68.0, ('C', 'I', 1): 51.0,
    ('C', 'H', 1): 99.0,
    ('N', 'N', 1): 38.0,  ('N', 'N', 2): 100.0,
    ('N', 'O', 1): 46.0,  ('N', 'H', 1): 93.0,
    ('O', 'O', 1): 35.0,  ('O', 'H', 1): 119.0,
    ('S', 'S', 1): 54.0,  ('S', 'H', 1): 82.0,
    ('P', 'O', 1): 86.0,  ('P', 'O', 2): 140.0,
}
_BDE_MIN = 20.0

def _get_bond_order_key(bond):
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: return 1
    elif bt == Chem.rdchem.BondType.DOUBLE: return 2
    elif bt == Chem.rdchem.BondType.TRIPLE: return 3
    elif bt == Chem.rdchem.BondType.AROMATIC: return 1.5
    return 1 

def _context_correction(mol, bond):
    correction = 0.0
    u_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    v_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
    
    is_benzylic, is_allylic, is_tertiary, is_alpha_heteroatom, is_amide = False, False, False, False, False

    if (u_atom.GetSymbol() == 'C' and v_atom.GetSymbol() == 'N') or \
       (u_atom.GetSymbol() == 'N' and v_atom.GetSymbol() == 'C'):
        c_atom = u_atom if u_atom.GetSymbol() == 'C' else v_atom
        if any(nbr.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE for nbr in c_atom.GetNeighbors()):
            is_amide = True

    for atom in [u_atom, v_atom]:
        if atom.GetSymbol() != 'C': continue
        if any(mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetIsAromatic() for nbr in atom.GetNeighbors()):
            is_benzylic = True
        if any(mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE for nbr in atom.GetNeighbors()):
            is_allylic = True
        if sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == 'C') >= 3:
            is_tertiary = True
        if any(nbr.GetSymbol() in ['N', 'O'] and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE for nbr in atom.GetNeighbors()):
            is_alpha_heteroatom = True

    if is_benzylic: correction -= 17.0
    elif is_allylic: correction -= 12.0
    if is_tertiary: correction -= 10.0
    if is_alpha_heteroatom: correction -= 15.0
    if is_amide: correction += 12.0 

    return correction

# ==========================================
# 2. PHYSICS & TARGET LOGIC
# ==========================================
def predict_bdes_for_molecule(mol):
    bde_dict = {}
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        sym_u = mol.GetAtomWithIdx(u).GetSymbol()
        sym_v = mol.GetAtomWithIdx(v).GetSymbol()
        bond_order = _get_bond_order_key(bond)

        bde = _BDE_LOOKUP.get((sym_u, sym_v, bond_order),
              _BDE_LOOKUP.get((sym_v, sym_u, bond_order), 83.0))
        
        bde += _context_correction(mol, bond)
        bde_dict[bond.GetIdx()] = max(bde, _BDE_MIN)
        
    return bde_dict

def generate_cleavage_labels(mol, spectrum_mz_array, mass_tolerance=0.01):
    labels = {bond.GetIdx(): 0.0 for bond in mol.GetBonds()}
    PROTON_MASS = 1.007276
    HYDROGEN_MASS = 1.007825 

    def check_match(fragment):
        exact_mass = rdMolDescriptors.CalcExactMolWt(fragment)
        mz_options = [
            exact_mass + PROTON_MASS, 
            exact_mass + PROTON_MASS + HYDROGEN_MASS, 
            exact_mass + PROTON_MASS - HYDROGEN_MASS
        ]
        for frag_mz in mz_options:
            mass_diffs = np.abs(spectrum_mz_array - frag_mz)
            if (mass_diffs <= mass_tolerance).any():
                return True
        return False

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        if bond.IsInRing(): continue
        try:
            fragmented_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
            fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
            if any(check_match(frag) for frag in fragments):
                labels[bond_idx] = 1.0
        except Exception: pass

    ring_info = mol.GetRingInfo()
    for bond_ring in ring_info.BondRings():
        is_aromatic = any(mol.GetBondWithIdx(idx).GetIsAromatic() for idx in bond_ring)
        if is_aromatic: continue
        best_pair = None
        for b1, b2 in itertools.combinations(bond_ring, 2):
            try:
                fragmented_mol = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=False)
                fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
                if len(fragments) < 2: continue
                if any(check_match(frag) for frag in fragments):
                    best_pair = (b1, b2)
                    break 
            except Exception: pass
        if best_pair:
            labels[best_pair[0]] = 1.0
            labels[best_pair[1]] = 1.0
                
    return labels

# ==========================================
# 3. [NEW] SPECTRAL PEAK PADDING
# ==========================================
def _process_peaks_to_lists(peaks_list, prec_mz, max_peaks=60, parent_intensity=1.1):
    """
    Pads peaks and returns standard Python lists (not tensors) 
    to prevent multiprocessing memory leaks.
    """
    if not isinstance(peaks_list, list) or len(peaks_list) == 0:
        return [[0.0, 0.0, 0.0, 0.0]] * max_peaks, [True] * max_peaks
        
    # Sort by intensity (highest first) and take top (max_peaks - 1)
    peaks_list = sorted(peaks_list, key=lambda x: x[1], reverse=True)[:max_peaks - 1]
    
    peak_features = []
    for mz, intensity in peaks_list:
        neutral_loss = prec_mz - mz
        relative_mz = mz / prec_mz if prec_mz > 0 else 0.0
        peak_features.append([mz / 1000.0, float(intensity), neutral_loss / 1000.0, relative_mz])
        
    # Always append the precursor peak
    peak_features.append([prec_mz / 1000.0, parent_intensity, 0.0, 1.0])
    
    pad_length = max_peaks - len(peak_features)
    mask = [False] * len(peak_features)
    
    if pad_length > 0:
        peak_features.extend([[0.0, 0.0, 0.0, 0.0]] * pad_length)
        mask.extend([True] * pad_length)
        
    return peak_features, mask

# ==========================================
# 4. WORKER FUNCTION FOR MULTIPROCESSING
# ==========================================
def process_molecule_row(row):
    smiles = row['smiles']
    
    raw_peaks = row['peaks']
    prec_mz = float(row.get('prec_mz', 0.0))
    ce = row.get('collision_energy', 35.0) 
    
    # [UPDATED] Process and pad the peaks here
    padded_peaks, peak_mask = _process_peaks_to_lists(raw_peaks, prec_mz)
    
    peaks_array = np.array(raw_peaks) 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
        
    bde_dict = predict_bdes_for_molecule(mol)
    mz_array = peaks_array[:, 0] if len(peaks_array) > 0 else np.array([])
    cleavage_dict = generate_cleavage_labels(mol, mz_array)
    
    edge_index, edge_attr_bde, edge_label_cleavage = [], [], []
    
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_idx = bond.GetIdx()
        
        for src, dst in [(u, v), (v, u)]:
            edge_index.append([src, dst])
            edge_attr_bde.append([bde_dict[bond_idx], ce])
            edge_label_cleavage.append(cleavage_dict[bond_idx])
            
    if not edge_index: return None 

    # [UPDATED] Return the peaks in the dictionary
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr_bde,
        'y_cleavage': edge_label_cleavage,
        'num_nodes': mol.GetNumAtoms(),
        'smiles': smiles,
        'peaks': padded_peaks,
        'peak_mask': peak_mask
    }

# ==========================================
# 5. MASTER EXECUTION LOOP
# ==========================================
if __name__ == '__main__':
    spec_df_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df_COMBINED.pkl"
    mol_df_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df_COMBINED.pkl"
    
    # [UPDATED] Save as phase2_graphs.pt so we don't overwrite Phase 1
    output_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/phase2_graphs.pt"

    print("[*] Loading DataFrames...")
    spec_df = pd.read_pickle(spec_df_path)
    mol_df = pd.read_pickle(mol_df_path)
    merged_df = pd.merge(spec_df, mol_df, on='mol_id')
    
    rows = [row for _, row in merged_df.iterrows()]
    
    num_cores = max(1, mp.cpu_count() - 2) 
    print(f"[*] Beginning parallel processing of {len(rows)} molecules using {num_cores} cores...")
    
    with mp.Pool(num_cores) as pool:
        raw_results = list(tqdm(pool.imap(process_molecule_row, rows), total=len(rows)))
        
    print(f"[*] Multiprocessing complete. Assembling PyTorch tensors...")
    
    valid_graphs = []
    
    for res in tqdm(raw_results, desc="Building PyG Graphs"):
        if res is not None:
            graph = Data(
                edge_index=torch.tensor(res['edge_index'], dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(res['edge_attr'], dtype=torch.float),
                y_cleavage=torch.tensor(res['y_cleavage'], dtype=torch.float),
                num_nodes=res['num_nodes'],
                smiles=res['smiles'],
                # [UPDATED] Attach the peak tensors to the graph object
                peaks=torch.tensor(res['peaks'], dtype=torch.float32),
                peak_mask=torch.tensor(res['peak_mask'], dtype=torch.bool)
            )
            valid_graphs.append(graph)
    
    print(f"[*] Successfully processed {len(valid_graphs)}/{len(rows)} valid graphs.")
    print(f"[*] Saving massive graph dataset to {output_path}...")
    torch.save(valid_graphs, output_path)
    print("[+] Done! You are ready for Phase 2 REINFORCE Training.")