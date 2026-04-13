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
    """Map RDKit bond type to the float key used in BDE_LOOKUP."""
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: return 1
    elif bt == Chem.rdchem.BondType.DOUBLE: return 2
    elif bt == Chem.rdchem.BondType.TRIPLE: return 3
    elif bt == Chem.rdchem.BondType.AROMATIC: return 1.5
    return 1  # fallback

def _context_correction(mol, bond):
    """
    Applies the FULL thermodynamic penalty if the resulting fragment 
    benefits from resonance, inductive stabilization, or MS/MS charge direction.
    """
    correction = 0.0
    u_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    v_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
    
    is_benzylic, is_allylic, is_tertiary, is_alpha_heteroatom, is_amide = False, False, False, False, False

    # 1. Check for Amide Bond
    if (u_atom.GetSymbol() == 'C' and v_atom.GetSymbol() == 'N') or \
       (u_atom.GetSymbol() == 'N' and v_atom.GetSymbol() == 'C'):
        c_atom = u_atom if u_atom.GetSymbol() == 'C' else v_atom
        if any(nbr.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE for nbr in c_atom.GetNeighbors()):
            is_amide = True

    # 2. Evaluate context for both atoms
    for atom in [u_atom, v_atom]:
        if atom.GetSymbol() != 'C': continue

        # Benzylic / Allylic
        if any(mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetIsAromatic() for nbr in atom.GetNeighbors()):
            is_benzylic = True
        if any(mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE for nbr in atom.GetNeighbors()):
            is_allylic = True
            
        # Tertiary
        if sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == 'C') >= 3:
            is_tertiary = True

        # MS/MS Alpha-Cleavage
        if any(nbr.GetSymbol() in ['N', 'O'] and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE for nbr in atom.GetNeighbors()):
            is_alpha_heteroatom = True

    # 3. Apply Penalties
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
    """Calculates heuristic BDE for every bond in the molecule."""
    bde_dict = {}
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        sym_u = mol.GetAtomWithIdx(u).GetSymbol()
        sym_v = mol.GetAtomWithIdx(v).GetSymbol()
        bond_order = _get_bond_order_key(bond)

        # Lookup BDE (try both directions, fallback to C-C single)
        bde = _BDE_LOOKUP.get((sym_u, sym_v, bond_order),
              _BDE_LOOKUP.get((sym_v, sym_u, bond_order), 83.0))
        
        bde += _context_correction(mol, bond)
        bde_dict[bond.GetIdx()] = max(bde, _BDE_MIN)
        
    return bde_dict

def generate_cleavage_labels(mol, spectrum_mz_array, mass_tolerance=0.01):
    """
    Simulates primary cleavage and non-aromatic ring opening.
    Matches against LC-ESI mass spectrum allowing for Hydrogen rearrangements.
    Strictly isolates 1-cut acyclic and 2-cut ring logic.
    """
    # Initialize all bonds to 0.0 (intact)
    labels = {bond.GetIdx(): 0.0 for bond in mol.GetBonds()}
    
    PROTON_MASS = 1.007276
    HYDROGEN_MASS = 1.007825 

    def check_match(fragment):
        """Helper function to check if a fragment mass matches the spectrum."""
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

    # ==========================================
    # PROCESS 1: ACYCLIC BONDS (Strictly 1-Cut Logic)
    # ==========================================
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        if bond.IsInRing():
            continue
            
        try:
            # Snap the single bond
            fragmented_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
            fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
            
            # If the break yields a match, flag the edge
            if any(check_match(frag) for frag in fragments):
                labels[bond_idx] = 1.0
        except Exception:
            pass

    # ==========================================
    # PROCESS 2: SATURATED RINGS (Strictly 2-Cut Logic)
    # ==========================================
    ring_info = mol.GetRingInfo()
    for bond_ring in ring_info.BondRings():
        # Only evaluate rings that are entirely non-aromatic
        is_aromatic = any(mol.GetBondWithIdx(idx).GetIsAromatic() for idx in bond_ring)
        if is_aromatic:
            continue
            
        # Try breaking every possible pair of bonds within this specific ring
        best_pair = None
        
        for b1, b2 in itertools.combinations(bond_ring, 2):
            try:
                # Snap exactly two bonds
                fragmented_mol = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=False)
                fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
                
                # Crucial Physics Check: Did breaking these two bonds actually split the molecule?
                # (If it's a fused ring system, breaking 2 bonds might not be enough to separate a piece)
                if len(fragments) < 2: 
                    continue
                
                if any(check_match(frag) for frag in fragments):
                    best_pair = (b1, b2)
                    break # Stop at the first valid pair to prevent "atomizing" the ring
                    
            except Exception:
                pass
                
        # If we found a valid 2-bond break that explains a peak, flag those two specific edges
        if best_pair:
            labels[best_pair[0]] = 1.0
            labels[best_pair[1]] = 1.0
                
    return labels

# ==========================================
# 3. WORKER FUNCTION FOR MULTIPROCESSING
# ==========================================
def process_molecule_row(row):
    smiles = row['smiles']
    
    # Force list of lists to NumPy array so slicing works
    peaks = np.array(row['peaks']) 
    ce = row.get('collision_energy', 35.0) 
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
        
    # Crucial: Add explicit hydrogens so exact mass and context scripts work perfectly
    # mol = Chem.AddHs(mol)
    
    bde_dict = predict_bdes_for_molecule(mol)
    mz_array = peaks[:, 0]
    cleavage_dict = generate_cleavage_labels(mol, mz_array)
    
    edge_index, edge_attr_bde, edge_label_cleavage = [], [], []
    
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_idx = bond.GetIdx()
        
        # PyG requires directed edges for undirected graphs (u->v and v->u)
        for src, dst in [(u, v), (v, u)]:
            edge_index.append([src, dst])
            edge_attr_bde.append([bde_dict[bond_idx], ce]) # Edge features: [BDE, CE]
            edge_label_cleavage.append(cleavage_dict[bond_idx]) # Target: 0 or 1
            
    if not edge_index: return None 

    # Return standard Python lists, NOT PyTorch tensors to prevent mmap explosions
    return {
        'edge_index': edge_index,
        'edge_attr': edge_attr_bde,
        'y_cleavage': edge_label_cleavage,
        'num_nodes': mol.GetNumAtoms(),
        'smiles': smiles
    }

# ==========================================
# 4. MASTER EXECUTION LOOP
# ==========================================
if __name__ == '__main__':
    # Define paths
    spec_df_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df_COMBINED.pkl"
    mol_df_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df_COMBINED.pkl"
    output_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/phase1_graphs.pt"

    print("[*] Loading DataFrames...")
    spec_df = pd.read_pickle(spec_df_path)
    mol_df = pd.read_pickle(mol_df_path)
    merged_df = pd.merge(spec_df, mol_df, on='mol_id')
    
    # NOTE: Uncomment the line below if you ever need to run a quick test batch again
    # merged_df = merged_df.head(500)
    
    rows = [row for _, row in merged_df.iterrows()]
    
    num_cores = max(1, mp.cpu_count() - 2) 
    print(f"[*] Beginning parallel processing of {len(rows)} molecules using {num_cores} cores...")
    
    with mp.Pool(num_cores) as pool:
        # 1. Collect standard Python dictionaries from workers
        raw_results = list(tqdm(pool.imap(process_molecule_row, rows), total=len(rows)))
        
    print(f"[*] Multiprocessing complete. Assembling PyTorch tensors...")
    
    valid_graphs = []
    
    # 2. Build the PyTorch tensors safely inside the main thread
    for res in tqdm(raw_results, desc="Building PyG Graphs"):
        if res is not None:
            graph = Data(
                edge_index=torch.tensor(res['edge_index'], dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(res['edge_attr'], dtype=torch.float),
                y_cleavage=torch.tensor(res['y_cleavage'], dtype=torch.float),
                num_nodes=res['num_nodes'],
                smiles=res['smiles']
            )
            valid_graphs.append(graph)
    
    print(f"[*] Successfully processed {len(valid_graphs)}/{len(rows)} valid graphs.")
    print(f"[*] Saving massive graph dataset to {output_path}...")
    torch.save(valid_graphs, output_path)
    print("[+] Done!")