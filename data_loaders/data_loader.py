import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
# 1. Input file paths
SPECTRA_FILE = '/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/msg_df.feather'

# 2. DataLoader parameters
BATCH_SIZE = 32
NUM_WORKERS = 4 # Adjust based on your machine's CPU cores

# --- Step 1: Detailed Molecular Featurization (MassFormer/Graphormer Style) ---

# Define the feature sets based on the provided script and common practices
ATOM_PERMITTED_LIST = ['Ac','Ag','Al','Am','Ar','As','At','Au','B','Ba','Be','Bh','Bi','Bk','Br','C','Ca','Cd','Ce','Cf','Cl','Cm','Cn','Co','Cr','Cs','Cu','Db','Ds','Dy','Er','Es','Eu','F','Fe','Fl','Fm','Fr','Ga','Gd','Ge','H','He','Hf','Hg','Ho','Hs','I','In','Ir','K','Kr','La','Li','Lr','Lu','Lv','Mc','Md','Mg','Mn','Mo','Mt','N','Na','Nb','Nd','Ne','Nh','Ni','No','Np','O','Og','Os','P','Pa','Pb','Pd','Pm','Po','Pr','Pt','Pu','Ra','Rb','Re','Rf','Rg','Rh','Rn','Ru','S','Sb','Sc','Se','Sg','Si','Sm','Sn','Sr','Ta','Tb','Tc','Te','Th','Ti','Tl','Tm','Ts','U','V','W','Xe','Y','Yb','Zn','Zr']
HYBRIDIZATION_PERMITTED_LIST = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED', 'OTHER']
BOND_PERMITTED_LIST = ['AROMATIC', 'DATIVE', 'DATIVEL', 'DATIVEONE', 'DATIVER', 'DOUBLE', 'FIVEANDAHALF', 'FOURANDAHALF', 'HEXTUPLE', 'HYDROGEN', 'IONIC', 'ONEANDAHALF', 'OTHER', 'QUADRUPLE', 'QUINTUPLE', 'SINGLE', 'THREEANDAHALF', 'THREECENTER', 'TRIPLE', 'TWOANDAHALF']

def one_hot_encode(value, permitted_list):
    """Converts a value to a one-hot encoded vector."""
    # If the value is not in the list, it's treated as 'Unknown' or 'OTHER'
    if value not in permitted_list:
        value = permitted_list[-1]
    return [int(value == s) for s in permitted_list]

def get_atom_features(atom: Chem.Atom):
    """
    Calculates detailed features for a single atom.
    This is a one-hot encoding scheme based on the provided script.
    """
    features = []
    # --- Feature Set ---
    # 1. Atomic Symbol
    features += one_hot_encode(atom.GetSymbol(), ATOM_PERMITTED_LIST)
    # 2. Degree (number of bonds)
    features += one_hot_encode(atom.GetDegree(), list(range(11))) # 0-10
    # 3. Number of bonded hydrogens
    features += one_hot_encode(atom.GetTotalNumHs(), list(range(9))) # 0-8
    # 4. Hybridization
    features += one_hot_encode(str(atom.GetHybridization()), HYBRIDIZATION_PERMITTED_LIST)
    # 5. Aromaticity (boolean)
    features.append(atom.GetIsAromatic())
    # 6. In Ring (boolean)
    features.append(atom.IsInRing())
    
    return features

def get_bond_features(bond: Chem.Bond):
    """
    Calculates detailed features for a single bond.
    """
    features = []
    # --- Feature Set ---
    # 1. Bond Type
    features += one_hot_encode(str(bond.GetBondType()), BOND_PERMITTED_LIST)
    # 2. Conjugation (boolean)
    features.append(bond.GetIsConjugated())
    
    return features

def featurize_smiles_massformer(smiles: str):
    """
    Converts a SMILES string into a dictionary of graph tensors,
    using the detailed MassFormer-style featurization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Get atom (node) features
    atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    node_features = torch.tensor(atom_features_list, dtype=torch.float32)

    # Get bond (edge) features and connectivity
    edge_indices = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feats = get_bond_features(bond)
        
        # Add edges in both directions for an undirected graph
        edge_indices.extend([[i, j], [j, i]])
        edge_features_list.extend([bond_feats, bond_feats])

    if not edge_indices: # Handle molecules with a single atom
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, len(BOND_PERMITTED_LIST) + 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features_list, dtype=torch.float32)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }


# --- Step 2: Custom PyTorch Dataset Class (Updated) ---

class SpectralPairDataset(Dataset):
    def __init__(self, pairs_file: str, spectra_file: str):
        print(f"Loading data for {os.path.basename(pairs_file)}...")
        self.pairs_df = pd.read_feather(pairs_file)
        spectra_df = pd.read_feather(spectra_file)
        self.id_to_smiles = spectra_df.set_index('spectrum_id')['smiles'].to_dict()
        print("Dataset initialized.")

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        pair_info = self.pairs_df.iloc[idx]
        id_a, id_b = pair_info['name_main'], pair_info['name_sub']
        similarity_score = pair_info['cosine_similarity']

        smiles_a = self.id_to_smiles.get(id_a)
        smiles_b = self.id_to_smiles.get(id_b)

        if not smiles_a or not smiles_b:
            return None

        # UPDATED: Use the new, detailed featurization function
        graph_a = featurize_smiles_massformer(smiles_a)
        graph_b = featurize_smiles_massformer(smiles_b)

        if graph_a is None or graph_b is None:
            return None

        return graph_a, graph_b, torch.tensor(similarity_score, dtype=torch.float32)

def collate_fn(batch):
    """Custom collate function to handle failed featurizations."""
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return empty tensors if the whole batch failed
        return None, None, None
    
    graphs_a, graphs_b, scores = zip(*batch)
    return list(graphs_a), list(graphs_b), torch.stack(scores)


# --- Step 3: Instantiate Datasets and DataLoaders ---
if __name__ == "__main__":
    print("\n--- Creating Datasets and DataLoaders with MassFormer Featurization ---")

    parser = argparse.ArgumentParser(description="Test the GNN data loader with rich features.")
    parser.add_argument("--datasplit_path", type=str, required=True, help="Path to the feather file.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of worker threads for DataLoader.")
    args = parser.parse_args()
    
    dataset = SpectralPairDataset(args.datasplit_path, SPECTRA_FILE)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    print(f"\nDataLoaders created with batch size {args.batch_size}.")
    print(f"Number of training batches: {len(loader)}")

    # --- Example: Iterate over one batch from the training loader ---
    print("\n--- Example Batch ---")
    graphs_a, graphs_b, scores = next(iter(loader))
    
    if graphs_a:
        print(f"Batch of graphs for molecule A: {len(graphs_a)} items")
        print(f"Batch of graphs for molecule B: {len(graphs_b)} items")
        print(f"Batch of similarity scores: {scores.shape}")
        
        first_graph = graphs_a[0]
        print("\n--- Inspecting the First Graph Object in the Batch ---")
        print(f"Node Features Shape: {first_graph['node_features'].shape}")
        print(f"  -> Interpretation: [{first_graph['node_features'].shape} atoms, {first_graph['node_features'].shape[1]} features/atom]")
        print(f"Edge Index Shape: {first_graph['edge_index'].shape}")
        print(f"  -> Interpretation: [2 (source/target), {first_graph['edge_index'].shape[1]} directed edges]")
        print(f"Edge Features Shape: {first_graph['edge_features'].shape}")
        print(f"  -> Interpretation: [{first_graph['edge_features'].shape} edges, {first_graph['edge_features'].shape[1]} features/bond]")
    else:
        print("Could not retrieve a valid batch.")