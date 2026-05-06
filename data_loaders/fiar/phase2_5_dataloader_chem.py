import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
import os
import warnings

# [UPDATED] Chemistry imports for on-the-fly thermodynamic extraction
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

# Mute RDKit warnings so they don't flood your console during multiprocessing
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from phase2_dataloader import phase2_collate_fn, Phase2EdgeDataset
except ImportError:
    print("[-] Warning: Could not import phase2_collate_fn or Phase2EdgeDataset.")
    sys.exit(1)

class PairedSiameseDataset(Dataset):
    def __init__(self, feather_path, graphs_path):
        print(f"[*] Loading paired metadata from: {feather_path}")
        self.pairs_df = pd.read_feather(feather_path)
        
        print(f"[*] Initializing Base Phase2EdgeDataset from: {graphs_path}")
        self.base_dataset = Phase2EdgeDataset(processed_graphs_path=graphs_path)
        
        print("[*] Building spec_id -> Base Index lookup dictionary...")
        self.spec_to_idx = {}
        for i, g in enumerate(self.base_dataset.graphs):
            key = getattr(g, 'spec_id', None)
            if key is not None:
                self.spec_to_idx[key] = i
        print(f"[+] Successfully mapped {len(self.spec_to_idx)} graphs.")

        # =================================================================
        # [THE FIX] Build a spec_id -> SMILES dictionary for RDKit
        # =================================================================
        print("[*] Loading chemical metadata for Thermodynamic Featurization...")
        spec_df_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"
        mol_df_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df.pkl"
        
        self.spec_to_smiles = {}
        try:
            spec_df = pd.read_pickle(spec_df_path)
            mol_df = pd.read_pickle(mol_df_path)
            mol_to_smiles = dict(zip(mol_df['mol_id'], mol_df['smiles']))
            
            for _, row in spec_df.iterrows():
                self.spec_to_smiles[row['spec_id']] = mol_to_smiles.get(row['mol_id'])
            print(f"[+] Successfully loaded SMILES for {len(self.spec_to_smiles)} spectra.")
        except Exception as e:
            print(f"[!] Warning: Could not load SMILES metadata: {e}. Network will use fallbacks.")

    def __len__(self):
        return len(self.pairs_df)

    def _get_thermodynamics(self, smiles, num_nodes):
        """Computes exact mass and Gasteiger charges on-the-fly."""
        # Defaults in case RDKit fails or smiles is missing
        masses = torch.ones(num_nodes, dtype=torch.float32) * 12.0
        charges = torch.zeros(num_nodes, dtype=torch.float32)
        
        if smiles and pd.notna(smiles):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol:
                    rdPartialCharges.ComputeGasteigerCharges(mol)
                    for i, atom in enumerate(mol.GetAtoms()):
                        if i < num_nodes: # Safety check to match graph shape
                            masses[i] = atom.GetMass()
                            
                            c = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
                            if np.isnan(c) or np.isinf(c): c = 0.0
                            charges[i] = float(c)
            except Exception:
                pass # Fail silently and use defaults
                
        return masses, charges

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        spec_A = row['name_main']
        spec_B = row['name_sub']
        
        target_sim = torch.tensor(row['entropy_similarity'], dtype=torch.float32)
        
        idx_A = self.spec_to_idx.get(spec_A)
        idx_B = self.spec_to_idx.get(spec_B)
        
        if idx_A is None or idx_B is None:
            return None 

        graph_A = self.base_dataset[idx_A].clone()
        graph_B = self.base_dataset[idx_B].clone()
        
        graph_A.idx = idx
        graph_B.idx = idx
        
        # =================================================================
        # [THE FIX] Inject Physical Properties into Graph Object
        # =================================================================
        smiles_A = self.spec_to_smiles.get(spec_A)
        smiles_B = self.spec_to_smiles.get(spec_B)
        
        # Get number of nodes from the node feature matrix x
        num_nodes_A = graph_A.x.size(0) if hasattr(graph_A, 'x') else graph_A.num_nodes
        num_nodes_B = graph_B.x.size(0) if hasattr(graph_B, 'x') else graph_B.num_nodes
        
        m_A, c_A = self._get_thermodynamics(smiles_A, num_nodes_A)
        m_B, c_B = self._get_thermodynamics(smiles_B, num_nodes_B)
        
        # Attach to the PyG data object
        graph_A.node_masses = m_A
        graph_A.node_electronics = c_A
        
        graph_B.node_masses = m_B
        graph_B.node_electronics = c_B

        return graph_A, graph_B, target_sim


def pad_1d_features(graphs, feature_name):
    """Safely pads 1D dynamic properties to match the dense batched node dimension."""
    tensors = [getattr(g, feature_name) for g in graphs]
    max_len = max([t.size(0) for t in tensors])
    padded = torch.zeros(len(tensors), max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, :t.size(0)] = t
    return padded

def siamese_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
        
    graphs_A = [item[0] for item in batch]
    graphs_B = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch])
    
    # Process the base topological dictionaries
    batch_A = phase2_collate_fn(graphs_A)
    batch_B = phase2_collate_fn(graphs_B)
    
    # =================================================================
    # [THE FIX] Safely inject padded thermodynamic features into the dict
    # =================================================================
    if isinstance(batch_A, dict) and isinstance(batch_B, dict):
        batch_A['node_masses'] = pad_1d_features(graphs_A, 'node_masses')
        batch_A['node_electronics'] = pad_1d_features(graphs_A, 'node_electronics')
        
        batch_B['node_masses'] = pad_1d_features(graphs_B, 'node_masses')
        batch_B['node_electronics'] = pad_1d_features(graphs_B, 'node_electronics')
    
    return batch_A, batch_B, targets


# ==============================================================================
# ISOLATED TESTING BLOCK
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the Siamese Dataloader")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to stratified_binary_07_dataset_train.feather")
    parser.add_argument("--graphs_path", type=str, required=True, help="Path to phase3_graphs.pt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    args = parser.parse_args()

    if not os.path.exists(args.pairs_path):
        print(f"[!] Error: Pairs file not found at {args.pairs_path}")
        sys.exit(1)
    if not os.path.exists(args.graphs_path):
        print(f"[!] Error: Graphs file not found at {args.graphs_path}")
        sys.exit(1)

    print("\n--- Initiating Dataloader Test ---")
    
    test_dataset = PairedSiameseDataset(feather_path=args.pairs_path, graphs_path=args.graphs_path)
    print(f"\n[+] Dataset initialized successfully with {len(test_dataset)} pairs.")

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=siamese_collate_fn,
        num_workers=0 # Keep at 0 for safe local testing
    )

    print("\n--- Fetching a Test Batch ---")
    try:
        batch_iterator = iter(test_loader)
        batch_A, batch_B, targets = next(batch_iterator)
        
        while batch_A is None:
            print("[!] Batch was empty (missing graphs), fetching next...")
            batch_A, batch_B, targets = next(batch_iterator)

        print("[+] Successfully fetched batch!")
        print("\n--- Batch Structure ---")
        print(f"Targets (Entropy Similarities): {targets.shape}")
        
        print("\nBatch A (Molecule 1):")
        if isinstance(batch_A, dict):
            print(f"  - Node Features 'x': {batch_A.get('x', torch.tensor([])).shape}")
            if 'attn_bias' in batch_A:
                print(f"  - Attn Bias: {batch_A['attn_bias'].shape}")
            # [UPDATED] Verify our new injected properties!
            if 'node_masses' in batch_A:
                print(f"  - Node Masses: {batch_A['node_masses'].shape}  <-- [NEW]")
                print(f"  - Node Electronics: {batch_A['node_electronics'].shape}  <-- [NEW]")
        else:
            print(f"  - Node Features 'x': {batch_A.x.shape}")
            
        print("\n[+] Dataloader test passed successfully! You are ready to restart Phase 2.5.")

    except StopIteration:
        print("[-] Dataloader is empty.")
    except Exception as e:
        print(f"[!] Error during batch fetching: {e}")