import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import os
import sys
from pathlib import Path

# =============================================================================
# 1. SETUP PATHS & IMPORTS
# =============================================================================
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
# Point this to wherever your gf_data_utils.py lives
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

try:
    from gf_data_utils import gf_preprocess, collator
except ImportError as e:
    print(f"[-] Error importing MassFormer utils: {e}")
    sys.exit(1)
    
# =============================================================================
# PHASE 2 DATASET CLASS
# =============================================================================
class Phase2EdgeDataset(Dataset):
    def __init__(self, processed_graphs_path, max_nodes=128):
        super().__init__()
        self.max_nodes = max_nodes
        print(f"[*] Loading Phase 2 graphs (with spectra) from {processed_graphs_path}...")
        self.graphs = torch.load(processed_graphs_path)
        print(f"[+] Loaded {len(self.graphs)} pairs.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        pyg_data = self.graphs[idx]
        mol = Chem.MolFromSmiles(pyg_data.smiles)
        
        if mol.GetNumAtoms() > self.max_nodes:
            return None 

        try:
            mf_item = gf_preprocess(mol, idx, algos_v="algos1")
        except Exception:
            return None

        # Transfer Edge Data
        mf_item.edge_attr_physics = pyg_data.edge_attr 
        mf_item.sparse_edge_index = pyg_data.edge_index 
        
        # Transfer Spectral Data for the Reward Function
        # Assuming your pyg_data object holds the padded peaks and masks
        mf_item.peaks = pyg_data.peaks 
        mf_item.peak_mask = pyg_data.peak_mask
        
        return mf_item

# =============================================================================
# PHASE 2 COLLATE FUNCTION
# =============================================================================
def phase2_collate_fn(items):
    valid_items = [item for item in items if item is not None]
    if len(valid_items) == 0:
        return None

    # 1. MassFormer Dense Backbone Collator
    batched_dict = collator(valid_items) 
    
    # 2. Sparse PyG Batching + Spectral Batching
    batched_edge_attr = []
    batched_edge_index = []
    batched_peaks = []
    batched_masks = []
    
    node_offset = 0
    for item in valid_items:
        num_nodes = item.x.size(0)
        
        batched_edge_attr.append(item.edge_attr_physics)
        shifted_edge_index = item.sparse_edge_index + node_offset
        batched_edge_index.append(shifted_edge_index)
        
        batched_peaks.append(item.peaks)
        batched_masks.append(item.peak_mask)
        
        node_offset += num_nodes
        
    batched_dict['edge_attr_physics'] = torch.cat(batched_edge_attr, dim=0)
    batched_dict['edge_index'] = torch.cat(batched_edge_index, dim=1)
    batched_dict['peaks'] = torch.stack(batched_peaks, dim=0)
    batched_dict['peak_mask'] = torch.stack(batched_masks, dim=0)
    
    return batched_dict