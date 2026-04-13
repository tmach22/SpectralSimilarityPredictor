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
# 2. PHASE 1 DATASET CLASS
# =============================================================================
class Phase1EdgeDataset(Dataset):
    def __init__(self, processed_graphs_path, max_nodes=128):
        super().__init__()
        self.max_nodes = max_nodes
        print(f"[*] Loading pre-computed Phase 1 graphs from {processed_graphs_path} into RAM...")
        self.graphs = torch.load(processed_graphs_path)
        print(f"[+] Loaded {len(self.graphs)} graphs.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # 1. Get our pre-computed graph with BDEs and cleavage targets
        pyg_data = self.graphs[idx]
        
        # 2. Reconstruct the RDKit molecule from the SMILES
        mol = Chem.MolFromSmiles(pyg_data.smiles)
        
        # We rely strictly on Implicit Hydrogens (Do NOT call Chem.AddHs!)
        if mol.GetNumAtoms() > self.max_nodes:
            # Skip massive molecules that exceed Graphormer padding limits
            return None 

        # 3. Run the heavy MassFormer/Graphormer featurization
        # This builds data.x, data.attn_bias, data.spatial_pos, etc.
        try:
            mf_item = gf_preprocess(mol, idx, algos_v="algos1")
        except Exception as e:
            # Catch rare RDKit pathfinding errors during spatial_pos generation
            return None

        # 4. MERGE: Transfer our Phase 1 specific targets
        # Rename to 'edge_attr_physics' so it doesn't collide with MassFormer's native edge tensors
        mf_item.y_cleavage = pyg_data.y_cleavage
        mf_item.edge_attr_physics = pyg_data.edge_attr 
        
        # Retain original edge_index for the regressor head
        mf_item.sparse_edge_index = pyg_data.edge_index 
        
        return mf_item

# =============================================================================
# 3. CUSTOM COLLATE FUNCTION (The Bridge)
# =============================================================================
def phase1_collate_fn(items):
    """
    Handles both the dense padding for the Graphormer backbone 
    and the sparse concatenation for the EdgeRegressorHead.
    """
    # Filter out None values from preprocessing failures
    valid_items = [item for item in items if item is not None]
    
    if len(valid_items) == 0:
        return None

    # 1. Use the original MassFormer collator for the backbone features
    # This returns a dictionary with padded x, attn_bias, spatial_pos, etc.
    batched_dict = collator(valid_items) 
    
    # 2. PyG-style Sparse Edge Batching for the Regressor Head
    batched_y_cleavage = []
    batched_edge_attr = []
    batched_edge_index = []
    
    node_offset = 0
    for item in valid_items:
        num_nodes = item.x.size(0)
        
        batched_y_cleavage.append(item.y_cleavage)
        batched_edge_attr.append(item.edge_attr_physics)
        
        # Shift edge index by the number of nodes already in the batch
        # This allows the Regressor Head to treat the entire batch as one giant disconnected graph
        shifted_edge_index = item.sparse_edge_index + node_offset
        batched_edge_index.append(shifted_edge_index)
        
        node_offset += num_nodes
        
    # Append the sparse tensors to the batch dictionary
    batched_dict['y_cleavage'] = torch.cat(batched_y_cleavage, dim=0)
    batched_dict['edge_attr_physics'] = torch.cat(batched_edge_attr, dim=0)
    batched_dict['edge_index'] = torch.cat(batched_edge_index, dim=1)
    
    return batched_dict

# =============================================================================
# 4. TESTING BLOCK
# =============================================================================
if __name__ == "__main__":
    # Point this to your generated test or full dataset
    data_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/phase1_graphs.pt"
    
    dataset = Phase1EdgeDataset(data_path)
    
    # Notice we use the standard PyTorch DataLoader, NOT the PyG DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=8, 
        collate_fn=phase1_collate_fn, 
        shuffle=True, 
        num_workers=4
    )
    
    print("\n--- Testing Phase 1 Dataloader ---")
    batch = next(iter(loader))
    
    if batch is not None:
        print("[*] MassFormer Backbone (Dense) Tensors:")
        print(f"    x shape:               {batch['x'].shape}")
        print(f"    attn_bias shape:       {batch['attn_bias'].shape}")
        print(f"    spatial_pos shape:     {batch['spatial_pos'].shape}")
        
        print("\n[*] EdgeRegressor Head (Sparse) Tensors:")
        print(f"    edge_index shape:      {batch['edge_index'].shape}")
        print(f"    edge_attr_physics:     {batch['edge_attr_physics'].shape}")
        print(f"    y_cleavage shape:      {batch['y_cleavage'].shape}")
        
        print("\n[+] SUCCESS! The tensors are ready for Phase 1 Training.")
    else:
        print("[-] Batch generation failed.")