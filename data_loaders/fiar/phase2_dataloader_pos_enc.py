import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem

# ==========================================
# 1. RDKIT TO PYTORCH GEOMETRIC PROCESSOR
# ==========================================
def process_smiles_to_graph(smiles, collision_energy=30.0, peaks=None, peak_mask=None):
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    UPDATED: Now includes Ring and Aromaticity structural encodings to 
    allow the MPNN to organically learn resonance and stability.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        # [EXISTING MASSFORMER FEATURES]
        # Offset atomic number by 1 so 0 can be used for padding
        atomic_num = atom.GetAtomicNum() + 1  
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        hybridization = int(atom.GetHybridization())
        
        # Offset hydrogens to match MassFormer index 4 mapping
        num_hs = atom.GetTotalNumHs() + 2050 
        
        # [NEW] 1-WL BYPASS ENCODINGS (Structural Topology)
        # These give the MPNN the explicit vocabulary to identify resonance
        is_in_ring = 1.0 if atom.IsInRing() else 0.0
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0

        # Append to node feature vector
        features = [
            atomic_num, 
            degree, 
            formal_charge, 
            hybridization, 
            num_hs, 
            is_in_ring,  # Feature Index 5
            is_aromatic  # Feature Index 6
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge Features (Bonds)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond type (Single, Double, Triple, Aromatic)
        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        
        # Undirected graph means we add edges in both directions
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [[bond_type, is_conjugated], [bond_type, is_conjugated]]

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_physics = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle single-atom cases
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_physics = torch.empty((0, 2), dtype=torch.float)

    # Compile into PyG Data object
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr_physics=edge_attr_physics,
        smiles=smiles,
        collision_energy=torch.tensor([collision_energy], dtype=torch.float)
    )
    
    # Attach empirical spectra if available
    if peaks is not None and peak_mask is not None:
        data.peaks = torch.tensor(peaks, dtype=torch.float)
        data.peak_mask = torch.tensor(peak_mask, dtype=torch.bool)
        
    return data

# ==========================================
# 2. THE DATASET CLASS
# ==========================================
class Phase2EdgeDataset(Dataset):
    def __init__(self, processed_graphs_path):
        """
        Loads the pre-processed PyTorch Geometric graphs containing the 
        SMILES, nodes, edges, and corresponding MS/MS spectra.
        """
        super().__init__()
        self.graphs = torch.load(processed_graphs_path)
        
    def __len__(self):
        return len(self.graphs)
        
    def __getitem__(self, idx):
        return self.graphs[idx]

# ==========================================
# 3. THE COLLATOR (BATCHING LOGIC)
# ==========================================
def phase2_collate_fn(batch):
    """
    Custom collate function to handle batches of graphs of varying sizes.
    Pads node features with 0s to fit the maximum molecule size in the batch.
    """
    # Filter out any None values that might have failed RDKit parsing
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Track maximum dimensions for padding
    max_nodes = max([data.x.shape[0] for data in batch])
    max_peaks = max([data.peaks.shape[0] for data in batch]) if hasattr(batch[0], 'peaks') else 0
    feature_dim = batch[0].x.shape[1]
    
    batched_x = []
    batched_edge_index = []
    batched_edge_attr = []
    batched_peaks = []
    batched_peak_masks = []
    batched_ce = []
    
    node_offset = 0

    for data in batch:
        num_nodes = data.x.shape[0]
        
        # Pad Node Features
        padded_x = torch.zeros((max_nodes, feature_dim), dtype=torch.float)
        padded_x[:num_nodes, :] = data.x
        batched_x.append(padded_x)
        
        # Shift Edge Indices to create a disjoint graph representation
        shifted_edges = data.edge_index + node_offset
        batched_edge_index.append(shifted_edges)
        batched_edge_attr.append(data.edge_attr_physics)
        
        node_offset += num_nodes
        
        # Pad Empirical Spectra (Peaks)
        if hasattr(data, 'peaks'):
            num_peaks = data.peaks.shape[0]
            padded_peaks = torch.zeros((max_peaks, 2), dtype=torch.float)
            padded_peaks[:num_peaks, :] = data.peaks
            batched_peaks.append(padded_peaks)
            
            padded_mask = torch.ones((max_peaks,), dtype=torch.bool)
            padded_mask[:num_peaks] = False # False means "valid peak"
            batched_peak_masks.append(padded_mask)
            
        if hasattr(data, 'collision_energy'):
            batched_ce.append(data.collision_energy)

    # Compile the final batch dictionary
    collated_batch = {
        'x': torch.stack(batched_x, dim=0),
        'edge_index': torch.cat(batched_edge_index, dim=1),
        'edge_attr_physics': torch.cat(batched_edge_attr, dim=0)
    }
    
    if len(batched_peaks) > 0:
        collated_batch['peaks'] = torch.stack(batched_peaks, dim=0)
        collated_batch['peak_mask'] = torch.stack(batched_peak_masks, dim=0)
        
    if len(batched_ce) > 0:
        collated_batch['collision_energy'] = torch.stack(batched_ce, dim=0)

    return collated_batch