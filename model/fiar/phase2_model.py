import torch
import torch.nn as nn
import torch.distributions as dist
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from torch_scatter import scatter_sum, scatter_mean

# Import your actual MassFormer backbone
from classifier_siamese_model import UnpooledMassFormerEncoder

class EdgeRegressorHead(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim=2, hidden_dim=128):
        super().__init__()
        input_dim = (node_emb_dim * 2) + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_embeddings_flat, edge_index, edge_attr):
        u_emb = node_embeddings_flat[edge_index[0]]
        v_emb = node_embeddings_flat[edge_index[1]]
        edge_features = torch.cat([u_emb, v_emb, edge_attr], dim=-1)
        return self.mlp(edge_features).squeeze(-1)

class DESAFNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        emb_dim = model_config.get('embed_dim', 768) if model_config.get('embed_dim') != -1 else 768
        self.edge_head = EdgeRegressorHead(node_emb_dim=emb_dim, edge_attr_dim=2, hidden_dim=256)

        # GPU Mass Lookup Dictionary
        self.register_buffer('mass_lookup', torch.zeros(120))
        self.mass_lookup[1] = 1.007825   # H
        self.mass_lookup[6] = 12.000000  # C
        self.mass_lookup[7] = 14.003074  # N
        self.mass_lookup[8] = 15.994915  # O
        self.mass_lookup[15] = 30.973762 # P
        self.mass_lookup[16] = 31.972071 # S
        self.mass_lookup[17] = 34.968853 # Cl
        self.mass_lookup[9] = 18.998403  # F
        self.mass_lookup[35] = 78.918336 # Br
        self.mass_lookup[53] = 126.90447 # I

    def freeze_backbone(self):
        """Freezes the MassFormer to stabilize early REINFORCE training."""
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        print("[*] MassFormer Backbone Frozen. Gradients routing strictly to Edge Head.")

    def forward_phase2(self, batch):
        device = batch['x'].device
        B = batch['x'].shape[0]
        
        # 1. Payload Extraction
        X_base = self.graph_encoder({'gf_v2_data': batch})
        X = X_base[0][:, 1:, :] if isinstance(X_base, tuple) else X_base[:, 1:, :]
        
        # 2. Bridge & Batch Tracking
        valid_X_list, node_batch_tracker, valid_raw_features = [], [], []
        for b in range(B):
            true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
            
            valid_X_list.append(X[b, :true_nodes, :])
            valid_raw_features.append(batch['x'][b, :true_nodes, :]) # Extract input features safely
            node_batch_tracker.append(torch.full((true_nodes,), b, device=device, dtype=torch.float))
            
        X_flat = torch.cat(valid_X_list, dim=0)
        raw_features_flat = torch.cat(valid_raw_features, dim=0) # Shape: [Total_Nodes, 9]
        node_batch_tracker = torch.cat(node_batch_tracker, dim=0)
        
        # 3. Phase 1 Anchor Scoring
        break_logits = self.edge_head(X_flat, batch['edge_index'], batch['edge_attr_physics'])
        
        # 4. Phase 2 Discrete Sampler
        m = dist.Bernoulli(logits=break_logits)
        cut_actions = m.sample() 
        log_probs = m.log_prob(cut_actions) 
        
        # 5. Topological Severing
        surviving_mask = (cut_actions == 0)
        surviving_edge_index = batch['edge_index'][:, surviving_mask]
        
        # 6. Deterministic Grouping
        adj_sparse = to_scipy_sparse_matrix(surviving_edge_index, num_nodes=X_flat.size(0))
        _, fragment_labels = connected_components(adj_sparse, directed=False)
        fragment_labels = torch.tensor(fragment_labels, device=device, dtype=torch.long)
        
        fragment_batch_idx = scatter_mean(node_batch_tracker, fragment_labels, dim=0).long()
        
        # 7. Mass Pooler
        raw_atomic_vals = raw_features_flat[:, 0]
        atomic_numbers = torch.clamp(raw_atomic_vals - 1, min=0, max=119).long()
        
        raw_hydrogens = raw_features_flat[:, 4]
        implicit_hydrogens = torch.clamp(raw_hydrogens - 2050, min=0).float()
        
        base_masses = self.mass_lookup[atomic_numbers]
        node_masses = base_masses + (implicit_hydrogens * 1.007825)
        
        fragment_masses = scatter_sum(node_masses, fragment_labels, dim=0)
        theoretical_mzs = fragment_masses + 1.007276 
        
        return theoretical_mzs, log_probs, fragment_batch_idx

def calculate_cosine_reward(theoretical_mzs, fragment_batch_idx, batched_peaks, batched_masks, tolerance=0.05):
    """Calculates Spectral Cosine Similarity as the REINFORCE Reward."""
    B = batched_peaks.shape[0]
    rewards = torch.zeros(B, device=theoretical_mzs.device)
    
    for b in range(B):
        frag_mzs = theoretical_mzs[fragment_batch_idx == b]
        peaks = batched_peaks[b]
        mask = batched_masks[b] 
        
        valid_peaks = peaks[~mask]
        if len(frag_mzs) == 0 or len(valid_peaks) == 0:
            continue
            
        exp_mzs = valid_peaks[:, 0] * 1000.0 
        exp_ints = valid_peaks[:, 1].clone()

        max_mz_idx = torch.argmax(exp_mzs)
        exp_ints[max_mz_idx] = 0.0 

        if len(frag_mzs) == 1:
            rewards[b] = 0.0
            continue
            
        diffs = torch.abs(frag_mzs.unsqueeze(1) - exp_mzs.unsqueeze(0))
        hits = (diffs <= tolerance)
        
        claimed_intensities = exp_ints * hits.max(dim=0).values 
        dot_product = torch.sum(claimed_intensities * exp_ints)
        norm_claimed = torch.sqrt(torch.sum(claimed_intensities ** 2)) + 1e-8
        norm_exp = torch.sqrt(torch.sum(exp_ints ** 2)) + 1e-8
        
        rewards[b] = dot_product / (norm_claimed * norm_exp)
        
    return rewards