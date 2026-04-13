import torch
import torch.nn as nn
import torch.distributions as dist
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from torch_scatter import scatter_sum, scatter_mean
import math

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


# ==============================================================================
# [NEW] HYDROGEN ROUTING HEAD
# ==============================================================================
class HydrogenRoutingHead(nn.Module):
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
            nn.Linear(hidden_dim // 2, 5) # 5 Classes: [-2H, -1H, 0H, +1H, +2H]
        )

    def forward(self, node_embeddings_flat, edge_index, edge_attr):
        u_emb = node_embeddings_flat[edge_index[0]]
        v_emb = node_embeddings_flat[edge_index[1]]
        edge_features = torch.cat([u_emb, v_emb, edge_attr], dim=-1)
        return self.mlp(edge_features)


class DESAFNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        emb_dim = model_config.get('embed_dim', 768) if model_config.get('embed_dim') != -1 else 768
        
        # Dual Heads
        self.edge_head = EdgeRegressorHead(node_emb_dim=emb_dim, edge_attr_dim=2, hidden_dim=256)
        self.h_routing_head = HydrogenRoutingHead(node_emb_dim=emb_dim, edge_attr_dim=2, hidden_dim=256)

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
            
        # Ensure RL heads are trainable
        for param in self.edge_head.parameters():
            param.requires_grad = True
        for param in self.h_routing_head.parameters():
            param.requires_grad = True
            
        print("[*] MassFormer Backbone Frozen. Gradients routing strictly to Edge & Routing Heads.")

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
            valid_raw_features.append(batch['x'][b, :true_nodes, :]) 
            node_batch_tracker.append(torch.full((true_nodes,), b, device=device, dtype=torch.float))
            
        X_flat = torch.cat(valid_X_list, dim=0)
        raw_features_flat = torch.cat(valid_raw_features, dim=0) # Shape: [Total_Nodes, 9]
        node_batch_tracker = torch.cat(node_batch_tracker, dim=0)
        
        # 3. Dual-Head Predictions
        break_logits = self.edge_head(X_flat, batch['edge_index'], batch['edge_attr_physics'])
        break_probs = torch.sigmoid(break_logits)
        
        h_logits = self.h_routing_head(X_flat, batch['edge_index'], batch['edge_attr_physics'])
        h_probs = torch.softmax(h_logits, dim=-1)
        
        # 4. Phase 2 Discrete Samplers
        m_cut = dist.Bernoulli(probs=break_probs)
        m_h = dist.Categorical(probs=h_probs)
        
        if self.training:
            cut_actions = m_cut.sample() 
            h_actions = m_h.sample()
        else:
            # Deterministic evaluation mapping for the visualizer script
            cut_actions = (break_probs > 0.5).float()
            h_actions = torch.argmax(h_probs, dim=-1)
            
        # Route gradients ONLY if a cut was actually made
        cut_log_probs = m_cut.log_prob(cut_actions) 
        h_log_probs = m_h.log_prob(h_actions)
        log_probs = cut_log_probs + (h_log_probs * cut_actions)
        
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
        
        # ========================================================
        # 8. PHYSICAL HYDROGEN ROUTING (The Rearrangement Simulator)
        # ========================================================
        H_MASS = 1.007825
        # Map classes [0, 1, 2, 3, 4] to physical mass shifts [-2H, -1H, 0H, +1H, +2H]
        routing_shifts = (h_actions.float() - 2.0) * H_MASS 
        
        # Only shift mass if the bond was actually severed
        active_shifts = routing_shifts * cut_actions 
        
        # PyG edges are bidirectional. Shift mass in one direction (u -> v) to prevent doubling
        u, v = batch['edge_index'][0], batch['edge_index'][1]
        directed_mask = (u < v).float()
        final_shifts = active_shifts * directed_mask
        
        # Physically move the mass from node v to node u
        node_masses = node_masses.clone() 
        node_masses.scatter_add_(0, u, final_shifts)
        node_masses.scatter_add_(0, v, -final_shifts)
        
        # 9. Final Assembly
        fragment_masses = scatter_sum(node_masses, fragment_labels, dim=0)
        theoretical_mzs = fragment_masses + 1.007276 
        
        if not self.training:
            return theoretical_mzs, log_probs, fragment_batch_idx, cut_actions
            
        return theoretical_mzs, log_probs, fragment_batch_idx


# ==============================================================================
# THE RL REWARD ENVIRONMENT
# ==============================================================================
def calculate_cosine_reward(theoretical_mzs, fragment_batch_idx, batched_peaks, batched_masks, tolerance=0.05):
    """Calculates Spectral Cosine Similarity as the REINFORCE Reward."""
    B = batched_peaks.shape[0]
    rewards = torch.zeros(B, device=theoretical_mzs.device)
    
    for b in range(B):
        frag_mzs = theoretical_mzs[fragment_batch_idx == b]
        num_frags = len(frag_mzs)
        
        peaks = batched_peaks[b]
        mask = batched_masks[b] 
        
        valid_peaks = peaks[~mask]
        if num_frags == 0 or len(valid_peaks) == 0:
            continue
            
        exp_mzs = valid_peaks[:, 0] * 1000.0 
        exp_ints = valid_peaks[:, 1].clone()

        # Precursor Exploit Guardrail
        if num_frags <= 1:
            rewards[b] = 0.0
            continue

        max_mz_idx = torch.argmax(exp_mzs)
        exp_ints[max_mz_idx] = 0.0 
            
        diffs = torch.abs(frag_mzs.unsqueeze(1) - exp_mzs.unsqueeze(0))
        hits = (diffs <= tolerance)
        
        claimed_intensities = exp_ints * hits.max(dim=0).values 
        dot_product = torch.sum(claimed_intensities * exp_ints)
        norm_claimed = torch.sqrt(torch.sum(claimed_intensities ** 2)) + 1e-8
        norm_exp = torch.sqrt(torch.sum(exp_ints ** 2)) + 1e-8
        
        base_reward = dot_product / (norm_claimed * norm_exp)
        
        # ========================================================
        # [NEW] THE PARSIMONY PENALTY
        # Crushes the reward if the network abuses the Alphabet Soup exploit.
        # ========================================================
        optimal_frags = 5.0
        if num_frags > optimal_frags:
            # Steep exponential decay (10 frags yields ~90% reward reduction)
            penalty_factor = math.exp(-0.45 * (num_frags - optimal_frags))
            rewards[b] = base_reward * penalty_factor
        else:
            rewards[b] = base_reward
            
    return rewards