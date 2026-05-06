import torch
import torch.nn as nn
import torch.distributions as dist
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from torch.nn.utils.rnn import pad_sequence
import math

# Sum Pooler for IVR/Degrees of Freedom Modeling
from torch_geometric.nn import global_add_pool

# Import your actual MassFormer backbone
from classifier_siamese_model import UnpooledMassFormerEncoder

# ==============================================================================
# PHASE 1 & 2 HEADS (THE "ACTORS")
# ==============================================================================
class EdgeRegressorHead(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim=2, hidden_dim=128):
        super().__init__()
        # Input: Local Nodes (x2) + Edge Attr + Global Sum-Pooled Vector
        input_dim = (node_emb_dim * 2) + edge_attr_dim + node_emb_dim
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

    def forward(self, node_embeddings_flat, edge_index, edge_attr, global_graph_embedding):
        u_emb = node_embeddings_flat[edge_index[0]]
        v_emb = node_embeddings_flat[edge_index[1]]
        # Inject the macroscopic global embedding into the local bond decision
        edge_features = torch.cat([u_emb, v_emb, edge_attr, global_graph_embedding], dim=-1)
        return self.mlp(edge_features).squeeze(-1)


class HydrogenRoutingHead(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim=2, hidden_dim=128):
        super().__init__()
        # Input: Local Nodes (x2) + Edge Attr + Global Sum-Pooled Vector
        input_dim = (node_emb_dim * 2) + edge_attr_dim + node_emb_dim
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

    def forward(self, node_embeddings_flat, edge_index, edge_attr, global_graph_embedding):
        u_emb = node_embeddings_flat[edge_index[0]]
        v_emb = node_embeddings_flat[edge_index[1]]
        # Inject the macroscopic global embedding into the local routing decision
        edge_features = torch.cat([u_emb, v_emb, edge_attr, global_graph_embedding], dim=-1)
        return self.mlp(edge_features)


# ==============================================================================
# PHASE 3 SET TRANSFORMER (THE "CRITIC" SCORER)
# ==============================================================================
class SpectralSetTransformer(nn.Module):
    def __init__(self, emb_dim=768, heads=8, layers=3):
        super().__init__()
        
        # 1. Collision Energy Projector
        self.ce_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim)
        )
        
        # 2. The Set Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=heads, 
            dim_feedforward=emb_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # 3. Intensity Scoring Head
        self.intensity_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, frag_embeddings, batch_idx, collision_energy):
        device = frag_embeddings.device
        B = batch_idx.max().item() + 1
        
        # Unflatten & Pad (Prepare the Bag)
        frag_list = []
        for b in range(B):
            frag_list.append(frag_embeddings[batch_idx == b])
        
        padded_seq = pad_sequence(frag_list, batch_first=True, padding_value=0.0)
        
        lengths = torch.tensor([len(f) for f in frag_list], device=device)
        max_len = padded_seq.size(1)
        pad_mask = torch.arange(max_len, device=device).expand(B, max_len) >= lengths.unsqueeze(1)
        
        # Collision Energy Injection
        ce_emb = self.ce_proj((collision_energy / 100).unsqueeze(1)).unsqueeze(1)
        conditioned_seq = padded_seq + ce_emb
        
        # Self-Attention (Thermodynamic Competition)
        out_seq = self.transformer(conditioned_seq, src_key_padding_mask=pad_mask)
        
        # Intensity Prediction & Re-Flattening
        out_seq_flat = out_seq.view(-1, out_seq.size(-1))
        raw_intensities_flat = self.intensity_head(out_seq_flat)
        raw_intensities = raw_intensities_flat.view(B, max_len)
        
        raw_intensities[pad_mask] = -1e9 
        temperature = 0.1
        rel_intensities_padded = torch.softmax(raw_intensities/temperature, dim=1)
        
        flat_intensities = torch.cat([rel_intensities_padded[b, :lengths[b]] for b in range(B)], dim=0)
        
        return flat_intensities


# ==============================================================================
# MAIN DESAF-NET ARCHITECTURE
# ==============================================================================
class DESAFNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        emb_dim = model_config.get('embed_dim', 768) if model_config.get('embed_dim') != -1 else 768
        
        # Phase 2 Heads
        self.edge_head = EdgeRegressorHead(node_emb_dim=emb_dim, edge_attr_dim=2, hidden_dim=256)
        self.h_routing_head = HydrogenRoutingHead(node_emb_dim=emb_dim, edge_attr_dim=2, hidden_dim=256)

        # Phase 3 Head
        self.set_transformer = SpectralSetTransformer(emb_dim=emb_dim)

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
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
            
        for param in self.edge_head.parameters():
            param.requires_grad = True
        for param in self.h_routing_head.parameters():
            param.requires_grad = True
            
        print("[*] MassFormer Backbone Frozen. Gradients routing strictly to Edge & Routing Heads.")

    def forward_phase2(self, batch):
        """
        Standard Phase 2 forward pass (Maintained for backwards compatibility)
        """
        device = batch['x'].device
        B = batch['x'].shape[0]
        
        X_base = self.graph_encoder({'gf_v2_data': batch})
        X = X_base[0][:, 1:, :] if isinstance(X_base, tuple) else X_base[:, 1:, :]
        
        valid_X_list, node_batch_tracker, valid_raw_features = [], [], []
        for b in range(B):
            true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
            
            valid_X_list.append(X[b, :true_nodes, :])
            valid_raw_features.append(batch['x'][b, :true_nodes, :]) 
            node_batch_tracker.append(torch.full((true_nodes,), b, device=device, dtype=torch.long))
            
        X_flat = torch.cat(valid_X_list, dim=0)
        raw_features_flat = torch.cat(valid_raw_features, dim=0) 
        node_batch_tracker = torch.cat(node_batch_tracker, dim=0)
        
        # Calculate Global Intact Precursor Embedding via Sum Pooling
        global_graph_emb = global_add_pool(X_flat, node_batch_tracker)
        edge_batch_idx = node_batch_tracker[batch['edge_index'][0]]
        global_emb_expanded = global_graph_emb[edge_batch_idx]
        
        break_logits = self.edge_head(X_flat, batch['edge_index'], batch['edge_attr_physics'], global_emb_expanded)
        break_probs = torch.sigmoid(break_logits)
        
        h_logits = self.h_routing_head(X_flat, batch['edge_index'], batch['edge_attr_physics'], global_emb_expanded)
        h_probs = torch.softmax(h_logits, dim=-1)
        
        m_cut = dist.Bernoulli(probs=break_probs)
        m_h = dist.Categorical(probs=h_probs)
        
        if self.training:
            cut_actions = m_cut.sample() 
            h_actions = m_h.sample()
        else:
            cut_actions = (break_probs > 0.5).float()
            h_actions = torch.argmax(h_probs, dim=-1)
            
        cut_log_probs = m_cut.log_prob(cut_actions) 
        h_log_probs = m_h.log_prob(h_actions)
        log_probs = cut_log_probs + (h_log_probs * cut_actions)
        
        surviving_mask = (cut_actions == 0)
        surviving_edge_index = batch['edge_index'][:, surviving_mask]
        
        adj_sparse = to_scipy_sparse_matrix(surviving_edge_index, num_nodes=X_flat.size(0))
        _, fragment_labels = connected_components(adj_sparse, directed=False)
        fragment_labels = torch.tensor(fragment_labels, device=device, dtype=torch.long)
        
        fragment_batch_idx = scatter_mean(node_batch_tracker.float(), fragment_labels, dim=0).long()
        
        raw_atomic_vals = raw_features_flat[:, 0]
        atomic_numbers = torch.clamp(raw_atomic_vals - 1, min=0, max=119).long()
        raw_hydrogens = raw_features_flat[:, 4]
        implicit_hydrogens = torch.clamp(raw_hydrogens - 2050, min=0).float()
        
        base_masses = self.mass_lookup[atomic_numbers]
        node_masses = base_masses + (implicit_hydrogens * 1.007825)
        
        H_MASS = 1.007825
        routing_shifts = (h_actions.float() - 2.0) * H_MASS 
        active_shifts = routing_shifts * cut_actions 
        
        u, v = batch['edge_index'][0], batch['edge_index'][1]
        directed_mask = (u < v).float()
        final_shifts = active_shifts * directed_mask
        
        node_masses = node_masses.clone() 
        node_masses.scatter_add_(0, u, final_shifts)
        node_masses.scatter_add_(0, v, -final_shifts)
        
        fragment_masses = scatter_sum(node_masses, fragment_labels, dim=0)
        theoretical_mzs = fragment_masses + 1.007276 
        
        if not self.training:
            return theoretical_mzs, log_probs, fragment_batch_idx, cut_actions
            
        return theoretical_mzs, log_probs, fragment_batch_idx

    def forward_phase3_cascade(self, batch, max_steps=3):
        """
        Loops the Phase 2 physics engine to generate a multi-step 
        Directed Acyclic Graph (DAG) of fragments. 
        Tracks action probabilities for REINFORCE Actor-Critic updating.
        """
        device = batch['x'].device
        B = batch['x'].shape[0]

        X_base = self.graph_encoder({'gf_v2_data': batch})
        X = X_base[0][:, 1:, :] if isinstance(X_base, tuple) else X_base[:, 1:, :]

        valid_X_list, node_batch_tracker, valid_raw_features = [], [], []
        for b in range(B):
            true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
            
            valid_X_list.append(X[b, :true_nodes, :])
            valid_raw_features.append(batch['x'][b, :true_nodes, :]) 
            node_batch_tracker.append(torch.full((true_nodes,), b, device=device, dtype=torch.long))

        X_flat = torch.cat(valid_X_list, dim=0)
        raw_features_flat = torch.cat(valid_raw_features, dim=0) 
        node_batch_tracker = torch.cat(node_batch_tracker, dim=0)

        current_edge_index = batch['edge_index'].clone()
        current_edge_attr = batch['edge_attr_physics'].clone()

        raw_atomic_vals = raw_features_flat[:, 0]
        atomic_numbers = torch.clamp(raw_atomic_vals - 1, min=0, max=119).long()
        raw_hydrogens = raw_features_flat[:, 4]
        implicit_hydrogens = torch.clamp(raw_hydrogens - 2050, min=0).float()
        
        base_masses = self.mass_lookup[atomic_numbers]
        current_node_masses = base_masses + (implicit_hydrogens * 1.007825)

        harvested_frag_masses = []
        harvested_frag_embeddings = []
        harvested_batch_idx = []

        # Tracks which isolated fragment each node currently belongs to
        current_node_labels = node_batch_tracker.clone()
        
        # Track cumulative log probabilities for the Actor-Critic REINFORCE loss
        cascade_log_probs = torch.zeros(B, device=device) 

        for step in range(max_steps):
            
            # Dynamically calculate Sum Pooling for current subgraphs (IVR)
            global_graph_emb = global_add_pool(X_flat, current_node_labels)
            edge_labels = current_node_labels[current_edge_index[0]]
            global_emb_expanded = global_graph_emb[edge_labels]

            # =================================================================
            # The Unidirectional Stop-Gradient Barrier
            # Protects the continuous MPNN representations from high-variance RL updates
            # =================================================================
            detached_X_flat = X_flat.detach()
            detached_global_emb = global_emb_expanded.detach()

            break_logits = self.edge_head(detached_X_flat, current_edge_index, current_edge_attr, detached_global_emb)
            break_probs = torch.sigmoid(break_logits)

            h_logits = self.h_routing_head(detached_X_flat, current_edge_index, current_edge_attr, detached_global_emb)
            h_probs = torch.softmax(h_logits, dim=-1)

            m_cut = dist.Bernoulli(probs=break_probs)
            m_h = dist.Categorical(probs=h_probs)

            if self.training:
                cut_actions = m_cut.sample() 
                h_actions = m_h.sample()
            else:
                cut_actions = (break_probs > 0.2).float()
                h_actions = torch.argmax(h_probs, dim=-1)

            # =================================================================
            # Accumulate Log Probs for the REINFORCE Policy Loss
            # =================================================================
            cut_lp = m_cut.log_prob(cut_actions)
            h_lp = m_h.log_prob(h_actions)
            step_lp = cut_lp + (h_lp * cut_actions) # Only route H if cut occurs
            
            # [FIXED] Use the static molecule tracker (node_batch_tracker) to avoid CUDA bounds crash
            mol_batch_indices = node_batch_tracker[current_edge_index[0]] 
            mol_level_lp = scatter_sum(step_lp, mol_batch_indices, dim=0, dim_size=B)
            cascade_log_probs += mol_level_lp

            # --- Apply Physics Shifts & Graph Updates ---
            H_MASS = 1.007825
            routing_shifts = (h_actions.float() - 2.0) * H_MASS 
            active_shifts = routing_shifts * cut_actions 
            
            u, v = current_edge_index[0], current_edge_index[1]
            directed_mask = (u < v).float()
            final_shifts = active_shifts * directed_mask
            
            current_node_masses = current_node_masses.clone() 
            current_node_masses.scatter_add_(0, u, final_shifts)
            current_node_masses.scatter_add_(0, v, -final_shifts)

            surviving_mask = (cut_actions == 0)
            current_edge_index = current_edge_index[:, surviving_mask]
            current_edge_attr = current_edge_attr[surviving_mask]

            adj_sparse = to_scipy_sparse_matrix(current_edge_index, num_nodes=X_flat.size(0))
            _, fragment_labels = connected_components(adj_sparse, directed=False)
            fragment_labels = torch.tensor(fragment_labels, device=device, dtype=torch.long)
            
            current_node_labels = fragment_labels

            step_frag_masses = scatter_sum(current_node_masses, fragment_labels, dim=0) + 1.007276 
            step_frag_embeddings = scatter_mean(X_flat, fragment_labels, dim=0)
            
            step_batch_idx, _ = scatter_max(node_batch_tracker, fragment_labels, dim=0)

            harvested_frag_masses.append(step_frag_masses)
            harvested_frag_embeddings.append(step_frag_embeddings)
            harvested_batch_idx.append(step_batch_idx)

            if surviving_mask.all():
                break

        final_masses = torch.cat(harvested_frag_masses, dim=0)
        final_embeddings = torch.cat(harvested_frag_embeddings, dim=0)
        final_batch_idx = torch.cat(harvested_batch_idx, dim=0)

        # Return the accumulated cascade_log_probs for the Actor loss
        return final_masses, final_embeddings, final_batch_idx, cascade_log_probs

    # ==============================================================================
    # PHASE 3 MASTER PIPELINE
    # ==============================================================================
    def forward_phase3(self, batch, max_steps=3):
        """
        Master Inference for Phase 3:
        1. Shatters the molecule into a cascade DAG (and logs discrete probabilities).
        2. Injects Collision Energy.
        3. Predicts final continuous Y-axis intensities.
        """
        device = batch['x'].device
        ce = batch.get('collision_energy', torch.full((batch['x'].shape[0],), 30.0, device=device))
        
        # 1. Autoregressive Cascade Loop
        final_masses, final_embeddings, final_batch_idx, cascade_log_probs = self.forward_phase3_cascade(batch, max_steps)
        
        # 2. Set Transformer (Score Module)
        final_intensities = self.set_transformer(final_embeddings, final_batch_idx, ce)
        
        return final_masses, final_intensities, final_batch_idx, cascade_log_probs


# ==============================================================================
# PHASE 2 RL REWARD ENVIRONMENT (Used as the Reward Signal in Phase 3)
# ==============================================================================
def calculate_cosine_reward(theoretical_mzs, fragment_batch_idx, batched_peaks, batched_masks, tolerance=0.05):
    """
    Calculates Spectral Cosine Similarity with Oracle Partial Credit.
    Used to generate the 'R' signal for the REINFORCE loss.
    """
    B = batched_peaks.shape[0]
    rewards = torch.zeros(B, device=theoretical_mzs.device)
    
    H_MASS = 1.007825
    
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

        if num_frags <= 1:
            rewards[b] = 0.0
            continue

        max_mz_idx = torch.argmax(exp_mzs)
        exp_ints[max_mz_idx] = 0.0 
            
        diffs = torch.abs(frag_mzs.unsqueeze(1) - exp_mzs.unsqueeze(0))
        
        hits_exact = (diffs <= tolerance)
        hits_1h = (torch.abs(diffs - H_MASS) <= tolerance) | (torch.abs(diffs + H_MASS) <= tolerance)
        hits_2h = (torch.abs(diffs - 2*H_MASS) <= tolerance) | (torch.abs(diffs + 2*H_MASS) <= tolerance)
        
        weight_matrix = torch.zeros_like(diffs)
        weight_matrix[hits_2h] = 0.4
        weight_matrix[hits_1h] = 0.7
        weight_matrix[hits_exact] = 1.0
        
        best_weights = weight_matrix.max(dim=0).values
        claimed_intensities = exp_ints * best_weights 
        
        dot_product = torch.sum(claimed_intensities * exp_ints)
        norm_claimed = torch.sqrt(torch.sum(claimed_intensities ** 2)) + 1e-8
        norm_exp = torch.sqrt(torch.sum(exp_ints ** 2)) + 1e-8
        
        base_reward = dot_product / (norm_claimed * norm_exp)
        
        optimal_frags = 5.0
        if num_frags > optimal_frags:
            penalty_factor = math.exp(-0.45 * (num_frags - optimal_frags))
            rewards[b] = base_reward * penalty_factor
        else:
            rewards[b] = base_reward
            
    return rewards