import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch, to_dense_adj

try:
    from classifier_siamese_model import UnpooledMassFormerEncoder
except ImportError:
    print("[-] Warning: UnpooledMassFormerEncoder not found. Check import path.")
    UnpooledMassFormerEncoder = nn.Module

class EdgeKeepHead(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim=2, hidden_dim=256):
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


class DeterministicPhase2(nn.Module):
    def __init__(self, model_config, max_fragments=10, lpe_dim=8):
        super().__init__()
        self.max_fragments = max_fragments
        
        self.graph_encoder = UnpooledMassFormerEncoder(model_config, checkpoint_path=None)
        self.emb_dim = model_config.get('embed_dim', 768) if model_config.get('embed_dim') != -1 else 768
        
        self.edge_keep_head = EdgeKeepHead(node_emb_dim=self.emb_dim, edge_attr_dim=2, hidden_dim=256)
        
        # [UPDATED] GCN dimension expanded to accept the Laplacian Positional Encodings
        self.fragment_gcn = GCNConv(self.emb_dim + lpe_dim, self.emb_dim)
        
        self.assignment_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.SiLU(),
            nn.Linear(self.emb_dim // 2, self.max_fragments)
        )

    def forward(self, batch):
        device = batch['x'].device
        B = batch['x'].shape[0]
        
        # =====================================================================
        # 1. BASE EMBEDDING EXTRACTION
        # =====================================================================
        X_base = self.graph_encoder({'gf_v2_data': batch})
        X = X_base[0][:, 1:, :] if isinstance(X_base, tuple) else X_base[:, 1:, :]
        
        valid_X_list, node_batch_tracker = [], []
        for b in range(B):
            true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
            valid_X_list.append(X[b, :true_nodes, :])
            node_batch_tracker.append(torch.full((true_nodes,), b, device=device, dtype=torch.long))
            
        X_flat = torch.cat(valid_X_list, dim=0)
        node_batch_tracker = torch.cat(node_batch_tracker, dim=0)

        # =====================================================================
        # 2. THE STRAIGHT-THROUGH ESTIMATOR (STE) MASK
        # =====================================================================
        keep_logits = self.edge_keep_head(X_flat, batch['edge_index'], batch['edge_attr_physics'])
        p_keep = torch.sigmoid(keep_logits)
        
        m_hard = (p_keep > 0.5).float()
        m_diff = m_hard.detach() - p_keep.detach() + p_keep

        # =====================================================================
        # 3. SYMMETRY BREAKER (LPE INJECTION)
        # =====================================================================
        if 'pe' in batch:
            X_flat_lpe = torch.cat([X_flat, batch['pe']], dim=-1)
        else:
            # Fallback tensor if dataset is missing LPEs
            X_flat_lpe = torch.cat([X_flat, torch.zeros(X_flat.size(0), 8, device=device)], dim=-1)

        X_shattered = self.fragment_gcn(X_flat_lpe, batch['edge_index'], edge_weight=m_diff)
        
        S_logits_flat = self.assignment_mlp(X_shattered)
        S_flat = torch.softmax(S_logits_flat, dim=-1)

        S_dense, node_mask = to_dense_batch(S_flat, node_batch_tracker)

        # =====================================================================
        # 4. [NEW] DENSE REACHABILITY LOSS (L_reach)
        # =====================================================================
        # 4a. Build the dense shattered adjacency matrix [Batch, N, N]
        A_dense = to_dense_adj(batch['edge_index'], batch=node_batch_tracker, edge_attr=m_diff.detach())
        
        # 4b. Add self-loops (Nodes can always reach themselves)
        I = torch.eye(A_dense.size(1), device=device).unsqueeze(0)
        R = (A_dense + I > 0).float()
        
        # 4c. Transitive Closure (Matrix Squaring)
        # 7 iterations maps paths up to 2^7 = 128 hops.
        for _ in range(7):
            R = torch.bmm(R, R)
            R = (R > 0).float()
            
        # 4d. Dense Co-Assignment Prediction
        Y_hat = torch.bmm(S_dense, S_dense.transpose(1, 2))
        Y_hat = torch.clamp(Y_hat, 1e-6, 1.0 - 1e-6)
        
        # 4e. Calculate Global BCE Loss (Masking out padding nodes)
        mask_2d = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        L_reach = F.binary_cross_entropy(Y_hat[mask_2d], R[mask_2d])

        # =====================================================================
        # 5. THE BAG OF MOTIFS GENERATION
        # =====================================================================
        X_orig_dense, _ = to_dense_batch(X_flat, node_batch_tracker)
        
        S_dense = torch.nan_to_num(S_dense, nan=0.0)
        S_dense = S_dense.masked_fill(~node_mask.unsqueeze(-1), 0.0)

        Z = torch.bmm(S_dense.transpose(1, 2), X_orig_dense)
        
        # Return L_reach so the distillation script can minimize it
        return Z, p_keep, S_dense, L_reach