import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 1. UNBALANCED SINKHORN OPTIMAL TRANSPORT (Log-Domain)
# =============================================================================
class UnbalancedSinkhornDivergence(nn.Module):
    """
    Computes the Debiased Unbalanced Sinkhorn Divergence between two sets of motif embeddings.
    Uses log-domain iterations for numerical stability.
    """
    def __init__(self, epsilon=0.05, rho=1.0, num_iters=30):
        super().__init__()
        self.epsilon = epsilon
        self.rho = rho
        self.num_iters = num_iters

    def _sinkhorn_unbalanced_log(self, C, a, b):
        """
        C: Cost matrix [B, K, K]
        a: Source marginals (mass-proportional) [B, K]
        b: Target marginals (mass-proportional) [B, K]
        """
        B, K, _ = C.shape
        
        # Add tiny epsilon to prevent log(0)
        log_a = torch.log(a + 1e-15)
        log_b = torch.log(b + 1e-15)
        
        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)
        
        # The contraction factor for unbalanced transport
        lam = self.rho / (self.rho + self.epsilon)

        for _ in range(self.num_iters):
            # Update g (target dual)
            g = lam * (self.epsilon * log_b - self.epsilon * torch.logsumexp((-C + f.unsqueeze(-1)) / self.epsilon, dim=-2))
            # Update f (source dual)
            f = lam * (self.epsilon * log_a - self.epsilon * torch.logsumexp((-C + g.unsqueeze(-2)) / self.epsilon, dim=-1))

        # Reconstruct transport plan
        log_gamma = (f.unsqueeze(-1) + g.unsqueeze(-2) - C) / self.epsilon
        gamma = torch.exp(log_gamma)
        
        # Return transport cost
        return (gamma * C).sum(dim=(-2, -1))

    def forward(self, emb_A, emb_B, a, b):
        """
        emb_A, emb_B: Motif embeddings [B, K, D]
        a, b: Motif mass distributions [B, K]
        """
        # 1. Compute Pairwise Squared Euclidean Distances
        C_AB = torch.cdist(emb_A, emb_B, p=2).pow(2)
        
        # 2. Compute Cross-Transport Cost
        ot_AB = self._sinkhorn_unbalanced_log(C_AB, a, b)
        
        # 3. Compute Auto-Transport Costs for Debiasing
        C_AA = torch.cdist(emb_A, emb_A, p=2).pow(2)
        C_BB = torch.cdist(emb_B, emb_B, p=2).pow(2)
        
        ot_AA = self._sinkhorn_unbalanced_log(C_AA, a, a)
        ot_BB = self._sinkhorn_unbalanced_log(C_BB, b, b)
        
        # 4. Return Debiased Divergence
        # S_eps(A,B) = OT(A,B) - 0.5*OT(A,A) - 0.5*OT(B,B)
        divergence = ot_AB - 0.5 * ot_AA - 0.5 * ot_BB
        return torch.relu(divergence) # Clamp to 0 to prevent numerical negatives


# =============================================================================
# 2. THE MLP REGRESSION HEAD
# =============================================================================
class DistanceToSimilarityHead(nn.Module):
    """
    Converts the Sinkhorn distance and metadata into a predicted Cosine Similarity [0, 1].
    """
    def __init__(self, metadata_dim=2, hidden_dim=64):
        super().__init__()
        # Learnable temperature for exponential decay prior
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        
        # Input: [Sinkhorn_Dist, Mass_Diff, Shift_Fraction, Exp(-Dist/Tau)]
        in_features = 1 + metadata_dim + 1
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), 
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, sinkhorn_dist, metadata):
        tau = torch.exp(self.log_tau).clamp(min=0.01, max=10.0)
        exp_dist = torch.exp(-sinkhorn_dist / tau).unsqueeze(-1)
        
        x = torch.cat([sinkhorn_dist.unsqueeze(-1), metadata, exp_dist], dim=-1)
        # Sigmoid ensures output is strictly bounded [0, 1] for cosine similarity
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


# =============================================================================
# 3. THE FULL STAGE 2 ARCHITECTURE
# =============================================================================
class SiameseSinkhornPredictor(nn.Module):
    # Added meta_dim to the arguments
    def __init__(self, stage1_encoder, meta_dim=2, epsilon=0.05, rho=1.0, num_iters=30):
        super().__init__()
        self.encoder = stage1_encoder 
        
        self.sinkhorn = UnbalancedSinkhornDivergence(epsilon=epsilon, rho=rho, num_iters=num_iters)
        # Pass the meta_dim to the head
        self.sim_head = DistanceToSimilarityHead(metadata_dim=meta_dim)

    def _get_mass_marginals(self, S, batched_graphs):
        """
        Calculates mass-proportional marginals (a_i) for the Sinkhorn algorithm.
        Instead of uniform [1/K, 1/K...], we sum the atomic masses of the atoms assigned to each motif.
        """
        B, N, K = S.shape
        # Assuming batched_graphs.x contains atomic mass or atomic number in its features.
        # For simplicity, we fallback to uniform degree-based if exact mass isn't parsed easily here.
        # A more precise implementation would extract mass from the `mol` object.
        
        # Simple proxy: Sum of assignment probabilities across the molecule.
        # This naturally encodes if Molecule A has 20 atoms and Molecule B has 40.
        motif_masses = S.sum(dim=1) # [B, K]
        return motif_masses

    def forward(self, b_A, b_B, brics_A, brics_B, b_massA, b_massB, b_meta, 
                A_bde_A=None, A_bde_B=None, shift_frac=None, ce_norm=None):
        """
        Takes unwrapped batch elements to cleanly compute embeddings and physics states.
        """
        # 1. Extract Motif Embeddings & Assignments (Stage 1 Forward Pass)
        # Note: We pass the specific BDE and collision energy inputs for each molecule
        Z_A, S_A, _, cut_probs_A, frag_probs_A = self.encoder.forward_graph(
            b_A, brics_A, A_bde_norm=A_bde_A, ce_norm=ce_norm
        )
        Z_B, S_B, _, cut_probs_B, frag_probs_B = self.encoder.forward_graph(
            b_B, brics_B, A_bde_norm=A_bde_B, ce_norm=ce_norm
        )
        
        # Normalize embeddings (critical for fixed epsilon Sinkhorn stability)
        Z_A = F.normalize(Z_A, p=2, dim=-1)
        Z_B = F.normalize(Z_B, p=2, dim=-1)

        # 2. Calculate Mass-Proportional Marginals
        # We pass the extracted structures so the Sinkhorn algorithm 
        # understands the physical size of the fragments being transported.
        a = self._get_mass_marginals(S_A, b_A)
        b = self._get_mass_marginals(S_B, b_B)

        # 3. Compute Unbalanced Sinkhorn Divergence
        sinkhorn_dist = self.sinkhorn(Z_A, Z_B, a, b)

        # 4. Predict Final Cosine Similarity
        # b_meta contains [Mass_Diff, Shift_Fraction]
        pred_sim = self.sim_head(sinkhorn_dist, b_meta)
        
        # Return the prediction, distance, and ALL intermediate physics tensors
        # so the training loop can apply the physics TV loss.
        return (pred_sim, sinkhorn_dist, S_A, S_B, 
                cut_probs_A, cut_probs_B, frag_probs_A, frag_probs_B)


# =============================================================================
# 4. SKEW-RESISTANT LOSS FUNCTION
# =============================================================================
class CombinedSimilarityLoss(nn.Module):
    """
    Combines Smooth L1 Regression with Margin Ranking to defeat the MS/MS Skew Problem.
    """
    def __init__(self, alpha=1.0, beta=0.1, margin=0.03):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, pred, target):
        # 1. Pointwise Regression (Robust to boundary noise at 0 and 1)
        L_reg = F.smooth_l1_loss(pred, target, reduction='mean', beta=0.1)

        # 2. Pairwise Margin Ranking (Preserves relative ordering)
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)
        
        # Only rank pairs that have a meaningful ground-truth difference (>0.05)
        mask = (target_diff.abs() > 0.05).float()
        sign = torch.sign(target_diff)
        
        L_rank = (F.relu(self.margin - sign * pred_diff) * mask).sum() / mask.sum().clamp(min=1)

        return (self.alpha * L_reg) + (self.beta * L_rank)