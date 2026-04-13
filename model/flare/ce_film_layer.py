import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CEEmbedder(nn.Module):
    def __init__(self, out_dim=32, n_freqs=8):
        super().__init__()
        freqs = torch.exp(
            torch.linspace(0, np.log(100.0), n_freqs)
        )
        self.register_buffer('freqs', freqs)
        fourier_dim = 2 * n_freqs + 1 
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, ce_norm):
        angles = ce_norm * self.freqs.unsqueeze(0) 
        ff = torch.cat([torch.sin(angles), torch.cos(angles), ce_norm], dim=-1)
        return self.mlp(ff)


class CEFiLMLayer(nn.Module):
    def __init__(self, feature_dim, ce_embed_dim=32):
        super().__init__()
        self.gamma_head = nn.Linear(ce_embed_dim, feature_dim)
        self.beta_head = nn.Linear(ce_embed_dim, feature_dim)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, features, ce_embed):
        gamma = self.gamma_head(ce_embed) 
        beta = self.beta_head(ce_embed)   
        if features.dim() == 3:
            return gamma.unsqueeze(1) * features + beta.unsqueeze(1)
        elif features.dim() == 4:
            return gamma.unsqueeze(1).unsqueeze(1) * features + beta.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected feature dim: {features.dim()}")


class CEAwareDenseGCNLayer(nn.Module):
    def __init__(self, hidden_dim, ce_embed_dim=32):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.node_film = CEFiLMLayer(hidden_dim, ce_embed_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, A_brics, ce_embed, A_bde_norm=None):
        if A_bde_norm is not None:
            A_eff = A_brics * A_bde_norm
        else:
            A_eff = A_brics
        d = A_eff.sum(dim=-1).clamp(min=1e-5)
        d_inv_sqrt = d.pow(-0.5)
        A_norm = d_inv_sqrt.unsqueeze(-1) * A_eff * d_inv_sqrt.unsqueeze(1)
        out = torch.bmm(A_norm, x)
        out = F.leaky_relu(self.lin(out), 0.1)
        out = self.node_film(out, ce_embed)
        out = self.norm(out)
        return out


class CEBDEFragmentationGate(nn.Module):
    """
    Per-bond fragmentation probability conditioned on both CE and BDE.
    """
    def __init__(self, hidden_dim=128, ce_embed_dim=32):
        super().__init__()
        self.threshold_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 1) # Raw logits for Softplus threshold
        )
        
        # EXPERT FIX: T-Zero Initialization
        # Biases the network output strongly negative so the initial T_eff begins cold
        nn.init.constant_(self.threshold_net[-1].bias, -3.0)
        
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self.ce_proj = nn.Linear(ce_embed_dim, 1)
        nn.init.zeros_(self.ce_proj.weight)
        nn.init.zeros_(self.ce_proj.bias)

    def forward(self, node_features, A_bde_norm, ce_embed, A_brics):
        B, N, D = node_features.shape
        T = torch.exp(self.log_temperature).clamp(min=0.05) + 0.01

        ce_scalar = torch.sigmoid(self.ce_proj(ce_embed)) 
        degree = (A_brics > 0).float().sum(dim=-1, keepdim=True).clamp(min=1) 
        atom_bde = (A_bde_norm * (A_brics > 0).float()).sum(dim=-1, keepdim=True) / degree 
        atom_input = torch.cat([node_features, atom_bde], dim=-1) 
        
        atom_threshold = self.threshold_net(atom_input) 
        
        # EXPERT FIX: Anchor heavily on the physical BDE
        anchored_threshold = atom_bde + 0.2 * atom_threshold

        bond_thresh = (anchored_threshold.unsqueeze(2) + anchored_threshold.unsqueeze(1)) / 2.0
        bond_thresh = bond_thresh.squeeze(-1) 

        ce_expanded = ce_scalar.unsqueeze(1).expand(B, N, N) 
        frag_logits = (ce_expanded - bond_thresh) / T 
        frag_probs = torch.sigmoid(frag_logits)

        bond_mask = (A_brics > 0).float()
        frag_probs = frag_probs * bond_mask

        return frag_probs