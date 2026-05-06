import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from deterministic_phase2 import DeterministicPhase2
except ImportError:
    print("[-] Warning: Could not import DeterministicPhase2. Update the import path.")

class SinusoidalMassEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, mass):
        half_dim = self.embed_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=mass.device) * -(np.log(10000.0) / half_dim))
        emb = mass.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class Phase3_Partial_SiameseNetwork(nn.Module):
    def __init__(self, config, phase2_checkpoint_path, device, max_fragments=10):
        super().__init__()
        
        # 1. INITIALIZE THE PHASE 2 EXTRACTOR (PARTIAL THAW)
        print("[*] Initializing Phase 2 Extractor (PARTIAL THAW)...")
        self.extractor = DeterministicPhase2(config['model'], max_fragments=max_fragments)
        self.extractor.load_state_dict(torch.load(phase2_checkpoint_path, map_location=device), strict=False)
        
        # Freeze the MassFormer, Thaw the Edge Head
        thawed_params = 0
        frozen_params = 0
        for name, param in self.extractor.named_parameters():
            if "edge" in name.lower() or "assignment" in name.lower() or "head" in name.lower():
                param.requires_grad = True
                thawed_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
                
        print(f"[+] Backbone Frozen Parameters: {frozen_params:,}")
        print(f"[+] Backbone Thawed Parameters (Edge Routing): {thawed_params:,}")
        
        self.embed_dim = config['model'].get('embed_dim', 768) if config['model'].get('embed_dim') != -1 else 768
        self.max_fragments = max_fragments
        num_heads = config['model'].get('num_heads', 8)

        # 2. PHASE 2.5 THERMODYNAMIC ADAPTERS 
        print("[*] Initializing Phase 2.5 Thermodynamic Interventions...")
        
        self.electronic_film = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, self.embed_dim), nn.Sigmoid() 
        )
        self.mass_encoder = SinusoidalMassEncoder(self.embed_dim)
        self.post_mass_layernorm = nn.LayerNorm(self.embed_dim)
        self.context_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # 3. GLOBAL CALIBRATION (RESTORED FOR CONTINUOUS HUBER LOSS)
        self.final_affine = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid() 
        )

    def _extract_and_contextualize(self, batch):
        Z, _, S_dense, _ = self.extractor(batch)
            
        if 'node_electronics' in batch: node_elec = batch['node_electronics'].unsqueeze(-1).float() 
        else: node_elec = torch.ones(S_dense.size(0), S_dense.size(1), 1, device=Z.device)
            
        if 'node_masses' in batch: node_mass = batch['node_masses'].unsqueeze(-1).float() 
        else: node_mass = torch.ones(S_dense.size(0), S_dense.size(1), 1, device=Z.device) * 12.0
            
        motif_elec = torch.bmm(S_dense.transpose(1, 2), node_elec) 
        motif_mass = torch.bmm(S_dense.transpose(1, 2), node_mass).squeeze(-1) 
        
        electronic_gate = self.electronic_film(motif_elec) 
        Z = (Z * electronic_gate) + Z  
        
        mass_embeddings = self.mass_encoder(motif_mass)
        Z = self.post_mass_layernorm(Z + mass_embeddings)
        
        padding_mask = (motif_mass < 1e-4) 
        Z_context, _ = self.context_attn(query=Z, key=Z, value=Z, key_padding_mask=padding_mask)
        
        return Z_context, motif_mass, padding_mask

    def forward(self, batch_A, batch_B, tau=0.01):
        Z_A, mass_A, mask_A = self._extract_and_contextualize(batch_A)
        Z_B, mass_B, mask_B = self._extract_and_contextualize(batch_B)
        
        Z_A_norm = F.normalize(Z_A, p=2, dim=-1)
        Z_B_norm = F.normalize(Z_B, p=2, dim=-1)
        
        sim_matrix = torch.bmm(Z_A_norm, Z_B_norm.transpose(1, 2))
        mask_B_expanded = mask_B.unsqueeze(1).expand(-1, self.max_fragments, -1)
        sim_matrix = sim_matrix.masked_fill(mask_B_expanded, -1e4)
        
        soft_max_sim = tau * torch.logsumexp(sim_matrix / tau, dim=-1) 
        soft_max_sim = soft_max_sim.masked_fill(mask_A, 0.0)
        
        total_mass_A = mass_A.sum(dim=1, keepdim=True) + 1e-6
        mass_fraction_A = mass_A / total_mass_A 
        
        weighted_sim = torch.sum(soft_max_sim * mass_fraction_A, dim=1, keepdim=True) 
        
        # Squeeze through the Sigmoid for the bounded Huber comparison
        final_output = self.final_affine(weighted_sim).squeeze(-1) 
        
        return final_output