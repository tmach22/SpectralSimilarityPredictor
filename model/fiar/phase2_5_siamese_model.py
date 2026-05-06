import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from deterministic_phase2 import DeterministicPhase2
except ImportError:
    print("[-] Warning: Could not import DeterministicPhase2. Update the import path.")

class Phase3_Partial_SiameseNetwork(nn.Module):
    def __init__(self, config, phase2_checkpoint_path, device, max_fragments=10):
        super().__init__()
        
        # =====================================================================
        # 1. INITIALIZE THE PHASE 2 EXTRACTOR (PARTIAL THAW)
        # =====================================================================
        print("[*] Initializing Phase 2 Extractor (PARTIAL THAW)...")
        self.extractor = DeterministicPhase2(config['model'], max_fragments=max_fragments)
        self.extractor.load_state_dict(torch.load(phase2_checkpoint_path, map_location=device), strict=False)
        
        # [THE SURGICAL FIX] Freeze the MassFormer, Thaw the Edge Head
        thawed_params = 0
        frozen_params = 0
        for name, param in self.extractor.named_parameters():
            # Target the edge prediction and assignment routing layers
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

        # =====================================================================
        # 2. PHASE 2.5 TRAINABLE BOTTLENECK (FULLY THAWED)
        # =====================================================================
        self.context_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads, 
            dropout=0.1, 
            batch_first=True
        )
        
        self.abundance_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 1)
        )
        nn.init.constant_(self.abundance_head[-1].bias, 2.0)

        # Anisotropy LayerNorm to stabilize the shifting motif space
        self.anisotropy_norm = nn.LayerNorm(self.embed_dim)

        # 3. GLOBAL CALIBRATION
        self.final_affine = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def _extract_and_contextualize(self, batch):
        # Gradients are allowed to flow, but will stop at the frozen MassFormer layers
        Z, _, S_dense, _ = self.extractor(batch)
            
        fragment_mass = S_dense.sum(dim=1) 
        padding_mask = (fragment_mass < 1e-4) 
        
        Z_context, _ = self.context_attn(
            query=Z, key=Z, value=Z, key_padding_mask=padding_mask
        )
        
        abundance_logits = self.abundance_head(Z_context).squeeze(-1) 
        abundance_logits = abundance_logits.masked_fill(padding_mask, -1e9)
        W = torch.softmax(abundance_logits, dim=-1) 
        
        Z_stabilized = self.anisotropy_norm(Z_context)
        
        return Z_stabilized, W, padding_mask

    def forward(self, batch_A, batch_B, tau=0.01):
        Z_A, W_A, mask_A = self._extract_and_contextualize(batch_A)
        Z_B, W_B, mask_B = self._extract_and_contextualize(batch_B)
        
        Z_A_norm = F.normalize(Z_A, p=2, dim=-1)
        Z_B_norm = F.normalize(Z_B, p=2, dim=-1)
        
        sim_matrix = torch.bmm(Z_A_norm, Z_B_norm.transpose(1, 2))
        
        mask_B_expanded = mask_B.unsqueeze(1).expand(-1, self.max_fragments, -1)
        sim_matrix = sim_matrix.masked_fill(mask_B_expanded, -1e4)
        
        soft_max_sim = tau * torch.logsumexp(sim_matrix / tau, dim=-1) 
        soft_max_sim = soft_max_sim.masked_fill(mask_A, 0.0)
        
        weighted_sim = torch.sum(soft_max_sim * W_A, dim=1, keepdim=True) 
        final_output = self.final_affine(weighted_sim).squeeze(-1) 
        
        return final_output