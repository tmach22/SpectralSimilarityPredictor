import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CurriculumScheduler:
    @staticmethod
    def get_lambdas(epoch):
        if epoch <= 15:
            return {
                'lambda_act': 10.0,   # Fixed High
                'lambda_sev': 0.1,    
                'lambda_smo': 0.1,    
                'lambda_mc': 0.0      
            }
        elif epoch <= 40:
            progress = (epoch - 16) / (40 - 16) 
            l_sev = 0.1 + 0.9 * progress 
            l_smo = 0.1 + 0.4 * progress # Capped at 0.5
            
            return {
                'lambda_act': 10.0,   # Fixed High
                'lambda_sev': float(l_sev),
                'lambda_smo': float(l_smo),
                'lambda_mc': 0.1      
            }
        else:
            return {
                'lambda_act': 10.0,   # Fixed High
                'lambda_sev': 1.0,    
                'lambda_smo': 0.5,    # Capped
                'lambda_mc': 0.1      
            }

    @staticmethod
    def get_phase(epoch):
        if epoch <= 15: return "Phase 1: Warmup & Activation"
        if epoch <= 40: return "Phase 2: Partial Thaw & Ramp"
        return "Phase 3: Joint Fine-Tuning"

class UnifiedPhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_act = 10.0
        self.lambda_sev = 0.1
        self.lambda_smo = 0.1
        self.lambda_mc  = 0.0

    def forward(self, loss_data, S, A_brics, cut_probs=None, A_bde_norm=None, 
                frag_probs=None, ce_norm=None, **kwargs):

        if cut_probs is None or ce_norm is None:
            return loss_data, {"L_total": loss_data.item(), "L_flare": loss_data.item()}

        B, N, K = S.size()
        edge_mask = (A_brics > 0).float()
        
        num_edges = torch.sum(edge_mask, dim=(1, 2))
        num_edges = torch.clamp(num_edges, min=1.0)

        # ---------------------------------------------------------
        # 1. Activation Bonus (L_active) - EXPERT FIX: L1 ABSOLUTE ERROR
        # ---------------------------------------------------------
        ce_norm_flat = ce_norm.view(B)
        target_cuts = torch.clamp(1.0 + 4.0 * ce_norm_flat, min=1.0, max=8.0)
        predicted_cuts = torch.sum(cut_probs * edge_mask, dim=(1, 2)) / 2.0
        
        # Replaced Hinge with Absolute Error to maintain gradient flow
        l_act = torch.mean(torch.abs(target_cuts - predicted_cuts))

        # ---------------------------------------------------------
        # Distance Matrix Calculation
        # ---------------------------------------------------------
        diff = S.unsqueeze(2) - S.unsqueeze(1)
        l1_dist = torch.norm(diff, p=1, dim=-1) 
        l2_dist_sq = torch.sum(diff ** 2, dim=-1) 

        # ---------------------------------------------------------
        # 2. The Repulsive Force (L_sever - Cut Consistency)
        # ---------------------------------------------------------
        exp_penalty = torch.exp(-l1_dist)
        l_sev_batch = torch.sum(cut_probs * edge_mask * exp_penalty, dim=(1, 2)) / num_edges
        l_sev = torch.mean(l_sev_batch)

        # ---------------------------------------------------------
        # 3. The Attractive Force (L_smooth - Laplacian Smoothing)
        # ---------------------------------------------------------
        l_smo_batch = torch.sum((1.0 - cut_probs) * edge_mask * l2_dist_sq, dim=(1, 2)) / num_edges
        l_smo = torch.mean(l_smo_batch)

        # ---------------------------------------------------------
        # 4. Spectral MinCut (L_mincut)
        # ---------------------------------------------------------
        l_mc = 0.0
        if self.lambda_mc > 0:
            S_T_A = torch.matmul(S.transpose(1, 2), A_brics)
            num = torch.diagonal(torch.matmul(S_T_A, S), dim1=1, dim2=2).sum(dim=-1)
            
            D_brics = torch.diag_embed(A_brics.sum(dim=-1))
            S_T_D = torch.matmul(S.transpose(1, 2), D_brics)
            den = torch.diagonal(torch.matmul(S_T_D, S), dim1=1, dim2=2).sum(dim=-1) + 1e-6
            
            l_mc = torch.mean(-num / den)

        # =========================================================
        # COMBINE AND RETURN
        # =========================================================
        total_loss = (
            loss_data + 
            self.lambda_act * l_act +
            self.lambda_sev * l_sev +
            self.lambda_smo * l_smo +
            self.lambda_mc  * l_mc
        )

        loss_components = {
            "L_total": total_loss.item(),
            "L_flare": loss_data.item(),
            "L_act": l_act.item() * self.lambda_act,
            "L_sev": l_sev.item() * self.lambda_sev,
            "L_smo": l_smo.item() * self.lambda_smo,
            "L_mc": l_mc.item() * self.lambda_mc if isinstance(l_mc, torch.Tensor) else 0.0
        }

        return total_loss, loss_components