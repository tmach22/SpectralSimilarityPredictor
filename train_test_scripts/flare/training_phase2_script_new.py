import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import math
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
    from siamese_sinkhorn_new import SiameseSinkhornPredictor, CombinedSimilarityLoss
    from unified_physics_loss import UnifiedPhysicsLoss, CurriculumScheduler
    from cross_modal_pretrainer_new import CrossModalFlarePretrainer
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    sys.exit(1)


def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if os.path.exists(custom_path):
        with open(custom_path, 'r', encoding='utf-8') as f:
            custom_config = yaml.safe_load(f)
        for section, subdict in custom_config.items():
            if isinstance(subdict, dict):
                if section not in config:
                    config[section] = {}
                for k, v in subdict.items():
                    config[section][k] = v
            else:
                config[section] = subdict
    return config


# =============================================================================
# EXPERT FIX: 3-Phase Sinkhorn Learning Schedule
# =============================================================================
def get_learning_rates(epoch, total_epochs, base_head_lr=3e-4, phase2_start=6, phase3_start=16):
    """
    Phase 1: Frozen Encoder, Train Heads Only (Stable OT geometry)
    Phase 2: Unfreeze Encoder, 10x Differential LR + Linear Warmup
    Phase 3: Cosine Decay for joint refinement
    """
    if epoch < phase2_start:
        return True, 0.0, base_head_lr
        
    elif epoch < phase3_start:
        warmup_steps = phase3_start - phase2_start
        current_step = epoch - phase2_start + 1
        base_enc_lr = base_head_lr * 0.1
        encoder_lr = base_enc_lr * (current_step / warmup_steps) 
        return False, encoder_lr, base_head_lr
        
    else:
        decay_epochs = total_epochs - phase3_start + 1
        current_step = epoch - phase3_start
        cosine_factor = 0.5 * (1 + math.cos(math.pi * current_step / decay_epochs))
        encoder_lr = (base_head_lr * 0.1) * cosine_factor
        head_lr = base_head_lr * cosine_factor
        return False, max(encoder_lr, 1e-7), max(head_lr, 1e-6)


def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # --- Datasets ---
    print("\n[*] Initializing Stage 2 Datasets...")
    train_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.train_pairs,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=True
    )
    val_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.val_pairs,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=binary_collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=binary_collate_fn, num_workers=4
    )

    # --- Model ---
    print("\n[*] Loading Pretrained Stage 1 Encoder...")
    
    stage1_encoder = CrossModalFlarePretrainer(
        model_config=full_config['model'],
        num_motifs=15, 
        drop_edge_p=0.30 
    )
    
    stage1_encoder.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))

    model = SiameseSinkhornPredictor(
        stage1_encoder=stage1_encoder,
        meta_dim=args.meta_dim,
        epsilon=0.05,
        rho=1.0,
        num_iters=30
    ).to(device)
    print("[+] Siamese-Sinkhorn Architecture Initialized.")

    # --- Optimiser Setup ---
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
    head_params = [p for n, p in model.named_parameters() if 'encoder' not in n]

    optimizer = AdamW([
        {'params': encoder_params, 'lr': 0.0, 'weight_decay': 0.01}, 
        {'params': head_params, 'lr': args.learning_rate, 'weight_decay': 1e-4} 
    ])

    sim_criterion = CombinedSimilarityLoss(alpha=1.0, beta=0.1, margin=0.03)
    physics_loss_fn = UnifiedPhysicsLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n[*] Commencing Stage 2: Sinkhorn Similarity Training (3-Phase Schedule)...")

    for epoch in range(1, args.epochs + 1):
        # 1. Apply Phase Schedule
        encoder_frozen, enc_lr, head_lr = get_learning_rates(epoch, args.epochs, base_head_lr=args.learning_rate)
        
        # Toggle Gradients & Update LRs
        for p in encoder_params:
            p.requires_grad = not encoder_frozen
            
        optimizer.param_groups[0]['lr'] = enc_lr
        optimizer.param_groups[1]['lr'] = head_lr
        
        phase_name = "Phase 1 (Frozen)" if encoder_frozen else ("Phase 2 (Warmup)" if epoch < 16 else "Phase 3 (Decay)")

        # 2. Setup Physics Curriculum Offset
        effective_epoch = epoch + args.stage1_epochs
        lambdas = CurriculumScheduler.get_lambdas(effective_epoch)

        physics_loss_fn.lambda_bde = lambdas['lambda_bde'] * args.guardrail_weight
        physics_loss_fn.lambda_shift = lambdas['lambda_shift'] * args.guardrail_weight
        physics_loss_fn.lambda_ce = lambdas['lambda_ce'] * args.guardrail_weight
        physics_loss_fn.lambda_thermo = lambdas['lambda_thermo'] * args.guardrail_weight

        # --- TRAINING ---
        model.train()
        total_train_loss = 0.0
        total_sim_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [{phase_name}]")

        for batch in train_bar:
            (b_A, b_B, b_meta, b_massA, b_massB, target_sim,
             brics_A, brics_B, bde_A, bde_B, shift_frac, ce_norm) = batch

            def to_dev(x):
                return {k: v.to(device) if hasattr(v, 'to') else v
                        for k, v in x.items()} if isinstance(x, dict) else x.to(device)

            b_A = to_dev(b_A)
            b_B = to_dev(b_B)
            b_meta = b_meta.to(device)
            b_massA = b_massA.to(device)
            b_massB = b_massB.to(device)
            brics_A = brics_A.to(device)
            brics_B = brics_B.to(device)
            bde_A = bde_A.to(device) if bde_A is not None else None
            bde_B = bde_B.to(device) if bde_B is not None else None
            shift_frac = shift_frac.to(device)
            ce_norm = ce_norm.to(device)
            target_sim = target_sim.view(-1).float().to(device) 

            optimizer.zero_grad()

            (pred_sim, sinkhorn_dist, 
             S_A, S_B, 
             cut_probs_A, cut_probs_B, 
             frag_probs_A, frag_probs_B) = model(
                b_A, b_B, brics_A, brics_B,
                b_massA, b_massB, b_meta,
                A_bde_A=bde_A, A_bde_B=bde_B,
                shift_frac=shift_frac, ce_norm=ce_norm
            )

            # 1. Primary Siamese Regression Loss
            loss_sim = sim_criterion(pred_sim, target_sim)

            # 2. Physics Guardrail Loss Calculation
            sf = shift_frac.squeeze(-1)
            m_A = b_massA.squeeze(-1)
            m_B = b_massB.squeeze(-1)

            # EXPERT FIX: Calculate physics independently for differing graph sizes
            loss_phys_A, _ = physics_loss_fn(
                loss_data=torch.tensor(0.0, device=device),
                S=S_A,
                A_brics=brics_A,
                cut_probs=cut_probs_A,
                A_bde_norm=bde_A,
                frag_probs=frag_probs_A,
                ce_norm=ce_norm,
                pred_sim=pred_sim,
                mass_A=m_A,
                mass_B=m_B,
                shift_fractions=sf,
            )

            loss_phys_B, _ = physics_loss_fn(
                loss_data=torch.tensor(0.0, device=device),
                S=S_B,
                A_brics=brics_B,
                cut_probs=cut_probs_B,
                A_bde_norm=bde_B,
                frag_probs=frag_probs_B,
                ce_norm=ce_norm,
                pred_sim=pred_sim,
                mass_A=m_A,
                mass_B=m_B,
                shift_fractions=sf,
            )

            # Average the scalar physics losses and combine with Siamese Regression loss
            loss_total = loss_sim + ((loss_phys_A + loss_phys_B) / 2.0)

            loss_total.backward()
            
            # Protect network from OT gradients during initial thaw
            clip_val = 0.5 if (epoch >= 6 and epoch <= 8) else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            
            optimizer.step()

            total_train_loss += loss_total.item()
            total_sim_loss += loss_sim.item()
            
            train_bar.set_postfix({
                'Total': f"{loss_total.item():.4f}",
                'SimReg': f"{loss_sim.item():.4f}"
            })

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0

        with torch.no_grad():
            for batch in val_loader:
                (b_A, b_B, b_meta, b_massA, b_massB, target_sim,
                 brics_A, brics_B, bde_A, bde_B, shift_frac, ce_norm) = batch

                b_A = to_dev(b_A)
                b_B = to_dev(b_B)
                b_meta = b_meta.to(device)
                b_massA = b_massA.to(device)
                b_massB = b_massB.to(device)
                brics_A = brics_A.to(device)
                brics_B = brics_B.to(device)
                bde_A = bde_A.to(device) if bde_A is not None else None
                bde_B = bde_B.to(device) if bde_B is not None else None
                shift_frac = shift_frac.to(device)
                ce_norm = ce_norm.to(device)
                target_sim = target_sim.view(-1).float().to(device)

                (pred_sim, sinkhorn_dist, 
                 S_A, S_B, 
                 cut_probs_A, cut_probs_B, 
                 frag_probs_A, frag_probs_B) = model(
                    b_A, b_B, brics_A, brics_B,
                    b_massA, b_massB, b_meta,
                    A_bde_A=bde_A, A_bde_B=bde_B,
                    shift_frac=shift_frac, ce_norm=ce_norm
                )

                v_loss_sim = sim_criterion(pred_sim, target_sim)
                
                sf_v = shift_frac.squeeze(-1)
                m_A_v = b_massA.squeeze(-1)
                m_B_v = b_massB.squeeze(-1)
                
                v_loss_phys_A, _ = physics_loss_fn(
                    loss_data=torch.tensor(0.0, device=device),
                    S=S_A,
                    A_brics=brics_A,
                    cut_probs=cut_probs_A,
                    A_bde_norm=bde_A,
                    frag_probs=frag_probs_A,
                    ce_norm=ce_norm,
                    pred_sim=pred_sim,
                    mass_A=m_A_v,
                    mass_B=m_B_v,
                    shift_fractions=sf_v,
                )
                
                v_loss_phys_B, _ = physics_loss_fn(
                    loss_data=torch.tensor(0.0, device=device),
                    S=S_B,
                    A_brics=brics_B,
                    cut_probs=cut_probs_B,
                    A_bde_norm=bde_B,
                    frag_probs=frag_probs_B,
                    ce_norm=ce_norm,
                    pred_sim=pred_sim,
                    mass_A=m_A_v,
                    mass_B=m_B_v,
                    shift_fractions=sf_v,
                )
                
                v_loss_total = v_loss_sim + ((v_loss_phys_A + v_loss_phys_B) / 2.0)
                total_val_loss += v_loss_total.item()

                # Regression Metrics
                total_mse += F.mse_loss(pred_sim, target_sim).item()
                total_mae += F.l1_loss(pred_sim, target_sim).item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        avg_mse = total_mse / max(len(val_loader), 1)
        avg_mae = total_mae / max(len(val_loader), 1)

        print(f"Epoch {epoch} | EncLR: {enc_lr:.1e} | TrainLoss: {avg_train_loss:.4f} | "
              f"ValLoss: {avg_val_loss:.4f} | ValMSE: {avg_mse:.4f} | ValMAE: {avg_mae:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "siamese_sinkhorn_best.pt"))
            print(" -> Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--train_pairs", type=str, required=True)
    parser.add_argument("--val_pairs", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trained_model/stage2_siamese")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4) # Base head LR
    parser.add_argument("--meta_dim", type=int, default=81)
    parser.add_argument("--stage1_epochs", type=int, default=20,
                        help="Number of Stage 1 epochs completed (for curriculum offset)")
    parser.add_argument("--guardrail_weight", type=float, default=0.1,
                        help="Global scale for physics penalties during Stage 2")
    args = parser.parse_args()
    train_stage2(args)