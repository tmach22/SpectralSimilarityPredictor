import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import math
import wandb
from pathlib import Path
import warnings

# Mute PyTorch Nested Tensor prototype warnings
warnings.filterwarnings("ignore", message=".*nested tensors.*")

# --- SETUP PATHS ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'fiar')) 
sys.path.insert(0, os.path.join(cwd, 'model', 'fiar'))
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model', 'bifurcate'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src'))
sys.path.insert(0, os.path.join(str(cwd), 'model', 'flare'))

try:
    from phase2_dataloader import Phase2EdgeDataset, phase2_collate_fn
    from phase3_model import DESAFNet, calculate_cosine_reward
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    sys.exit(1)

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    if os.path.exists(custom_path):
        with open(custom_path, 'r', encoding='utf-8') as f: custom_config = yaml.safe_load(f)
        for section, subdict in custom_config.items():
            if isinstance(subdict, dict):
                if section not in config: config[section] = {}
                for k, v in subdict.items(): config[section][k] = v
            else: config[section] = subdict
    return config

# ==============================================================================
# DOUBLE WEIGHT SURGERY (1-WL BYPASS & IVR EXPANSION)
# ==============================================================================
def load_phase2_weights_with_surgery(model, weights_path, device, emb_dim=768):
    """
    Intercepts the Phase 2 state_dict and applies zero-padding surgery to:
    1. The MassFormer input layer (to accommodate Ring/Aromaticity flags).
    2. The Phase 2 Heads (to accommodate the new 768-dim global IVR vector).
    """
    print(f"[*] Loading Phase 2 weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)

    for key, tensor in list(state_dict.items()):
        
        # SURGERY 1: MassFormer Input Projection (1-WL Bypass)
        if tensor.dim() == 2 and tensor.shape[1] == 5 and "graph_encoder" in key:
            print(f"[*] Found input projection layer: '{key}' {tensor.shape}")
            zero_padding = torch.zeros((tensor.shape[0], 2), device=device)
            surgically_expanded_tensor = torch.cat([tensor, zero_padding], dim=1)
            state_dict[key] = surgically_expanded_tensor
            print(f"[+] Surgically expanded '{key}' to {surgically_expanded_tensor.shape}")

        # SURGERY 2: Phase 2 Heads (IVR / Degrees of Freedom Dampening)
        elif key in ["edge_head.mlp.0.weight", "h_routing_head.mlp.0.weight"]:
            print(f"[*] Found Phase 2 Head: '{key}' {tensor.shape}")
            zero_padding = torch.zeros((tensor.shape[0], emb_dim), device=device)
            surgically_expanded_tensor = torch.cat([tensor, zero_padding], dim=1)
            state_dict[key] = surgically_expanded_tensor
            print(f"[+] Surgically expanded '{key}' to {surgically_expanded_tensor.shape}")

    # strict=False allows us to ignore the uninitialized Phase 3 Set Transformer keys
    model.load_state_dict(state_dict, strict=False)
    print("[+] Double Weight Surgery Complete! Pre-trained heuristics safely preserved.")

# ==============================================================================
# PHASE 3 SPECTRAL ENTROPY LOSS (The "Critic" Objective)
# ==============================================================================
def spectral_entropy_loss(theoretical_mzs, predicted_ints, fragment_batch_idx, batched_peaks, batched_masks, tolerance=0.05):
    device = theoretical_mzs.device
    B = batched_peaks.shape[0]
    batch_loss = torch.tensor(0.0, device=device)
    valid_batches = 0

    for b in range(B):
        frag_mzs = theoretical_mzs[fragment_batch_idx == b]
        pred_ints = predicted_ints[fragment_batch_idx == b]
        
        peaks = batched_peaks[b]
        mask = batched_masks[b] 
        valid_peaks = peaks[~mask]
        
        if len(frag_mzs) == 0 or len(valid_peaks) == 0:
            continue
            
        exp_mzs = valid_peaks[:, 0] * 1000.0 
        exp_ints = valid_peaks[:, 1].clone()
        
        diffs = torch.abs(frag_mzs.unsqueeze(1) - exp_mzs.unsqueeze(0))
        hits = (diffs <= tolerance)
        
        aligned_preds = []
        aligned_targets = []
        
        for exp_idx in range(len(exp_mzs)):
            matching_frag_indices = torch.where(hits[:, exp_idx])[0]
            if len(matching_frag_indices) > 0:
                claimed_int = torch.sum(pred_ints[matching_frag_indices])
                aligned_preds.append(claimed_int)
            else:
                aligned_preds.append(torch.tensor(1e-8, device=device)) 
                
            aligned_targets.append(exp_ints[exp_idx])
            
        unmatched_frags = torch.where(~hits.any(dim=1))[0]
        for frag_idx in unmatched_frags:
            aligned_preds.append(pred_ints[frag_idx])
            aligned_targets.append(torch.tensor(1e-8, device=device)) 
            
        A = torch.stack(aligned_preds)
        B_target = torch.stack(aligned_targets)
        
        A = A / (torch.sum(A) + 1e-8)
        B_target = B_target / (torch.sum(B_target) + 1e-8)
        
        mixed_dist = A + B_target
        eps = 1e-12
        term_A = A * torch.log((2 * A + eps) / (mixed_dist + eps))
        term_B = B_target * torch.log((2 * B_target + eps) / (mixed_dist + eps))
        
        entropy = (term_A + term_B).sum() / math.log(4)
        
        batch_loss += entropy
        valid_batches += 1
        
    return batch_loss / max(1, valid_batches)

# ==============================================================================
# MASTER TRAINING EXECUTION
# ==============================================================================
def train_phase3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    wandb.init(project="desaf-phase3-autoregressive", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[*] Loading Model Configurations...")
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print("\n[*] Initializing Phase 3 Dataset...")
    full_dataset = Phase2EdgeDataset(processed_graphs_path=args.data_path)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=phase2_collate_fn, num_workers=6, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=phase2_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing DESAF-Net (Phase 3 Actor-Critic Mode)...")
    model = DESAFNet(model_config=full_config['model']).to(device)
    emb_dim = full_config['model'].get('embed_dim', 768) if full_config['model'].get('embed_dim') != -1 else 768
    
    if not os.path.exists(args.phase2_checkpoint):
        raise FileNotFoundError(f"[-] Phase 2 Checkpoint not found at {args.phase2_checkpoint}.")
    
    load_phase2_weights_with_surgery(model, args.phase2_checkpoint, device, emb_dim)
    
    # ==========================================
    # PHASE 3 INITIAL WARM-UP FREEZE
    # ==========================================
    for param in model.graph_encoder.parameters(): param.requires_grad = False
    for param in model.edge_head.parameters(): param.requires_grad = False
    for param in model.h_routing_head.parameters(): param.requires_grad = False
    
    for param in model.set_transformer.parameters(): param.requires_grad = True

    optimizer = AdamW(model.set_transformer.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.thaw_epoch, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize Actor-Critic Baseline
    reward_baseline = 0.0
    lambda_policy = args.lambda_policy

    print(f"\n[*] Commencing Phase 3 Autoregressive Warm-up Loop...")

    for epoch in range(1, args.epochs + 1):
        
        # ==========================================
        # THE DIFFERENTIAL THAW TRIGGER
        # ==========================================
        if epoch == args.thaw_epoch:
            print("\n[!] ========================================================")
            print(f"[!] EPOCH {epoch}: WARM-UP COMPLETE. THAWING BACKBONE WITH DIFFERENTIAL LR!")
            print("[!] ========================================================\n")
            for param in model.parameters(): param.requires_grad = True
            
            optimizer = AdamW([
                {'params': model.graph_encoder.parameters(), 'lr': 1e-5}, 
                {'params': model.edge_head.parameters(), 'lr': 1e-5},      
                {'params': model.h_routing_head.parameters(), 'lr': 1e-5}, 
                {'params': model.set_transformer.parameters(), 'lr': args.learning_rate} 
            ], weight_decay=1e-4)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.thaw_epoch + 1), eta_min=1e-6)
        # ==========================================

        model.train()
        total_train_entropy = 0.0
        total_train_rl_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(train_bar):
            if batch is None: continue
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            
            theoretical_mzs, predicted_ints, fragment_batch_idx, cascade_log_probs = model.forward_phase3(batch, max_steps=args.max_steps)
            
            # ==========================================
            # SET TRANSFORMER PANIC TELEMETRY
            # ==========================================
            if torch.isnan(predicted_ints).any():
                print(f"\n[CRITICAL ERROR] NaN detected in Set Transformer outputs at Epoch {epoch}, Batch {batch_idx}!")
                sys.exit(1)
                
            if batch_idx % 100 == 0:
                mol_0_ints = predicted_ints[fragment_batch_idx == 0]
                if len(mol_0_ints) > 0:
                    max_int = mol_0_ints.max().item()
                    min_int = mol_0_ints.min().item()
                    if max_int < 0.2:
                        print(f"\n[WARNING] Mol 0 Predicted Max Int: {max_int*100:.1f}%. Network is outputting uniform flatlines.")
            # ==========================================

            # 1. The Continuous "Critic" Loss (Set Transformer & Backbone)
            entropy_loss = spectral_entropy_loss(
                theoretical_mzs, predicted_ints, fragment_batch_idx, batch['peaks'], batch['peak_mask']
            )
            
            # 2. The Discrete "Actor" Loss (Phase 2 Heads via REINFORCE)
            # [FIXED] Use the dense topological reward instead of the Critic's entropy!
            with torch.no_grad():
                topological_rewards = calculate_cosine_reward(
                    theoretical_mzs, fragment_batch_idx, batch['peaks'], batch['peak_mask']
                )
            
            # Update the moving average baseline (b)
            reward_baseline = 0.9 * reward_baseline + 0.1 * topological_rewards.mean().item()
            
            # Calculate Advantage (R - b)
            advantage = topological_rewards - reward_baseline
            
            # Policy Loss: - (Advantage) * Sum(LogProbs)
            rl_loss = -(advantage * cascade_log_probs).mean()
            
            # 3. The Joint Objective Function
            total_loss = entropy_loss + (lambda_policy * rl_loss)
            
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_train_entropy += entropy_loss.item()
            total_train_rl_loss += rl_loss.item()
            
            train_bar.set_postfix({
                'Entropy': f"{entropy_loss.item():.4f}", 
                'RL Loss': f"{rl_loss.item():.4f}"
            })

            if batch_idx % 100 == 0 and wandb.run is not None:
                wandb.log({
                    "batch/spectral_entropy_loss": entropy_loss.item(),
                    "batch/policy_loss": rl_loss.item(),
                    "batch/advantage": advantage.mean().item(),
                    "batch/avg_fragments_generated": len(theoretical_mzs) / max(1, batch['x'].shape[0])
                }, commit=False)

        avg_train_entropy = total_train_entropy / max(len(train_loader), 1)

        # ==========================================
        # VALIDATION PASS & TELEMETRY
        # ==========================================
        model.eval()
        total_val_entropy = 0.0
        total_val_fragments = 0
        total_val_molecules = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

                theoretical_mzs, predicted_ints, fragment_batch_idx, _ = model.forward_phase3(batch, max_steps=args.max_steps)
                
                v_loss = spectral_entropy_loss(
                    theoretical_mzs, predicted_ints, fragment_batch_idx, batch['peaks'], batch['peak_mask']
                )
                total_val_entropy += v_loss.item()
                
                total_val_fragments += len(theoretical_mzs)
                total_val_molecules += batch['x'].shape[0]

        avg_val_entropy = total_val_entropy / max(len(val_loader), 1)
        avg_val_frags = total_val_fragments / max(total_val_molecules, 1)
        
        current_transformer_lr = optimizer.param_groups[-1]['lr']
        current_backbone_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 1 else 0.0
        
        wandb.log({
            "epoch": epoch, 
            "train/spectral_entropy_loss": avg_train_entropy, 
            "val/spectral_entropy_loss": avg_val_entropy,
            "val/avg_frags_per_mol": avg_val_frags,
            "lr/set_transformer": current_transformer_lr,
            "lr/backbone": current_backbone_lr
        })
        
        print(f"Epoch {epoch} | Train Entropy: {avg_train_entropy:.4f} | Val Entropy: {avg_val_entropy:.4f} | Avg Frags/Mol: {avg_val_frags:.2f}")

        if avg_val_frags <= 1.05:
            print("[!] WARNING: Cascade Collapse Detected. The model has stopped cutting bonds.")

        scheduler.step()

        if avg_val_entropy < best_val_loss:
            best_val_loss = avg_val_entropy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "desaf_phase3_best.pt"))
            print(f" -> Best Phase 3 model saved! (Val Entropy: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}.")
                break

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to graphs containing spectral peaks")
    parser.add_argument("--phase2_checkpoint", type=str, required=True, help="Path to desaf_phase2_best.pt (MANDATORY)")
    parser.add_argument("--output_dir", type=str, default="trained_model/desaf_phase3")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--epochs", type=int, default=100) 
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    parser.add_argument("--max_steps", type=int, default=3, help="Max iterations for the autoregressive cascade")
    
    parser.add_argument("--thaw_epoch", type=int, default=6, help="Epoch at which the MassFormer backbone is unfrozen.")
    parser.add_argument("--lambda_policy", type=float, default=1.0, help="Scaling factor for the REINFORCE loss.")
    args = parser.parse_args()
    
    train_phase3(args)