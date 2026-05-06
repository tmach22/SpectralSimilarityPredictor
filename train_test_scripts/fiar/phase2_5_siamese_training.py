import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import wandb
import numpy as np
from pathlib import Path
import warnings

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
    from phase2_5_dataloader import PairedSiameseDataset, siamese_collate_fn
    # [UPDATED] Import Partial Thaw Architecture
    from phase2_5_siamese_model import Phase3_Partial_SiameseNetwork 
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

def train_phase3(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        print(f"[*] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")

    wandb.init(project="desaf-phase3-partial-thaw", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print("\n[*] Initializing Siamese Datasets (Target: Spectral Entropy)...")
    primary_dataset = PairedSiameseDataset(feather_path=args.pairs_path, graphs_path=args.graphs_path)

    if args.val_pairs_path:
        train_dataset = primary_dataset
        val_dataset = PairedSiameseDataset(feather_path=args.val_pairs_path, graphs_path=args.graphs_path)
    else:
        train_size = int(0.8 * len(primary_dataset))
        val_size = len(primary_dataset) - train_size
        train_dataset, val_dataset = random_split(primary_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=siamese_collate_fn, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=siamese_collate_fn, num_workers=6, pin_memory=True)

    print("\n[*] Building Partial Thaw Network & Loading Previous Weights...")
    model = Phase3_Partial_SiameseNetwork(full_config, args.phase2_checkpoint, device, max_fragments=10).to(device)
    
    if os.path.exists(args.phase2_5_ckpt):
        print(f"[*] Loading pretrained Phase 2.5 Bottleneck from: {args.phase2_5_ckpt}")
        model.load_state_dict(torch.load(args.phase2_5_ckpt, map_location=device), strict=False)
    
    criterion = nn.HuberLoss(delta=0.5) 
    
    # =================================================================
    # [THE SURGICAL FIX] Parameter Grouping for Differential LR
    # =================================================================
    # Only grab extractor parameters that we explicitly set to requires_grad=True
    edge_head_params = [p for n, p in model.extractor.named_parameters() if p.requires_grad]
    
    bottleneck_params = (
        list(model.context_attn.parameters()) + 
        list(model.abundance_head.parameters()) +
        list(model.anisotropy_norm.parameters()) +
        list(model.final_affine.parameters())
    )
    
    optimizer = AdamW([
        {'params': edge_head_params, 'lr': args.lr_edge_head},    # 5e-6
        {'params': bottleneck_params, 'lr': args.lr_bottleneck}   # 1e-5
    ], weight_decay=1e-3)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n[*] Commencing Phase 3 Partial Thaw...")
    for epoch in range(1, args.epochs + 1):
        
        model.train()
        total_train_loss = 0.0
        current_tau = 0.01 
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train] (tau={current_tau})")
        
        for batch_idx, (batch_A, batch_B, targets) in enumerate(train_bar):
            if batch_A is None or batch_B is None: continue
            
            batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            predicted_sims = model(batch_A, batch_B, tau=current_tau)
            loss = criterion(predicted_sims, targets)
            
            if torch.isnan(loss):
                print("\n[!] FATAL: NaN loss detected. Halting to protect checkpoint.")
                wandb.finish()
                sys.exit(1)
                
            loss.backward()
            
            all_trainable = edge_head_params + bottleneck_params
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in all_trainable if p.grad is not None]), 2.0)
            
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix({'Huber Loss': f"{loss.item():.4f}", 'GradNorm': f"{total_norm.item():.2f}"})

            if batch_idx % 100 == 0 and wandb.run is not None:
                wandb.log({
                    "batch/train_loss": loss.item(), 
                    "batch/grad_norm": total_norm.item()
                }, commit=False)

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        total_val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        
        with torch.no_grad():
            for batch_A, batch_B, targets in val_loader:
                if batch_A is None or batch_B is None: continue
                
                batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
                batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}
                targets = targets.to(device, non_blocking=True)

                predicted_sims = model(batch_A, batch_B, tau=current_tau)
                v_loss = criterion(predicted_sims, targets)
                total_val_loss += v_loss.item()
                
                all_val_preds.extend(predicted_sims.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        
        val_preds_arr = np.array(all_val_preds)
        val_targets_arr = np.array(all_val_targets)
        p_min, p_max = val_preds_arr.min(), val_preds_arr.max()
        p_mean, p_std = val_preds_arr.mean(), val_preds_arr.std()
        t_min, t_max = val_targets_arr.min(), val_targets_arr.max()
        
        print("\n" + "-"*50)
        print(f" 📊 EPOCH {epoch} DIAGNOSTICS (PARTIAL THAW) 📊")
        print(f" Targets -> Range: [{t_min:.2f}, {t_max:.2f}] | Mean: {val_targets_arr.mean():.4f}")
        print(f" Preds   -> Range: [{p_min:.2f}, {p_max:.2f}] | Mean: {p_mean:.4f} | Std: {p_std:.4f}")
        print("-" * 50)

        wandb.log({
            "epoch": epoch, 
            "train/loss": avg_train_loss, 
            "val/loss": avg_val_loss,
            "val/pred_mean": p_mean,
            "val/pred_max": p_max, 
            "val/pred_min": p_min,
            "lr_edge_head": optimizer.param_groups[0]['lr'],
            "lr_bottleneck": optimizer.param_groups[1]['lr']
        })
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "desaf_phase3_partial_thaw_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}.")
                break

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, default=None)
    parser.add_argument("--graphs_path", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True, help="Path to deterministic_student_best.pt")
    parser.add_argument("--phase2_5_ckpt", type=str, required=True, help="Path to desaf_phase2_5_entropy_best.pt")
    parser.add_argument("--output_dir", type=str, default="trained_model/desaf_phase3_partial")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--gpu_id", type=int, default=0)
    # You can likely safely raise the batch size again since the MassFormer is frozen
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--epochs", type=int, default=30) 
    parser.add_argument("--patience", type=int, default=5)
    
    # Differential Learning Rates (Expert Recommendation)
    parser.add_argument("--lr_edge_head", type=float, default=5e-6) 
    parser.add_argument("--lr_bottleneck", type=float, default=1e-5) 
    
    args = parser.parse_args()
    train_phase3(args)