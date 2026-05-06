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
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
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
    from phase3_multitask_dataloader import PairedSiameseDataset, siamese_collate_fn
    from phase3_multitask_siamese import Phase3_LinearProbe_SiameseNetwork 
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    sys.exit(1)

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

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

def train_linear_probe(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    wandb.init(project="desaf-phase3-linear-probe", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # --- VERIFY TARGET DATASET ---
    print("\n[*] Verifying Analog Discovery Target Dataset...")
    df_check = pd.read_feather(args.pairs_path) if str(args.pairs_path).endswith('.feather') else pd.read_csv(args.pairs_path)
    if 'label' not in df_check.columns:
        print("[!] FATAL: The 'label' column (Modified Cosine) was not found in your dataset.")
        sys.exit(1)
    
    num_pos = len(df_check[df_check['label'] == 1.0])
    num_neg = len(df_check[df_check['label'] == 0.0])
    print(f"    -> Found {num_pos} Positive Analogs and {num_neg} Negatives.")

    print("\n[*] Initializing Datasets...")
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

    print("\n[*] Building Linear Probe Network...")
    model = Phase3_LinearProbe_SiameseNetwork(full_config, args.phase2_checkpoint, device, max_fragments=10).to(device)
    
    if os.path.exists(args.phase3_ckpt):
        print(f"[*] Loading pretrained Phase 3 Partial Thaw weights: {args.phase3_ckpt}")
        model.load_state_dict(torch.load(args.phase3_ckpt, map_location=device), strict=False)
    
    # --- EXPLICIT MANIFOLD FREEZE ---
    print("[*] Executing Topological Lockdown (Freezing Backbone)...")
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Thaw ONLY the binary head
    trainable_params = []
    if hasattr(model, 'binary_head'):
        for param in model.binary_head.parameters():
            param.requires_grad = True
            trainable_params.append(param)
    else:
        print("[!] FATAL: Could not locate 'binary_head'.")
        sys.exit(1)

    # Loss criteria
    criterion_huber = nn.HuberLoss(delta=0.5) 
    criterion_focal = BinaryFocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    # Pass only the explicitly thawed parameters to the optimizer
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_val_auc = 0.0

    print(f"\n[*] Commencing Strict Linear Probe")
    print(f"    -> Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"    -> Target: Modified Cosine Similarity (Analog Discovery)")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_focal_loss, total_huber_loss = 0.0, 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for batch_idx, (batch_A, batch_B, targets_sim, targets_label) in enumerate(train_bar):
            if batch_A is None or batch_B is None: continue
            
            batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}
            
            targets_sim = targets_sim.to(device, non_blocking=True)
            targets_label = targets_label.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            
            continuous_pred, binary_logit = model(batch_A, batch_B, tau=0.01)
            
            # --- THE FIX: Force both outputs and targets to flat 1D tensors ---
            binary_logit = binary_logit.view(-1)
            targets_label = targets_label.view(-1)
            
            with torch.no_grad():
                loss_huber = criterion_huber(continuous_pred.view(-1), targets_sim.view(-1))
                
            loss_focal = criterion_focal(binary_logit, targets_label)
            
            if torch.isnan(loss_focal):
                print("\n[!] FATAL: NaN focal loss detected. Halting.")
                sys.exit(1)
                
            loss_focal.backward()
            optimizer.step()

            total_focal_loss += loss_focal.item()
            total_huber_loss += loss_huber.item()
            train_bar.set_postfix({'Focal': f"{loss_focal.item():.4f}", 'Huber (Frozen)': f"{loss_huber.item():.4f}"})

        model.eval()
        total_val_focal, total_val_huber = 0.0, 0.0
        all_val_logits, all_val_targets = [], []
        
        with torch.no_grad():
            for batch_A, batch_B, targets_sim, targets_label in val_loader:
                if batch_A is None or batch_B is None: continue
                
                batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
                batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}
                targets_sim = targets_sim.to(device, non_blocking=True)
                targets_label = targets_label.to(device, non_blocking=True).float()

                continuous_pred, binary_logit = model(batch_A, batch_B, tau=0.01)
                
                # --- THE FIX: Force both outputs and targets to flat 1D tensors ---
                binary_logit = binary_logit.view(-1)
                targets_label = targets_label.view(-1)
                
                loss_huber = criterion_huber(continuous_pred.view(-1), targets_sim.view(-1))
                loss_focal = criterion_focal(binary_logit, targets_label)
                
                total_val_huber += loss_huber.item()
                total_val_focal += loss_focal.item()
                
                all_val_logits.extend(binary_logit.cpu().numpy())
                all_val_targets.extend(targets_label.cpu().numpy())

        avg_val_focal = total_val_focal / max(len(val_loader), 1)
        avg_val_huber = total_val_huber / max(len(val_loader), 1)
        
        try:
            val_auc = roc_auc_score(all_val_targets, all_val_logits)
        except ValueError:
            val_auc = 0.0
        
        print("\n" + "-"*50)
        print(f" 📊 EPOCH {epoch} LINEAR PROBE DIAGNOSTICS 📊")
        print(f" Val Focal Loss : {avg_val_focal:.4f} (Decision Boundary)")
        print(f" Val Huber Loss : {avg_val_huber:.4f} (Locked Anchor)")
        print(f" ROC-AUC Score  : {val_auc:.4f}")
        print("-" * 50)

        wandb.log({
            "epoch": epoch, 
            "val/loss_focal": avg_val_focal,
            "val/loss_huber": avg_val_huber,
            "val/roc_auc": val_auc,
            "binary_head/weight": model.binary_head.weight.item() if hasattr(model.binary_head, 'weight') else 0,
            "binary_head/bias": model.binary_head.bias.item() if hasattr(model.binary_head, 'bias') else 0
        })
        
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"[+] New best AUC ({best_val_auc:.4f})! Saving weights.")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "desaf_phase3_linear_probe_best.pt"))

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, default=None)
    parser.add_argument("--graphs_path", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True)
    parser.add_argument("--phase3_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trained_model/desaf_phase3_linear_probe")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--epochs", type=int, default=10) 
    
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--focal_gamma", type=float, default=3.0)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    
    args = parser.parse_args()
    train_linear_probe(args)