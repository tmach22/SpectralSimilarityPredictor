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
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
# Adjust these paths to where you saved the Phase 1 dataloader and master model script
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'fiar')) 
sys.path.insert(0, os.path.join(cwd, 'model', 'fiar'))

try:
    from data_loader import Phase1EdgeDataset, phase1_collate_fn
    from phase1_model import DESAFNet # Your new master architecture script
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

def train_phase1_anchor(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Initialize wandb securely
    wandb.init(project="desaf-phase1-anchor", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[*] Loading Model Configurations...")
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print("\n[*] Initializing Phase 1 Dataset...")
    full_dataset = Phase1EdgeDataset(processed_graphs_path=args.phase1_data_path)

    # 80/20 Train-Validation Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=phase1_collate_fn, num_workers=6, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=phase1_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing DESAF-Net (Phase 1 Mode)...")
    model = DESAFNet(model_config=full_config['model']).to(device)
    
    # CRITICAL: Freeze the MassFormer entirely.
    model.freeze_backbone()

    # Optimizer ONLY targets the Edge Head
    optimizer = AdamW(
        model.edge_head.parameters(), 
        lr=args.learning_rate, 
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # CRITICAL: Address the 6.49% class imbalance
    pos_weight = torch.tensor([14.4]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n[*] Commencing Phase 1 Anchor Training Loop...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_true_positives, train_actual_positives = 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in train_bar:
            if batch is None: continue

            # Move dictionary items to GPU
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            
            # Forward pass: Uses Phase 1 dense-to-sparse bridge
            logits = model.forward_phase1(batch)
            
            # Loss calculation against MAGMa labels
            loss = criterion(logits, batch['y_cleavage'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.edge_head.parameters(), max_norm=1.0)
            optimizer.step()

            # Tracking Metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            train_true_positives += ((preds == 1.0) & (batch['y_cleavage'] == 1.0)).sum().item()
            train_actual_positives += (batch['y_cleavage'] == 1.0).sum().item()

            total_train_loss += loss.item()
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_recall = (train_true_positives / train_actual_positives) if train_actual_positives > 0 else 0.0

        # --- Validation Pass ---
        model.eval()
        total_val_loss = 0.0
        val_true_positives, val_actual_positives = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

                logits = model.forward_phase1(batch)
                v_loss = criterion(logits, batch['y_cleavage'])
                total_val_loss += v_loss.item()
                
                # Tracking Metrics
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_true_positives += ((preds == 1.0) & (batch['y_cleavage'] == 1.0)).sum().item()
                val_actual_positives += (batch['y_cleavage'] == 1.0).sum().item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_recall = (val_true_positives / val_actual_positives) if val_actual_positives > 0 else 0.0
        
        # Log to Weights & Biases
        wandb.log({
            "epoch": epoch, 
            "train/loss": avg_train_loss, 
            "val/loss": avg_val_loss,
            "train/recall": train_recall,
            "val/recall": val_recall
        })
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Recall: {val_recall*100:.2f}%")

        scheduler.step()

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "desaf_phase1_best.pt"))
            print(f" -> Best Phase 1 model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}.")
                break

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Updated Argument: We only need the generated .pt graphs for Phase 1
    parser.add_argument("--phase1_data_path", type=str, required=True, help="Path to phase1_graphs.pt")
    parser.add_argument("--output_dir", type=str, default="trained_model/desaf_phase1")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3) # Start slightly higher since it's just an MLP
    args = parser.parse_args()
    
    train_phase1_anchor(args)