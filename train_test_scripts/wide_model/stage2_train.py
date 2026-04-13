import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path

# --- SETUP ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'wide_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'wide_model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Import the updated 6-item dataloader and the geometry-fixed model
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
from model_stage2 import StageTwoClassificationModel

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
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

def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # --- HARDWARE OPTIMIZATIONS ---
    num_workers = full_config.get('run', {}).get('num_workers', 6)
    use_pin_memory = full_config.get('run', {}).get('pin_memory', True)
    print(f"DataLoader Config: num_workers={num_workers}, pin_memory={use_pin_memory}")

    # 1. Init Data (Strictly NO upsampling to avoid class imbalance)
    print("Loading Training Dataset...")
    train_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=False # <- CURES THE 100% RECALL MODE COLLAPSE
    )
    
    print("Loading Validation Dataset...")
    val_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=False 
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=binary_collate_fn, pin_memory=use_pin_memory, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, pin_memory=use_pin_memory, num_workers=num_workers
    )

    # 2. Init Model
    model = StageTwoClassificationModel(
        model_config=full_config['model'], 
        stage1_checkpoint=args.stage1_checkpoint
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    # 3. Init Optimizer (ONLY THE HEAD!)
    optimizer = AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # --- EARLY STOPPING TRACKERS ---
    best_val_loss = float('inf')
    patience_counter = 0

    # 4. Training Loop
    for epoch in range(args.epochs):
        # --- TRAIN PHASE ---
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        # Unpack 6 items now
        for batch_A, batch_B, spec_meta, mass_A, mass_B, labels in train_bar:
            batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
            
            spec_meta = spec_meta.to(device, non_blocking=True)
            mass_A = mass_A.to(device, non_blocking=True)
            mass_B = mass_B.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1) 
            
            optimizer.zero_grad()
            
            # Pass individual masses to let the model sort h_heavy and h_light
            logits = model(batch_A, batch_B, spec_meta, mass_A, mass_B)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_A, batch_B, spec_meta, mass_A, mass_B, labels in val_loader:
                batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
                batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
                
                spec_meta = spec_meta.to(device, non_blocking=True)
                mass_A = mass_A.to(device, non_blocking=True)
                mass_B = mass_B.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1) 
                
                logits = model(batch_A, batch_B, spec_meta, mass_A, mass_B)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            save_path = os.path.join(args.output_dir, "massformer_stage2_best_head.pt")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ New best validation loss! Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f" -> ⚠️ Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}. No improvement for {args.patience} epochs.")
                break

    print("\nStage 2 Training Complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="The .pt file generated by Stage 1")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage2")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait before early stopping")
    
    args = parser.parse_args()
    train_stage2(args)