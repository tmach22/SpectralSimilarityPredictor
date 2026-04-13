import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
    from siamese_sinkhorn import SiameseSinkhornPredictor
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

def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # 1. Initialize Binary Datasets (Train & Val)
    print("\n[*] Initializing Stage 2 Binary Datasets...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=4)

    # 2. Initialize the Siamese Sinkhorn Model
    model = SiameseSinkhornPredictor(
        model_config=full_config['model'], 
        stage1_ckpt_path=args.stage1_ckpt,
        meta_dim=args.meta_dim
    ).to(device)
    
    # 3. Setup Optimizer (Differential Learning Rates)
    # We use a standard LR for the new Sinkhorn classifier, but a tiny LR to gently fine-tune the FLARE motifs
    classifier_params = list(model.classifier.parameters())
    motif_params = (
        list(model.motif_extractor.kinetic_mlp.parameters()) +
        list(model.motif_extractor.gcn1.parameters()) +
        list(model.motif_extractor.gcn2.parameters()) +
        list(model.motif_extractor.assignment_head.parameters())
    )
    
    optimizer = AdamW([
        {'params': classifier_params, 'lr': args.learning_rate},
        {'params': motif_params, 'lr': args.learning_rate * 0.1} # Fine-tuning LR
    ], weight_decay=1e-4)
    
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # --- TRAINING LOOP ---
    print("\n[*] Commencing Stage 2: Sinkhorn Similarity Training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for b_A, b_B, b_meta, b_massA, b_massB, labels, brics_A, brics_B in train_bar:
            
            # Safe device placement
            b_A = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b_A.items()} if isinstance(b_A, dict) else b_A.to(device)
            b_B = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b_B.items()} if isinstance(b_B, dict) else b_B.to(device)
            
            b_meta, b_massA, b_massB, labels = b_meta.to(device), b_massA.to(device), b_massB.to(device), labels.to(device)
            brics_A = brics_A.to(device) if brics_A is not None else None
            brics_B = brics_B.to(device) if brics_B is not None else None
            
            optimizer.zero_grad()
            
            # Forward Pass
            logits = model(b_A, b_B, brics_A, brics_B, b_massA, b_massB, b_meta)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'BCE_Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        with torch.no_grad():
            for b_A, b_B, b_meta, b_massA, b_massB, labels, brics_A, brics_B in val_loader:
                b_A = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b_A.items()} if isinstance(b_A, dict) else b_A.to(device)
                b_B = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b_B.items()} if isinstance(b_B, dict) else b_B.to(device)
                
                b_meta, b_massA, b_massB, labels = b_meta.to(device), b_massA.to(device), b_massB.to(device), labels.to(device)
                brics_A = brics_A.to(device) if brics_A is not None else None
                brics_B = brics_B.to(device) if brics_B is not None else None
                
                logits = model(b_A, b_B, brics_A, brics_B, b_massA, b_massB, b_meta)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                # Calculate Accuracy
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_samples
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "siamese_sinkhorn_best.pt"))
            print(" -> ⭐ Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Ensure you point to your specific Stage 1 checkpoint here!
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # This must match the output dimension of your FixedMetadataEncoder (usually around ~78)
    parser.add_argument("--meta_dim", type=int, default=81)
    
    args = parser.parse_args()
    train_stage2(args)