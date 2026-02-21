import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

# --- 1. SETUP PATHS ---
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    # Ensure we use the NEW model file with Mass Gating
    from classifier_siamesemodel_new import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 2. AUGMENTATION FUNCTION ---
def apply_spec_augment(batch_dict, mask_prob=0.15, device='cuda'):
    """
    Applies Peak Masking (SpecAugment) to Graph/Feature dictionaries.
    """
    if not isinstance(batch_dict, dict): return batch_dict
    
    target_keys = ['x', 'edge_attr', 'intensities', 'peaks', 'spectrum']
    
    augmented_batch = {}
    
    for key, val in batch_dict.items():
        if key in target_keys and isinstance(val, torch.Tensor) and val.is_floating_point():
            mask = torch.bernoulli(torch.full_like(val, 1 - mask_prob)).to(device)
            augmented_batch[key] = val * mask
        else:
            augmented_batch[key] = val
            
    return augmented_batch

def train_specaugment_only(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 5b: Full SpecAugment Fine-Tuning (Mass Gated) ---")
    print(f"Strategy: Train on FULL dataset with Peak Masking (p={args.mask_prob})")
    
    # 3. Config & Model Init
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    print("Loading FULL Training Data...")
    train_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # Dynamically calculate meta dim
    spec_meta_dim = train_dataset[0][2].shape[1]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    print("Initializing Model...")
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)
    
    # 4. Load Weights (Start from Champion)
    print(f"Loading Weights from: {args.start_weights}")
    state_dict = torch.load(args.start_weights, map_location=device)
    
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    elif 'best_model_sd' in state_dict: state_dict = state_dict['best_model_sd']
        
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded weights.")
    except RuntimeError as e:
        print(f"WARNING: Slight mismatch in loading weights: {e}")

    # 5. Unfreezing Strategy
    print("\n--- Unfreezing Head, Gating Branch + Last 2 Graph Layers ---")
    for param in model.parameters(): param.requires_grad = False
    
    # Unfreeze Similarity Head
    for param in model.head.parameters(): param.requires_grad = True
    
    # Unfreeze Mass Penalty Head (Important!)
    for param in model.mass_gate.parameters(): param.requires_grad = True
    
    # Unfreeze Encoder Layers
    try:
        # Access the underlying graph layers
        # Path might vary based on your exact hierarchy, using generic traversal or try/except
        encoder_layers = model.encoder.encoder.encoder.graph_encoder.layers
        total_layers = len(encoder_layers)
        print(f"Unfreezing last 2 layers of {total_layers}...")
        for i in range(total_layers - 2, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
    except AttributeError:
        print("Fallback: Unfreezing full encoder.")
        for param in model.encoder.parameters(): param.requires_grad = True

    # 6. Optimizer & Loss
    # Low LR for fine-tuning
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5) 
    # Use Weighted Loss if class imbalance persists, otherwise standard BCE
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 7. Training Loop
    best_val_f1 = 0.0
    early_stop_counter = 0
    
    print(f"\nTraining for {args.epochs} epochs with SpecAugment...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # UPDATED: Unpack 5 items (including mass_diffs)
        for batch_A, batch_B, batch_meta, batch_mass_diffs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            batch_mass_diffs = batch_mass_diffs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            
            # --- APPLY AUGMENTATION ---
            aug_A = apply_spec_augment(batch_A, mask_prob=args.mask_prob, device=device)
            aug_B = apply_spec_augment(batch_B, mask_prob=args.mask_prob, device=device)
            
            # Forward (Pass Mass Diff)
            logits = model(aug_A, aug_B, batch_meta, batch_mass_diffs).view(-1)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation (NO Augmentation - Clean Test)
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, batch_mass_diffs, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                batch_mass_diffs = batch_mass_diffs.to(device)
                labels = labels.to(device).float()

                # Forward (Pass Mass Diff)
                logits = model(batch_A, batch_B, batch_meta, batch_mass_diffs).view(-1)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "best_gated_specaug_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best F1! Saved to {save_path}")
        else:
            early_stop_counter += 1
            print(f"No improvement ({early_stop_counter}/{args.patience})")
            if early_stop_counter >= args.patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    # Model Weights (Start from Best Gated Model)
    parser.add_argument("--start_weights", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, default="./specaugment_gated")
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    
    args = parser.parse_args()
    train_specaugment_only(args)