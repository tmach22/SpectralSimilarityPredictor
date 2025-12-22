import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

# --- SETUP PATHS ---
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from classifier_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError:
    sys.exit(1)

def apply_spec_augment(batch_dict, mask_prob=0.15, device='cuda'):
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

def train_hard_augment_balanced(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Phase 6: Balanced Hard Fine-Tuning + SpecAugment ---")
    
    # Config & Model
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    full_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    spec_meta_dim = full_dataset[0][2].shape[1]
    
    model = SiameseSpectralSimilarityModel(full_config.get('model', {}), args.checkpoint_path, spec_meta_dim).to(device)
    
    # Load Weights (From Baseline)
    state_dict = torch.load(args.start_weights, map_location=device)
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    elif 'best_model_sd' in state_dict: state_dict = state_dict['best_model_sd']
    model.load_state_dict(state_dict, strict=False)

    # --- BALANCING LOGIC ---
    print("Filtering and Balancing Hard Examples...")
    df = full_dataset.pairs_df
    
    # Criteria
    hard_pos_mask = (df['tanimoto'] < 0.5) & ((df['label'] == 1) | (df['cosine_similarity'] >= 0.7))
    hard_neg_mask = (df['tanimoto'] > 0.6) & ((df['label'] == 0) | (df['cosine_similarity'] < 0.7))
    
    pos_indices = df.index[hard_pos_mask].tolist()
    neg_indices = df.index[hard_neg_mask].tolist()
    
    print(f"  Hard Positives found: {len(pos_indices)}")
    print(f"  Hard Negatives found: {len(neg_indices)}")
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print("Error: One class is empty. Cannot balance.")
        return

    # Downsample majority to match minority
    min_len = min(len(pos_indices), len(neg_indices))
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)
    
    balanced_indices = pos_indices[:min_len] + neg_indices[:min_len]
    print(f"  Balanced Dataset Size: {len(balanced_indices)} ({min_len} Pos + {min_len} Neg)")
    
    hard_dataset = Subset(full_dataset, balanced_indices)
    train_loader = DataLoader(hard_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    # Unfreeze Head + Last 2 Layers
    for param in model.parameters(): param.requires_grad = False
    for param in model.similarity_head.parameters(): param.requires_grad = True
    try:
        layers = model.encoder.encoder.encoder.graph_encoder.layers
        for i in range(len(layers)-2, len(layers)):
            for p in layers[i].parameters(): p.requires_grad = True
    except:
        for p in model.encoder.parameters(): p.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6) 
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    patience = 3
    early_stop = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move to device
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            aug_A = apply_spec_augment(batch_A, args.mask_prob, device)
            aug_B = apply_spec_augment(batch_B, args.mask_prob, device)
            
            logits = model(aug_A, aug_B, batch_meta).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                labels = labels.to(device).float()
                logits = model(batch_A, batch_B, batch_meta).view(-1)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds)
        print(f"Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_balanced_hard_augment.pth"))
            print("Saved Best.")
        else:
            early_stop += 1
            if early_stop >= patience: break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add your standard args here (paths, etc.)...
    # For brevity, I am assuming you know to add the arguments from previous scripts
    # Copy the argparse block from train_hard_augment.py
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--start_weights", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./balanced_hard_augment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    
    args = parser.parse_args()
    train_hard_augment_balanced(args)