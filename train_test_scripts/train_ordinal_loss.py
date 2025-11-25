import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Setup paths
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from multiclass_classifier_siamesemodel import SiameseSpectralSimilarityModel
from updated_train import merge_configs
from multiclass_data_loader import MulticlassDataset, multiclass_collate_fn

# --- HELPER: Convert Labels to Ordinal Targets ---
def make_ordinal_targets(labels, num_tasks, device):
    """
    Converts integer labels into ordinal binary targets.
    Example for 3 classes (2 tasks):
      Label 0 -> [0, 0]
      Label 1 -> [1, 0]
      Label 2 -> [1, 1]
    """
    # Create range [0, 1] for tasks
    tasks = torch.arange(num_tasks, device=device).unsqueeze(0)
    labels = labels.unsqueeze(1)
    
    # Broadcast comparison: Is Label > Task Index?
    # Task 0 (Thresh 1): Label 1 > 0 (True)
    # Task 1 (Thresh 2): Label 1 > 1 (False)
    return (labels > tasks).float()

def train_ordinal(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Ordinal Multiclass Training (3 Classes -> 2 Binary Tasks) ---")
    
    # 1. Config & Dataset
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    print("Initializing Dataset...")
    train_dataset = MulticlassDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = MulticlassDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # --- 2. Calculate Weights for BCE ---
    print("Calculating Ordinal Task Weights...")
    # Extract labels from dataframe for fast counting
    # Logic matches your dataloader: <0.65=0, <0.85=1, >=0.85=2
    y_sims = train_dataset.pairs_df['cosine_similarity'].values
    y_labels = np.zeros_like(y_sims, dtype=int)
    y_labels[(y_sims >= 0.65) & (y_sims < 0.85)] = 1
    y_labels[y_sims >= 0.85] = 2
    
    n_0 = (y_labels == 0).sum()
    n_1 = (y_labels == 1).sum()
    n_2 = (y_labels == 2).sum()
    
    print(f"Class Counts: Low={n_0}, Med={n_1}, High={n_2}")
    
    # Task 0: Is it > Low? (Target 1 for Med & High)
    t0_pos = n_1 + n_2
    t0_neg = n_0
    w0 = t0_neg / t0_pos if t0_pos > 0 else 1.0
    
    # Task 1: Is it > Med? (Target 1 for High only)
    t1_pos = n_2
    t1_neg = n_0 + n_1
    w1 = t1_neg / t1_pos if t1_pos > 0 else 1.0
    
    pos_weights = torch.tensor([w0, w1], device=device)
    print(f"BCE pos_weights: {pos_weights}")
    
    # Define Task Importance Weights (Zoom-In Strategy)
    # We give 2x weight to the harder "High" separation task
    task_importance = torch.tensor([1.0, 2.0], device=device)
    print(f"Task Importance Weights: {task_importance}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multiclass_collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multiclass_collate_fn, num_workers=4)

    # 3. Initialize Model
    spec_meta_dim = train_dataset[0][2].shape[1]
    print("Initializing Siamese Model...")
    
    # IMPORTANT: We set num_classes=2 because we need 2 output logits for the 2 tasks.
    # The model architecture code simply sets output_dim = num_classes.
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim,
        num_classes=2  # <--- 2 Output Nodes (Thresholds)
    ).to(device)
    
    if args.finetuned_encoder_path:
        print(f"Loading 'Zoom-In' Encoder: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft_weights: ft_weights = ft_weights['best_model_sd']
        model.encoder.load_state_dict(ft_weights, strict=False)

    if args.freeze_encoder:
        print("Freezing Encoder.")
        for param in model.encoder.parameters(): param.requires_grad = False

    # 4. Loss & Optimizer
    # BCEWithLogitsLoss handles the Sigmoid internally
    criterion_raw = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader, desc="Training"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward -> Logits [B, 2]
            logits = model(batch_A, batch_B, batch_meta) 
            
            # Generate Ordinal Targets [B, 2]
            targets = make_ordinal_targets(labels, num_tasks=2, device=device)
            
            # Calculate Loss (Unreduced)
            loss_per_task = criterion_raw(logits, targets)
            
            # Apply Task Importance Weights
            loss_weighted = loss_per_task * task_importance
            
            # Mean reduction
            loss = loss_weighted.mean()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                
                logits = model(batch_A, batch_B, batch_meta)
                
                # --- DECODING STRATEGY ---
                # 1. Sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                
                # 2. Threshold at 0.5 to get binary decisions [0/1, 0/1]
                binary_preds = (probs > 0.5).int()
                
                # 3. Sum to get the class (0, 1, or 2)
                # [0,0]->0, [1,0]->1, [1,1]->2
                preds = torch.sum(binary_preds, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Metrics
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New Best Model! Saving...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_ordinal_model.pth"))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--finetuned_encoder_path", type=str, required=True)
    
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ordinal_model")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--freeze_encoder", action='store_true')

    args = parser.parse_args()
    train_ordinal(args)