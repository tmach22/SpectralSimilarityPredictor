import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Ensure you are importing the UPDATED model class (Decoupled Architecture)
from classifier_siamesemodel_new import SiameseSpectralSimilarityModel
from updated_train import merge_configs
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn

def train_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Binary Training with Mass Gating (Physics-Aware) ---")

    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    train_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # Dynamically calculate meta dim
    spec_meta_dim = train_dataset[0][2].shape[1]
    print(f"Metadata Dimension detected: {spec_meta_dim}")

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)
    
    # --- 1. Weight Loading Logic ---
    if args.resume_path:
        # PHASE 2: Load FULL model (Heads + Encoder) from previous phase
        print(f"PHASE 2: Resuming FULL model weights from {args.resume_path}")
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict)
        
    elif args.finetuned_encoder_path:
        # PHASE 1 / STANDARD: Load ONLY Encoder weights
        print(f"Loading Fine-Tuned Encoder: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft_weights: ft_weights = ft_weights['best_model_sd']
        model.encoder.load_state_dict(ft_weights, strict=False)

    # --- 2. Freeze/Unfreeze Logic ---
    if args.freeze_encoder:
        print("PHASE 1: Freezing ENTIRE Encoder (Training Heads Only)...")
        for param in model.encoder.parameters(): 
            param.requires_grad = False
    else:
        # PHASE 2: Smart Unfreeze (Last 2 Layers Only)
        print("PHASE 2: Unfreezing Last 2 Layers of Graph Encoder...")
        
        # First, freeze everything in encoder to be safe
        for param in model.encoder.parameters(): 
            param.requires_grad = False
            
        try:
            # Navigate to the graph encoder layers
            gf_encoder = model.encoder.encoder 
            encoder_layers = gf_encoder.encoder.graph_encoder.layers
            
            total_layers = len(encoder_layers)
            print(f"  > Detected {total_layers} Graph Encoder layers.")
            print(f"  > Unfreezing layers {total_layers-2} and {total_layers-1}...")
            
            for i in range(total_layers - 2, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
                    
        except AttributeError as e:
            print(f"  > WARNING: Smart unfreeze failed ({e}). Fallback: Unfreezing FULL encoder.")
            for param in model.encoder.parameters(): 
                param.requires_grad = True

    # Weighted Loss for Imbalanced Data
    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer (Filters out frozen params automatically)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    best_val_acc = 0.0
    # Determine save name based on phase
    save_name = "best_phase1_model.pth" if args.freeze_encoder else "best_phase2_model.pth"
    
    # --- EARLY STOPPING INIT ---
    early_stop_counter = 0
    print(f"Early stopping patience set to: {args.patience}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        
        for batch_A, batch_B, batch_meta, batch_mass_diffs, labels in tqdm(train_loader):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            batch_mass_diffs = batch_mass_diffs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Pass mass_diffs to model
            logits = model(batch_A, batch_B, batch_meta, batch_mass_diffs).squeeze()
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss/len(train_loader):.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, batch_mass_diffs, labels in tqdm(val_loader):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                batch_mass_diffs = batch_mass_diffs.to(device)
                
                logits = model(batch_A, batch_B, batch_meta, batch_mass_diffs).squeeze()
                
                preds = torch.sigmoid(logits) > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Val Acc: {acc:.4f}")
        
        # --- EARLY STOPPING LOGIC ---
        if acc > best_val_acc:
            best_val_acc = acc
            early_stop_counter = 0 # Reset counter if improvement found
            
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model Saved: {save_path}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early Stop Counter: {early_stop_counter}/{args.patience}")
            
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered! Best Validation Accuracy: {best_val_acc:.4f}")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    # Checkpoint args
    parser.add_argument("--finetuned_encoder_path", type=str, default=None, help="Load ONLY encoder weights")
    parser.add_argument("--resume_path", type=str, default=None, help="Load FULL model (Phase 2)")
    
    parser.add_argument("--output_dir", type=str, default="./gated_model")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze_encoder", action='store_true')
    
    # NEW ARGUMENT
    parser.add_argument("--patience", type=int, default=3, help="Epochs to wait for improvement before stopping")
    
    args = parser.parse_args()
    train_binary(args)