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

# Setup paths
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Import your custom modules
from classifier_siamesemodel import SiameseSpectralSimilarityModel
from updated_train import merge_configs
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn

def train_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Binary Classification Training (Threshold 0.7) ---")
    print(f"Early Stopping Patience: {args.patience} epochs")
    
    # 1. Config & Model
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    # Initialize dataset first to get meta dim
    train_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    spec_meta_dim = train_dataset[0][2].shape[1]
    
    print("Initializing Model...")
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, # Base MassFormer weights
        spec_meta_dim=spec_meta_dim
    ).to(device)
    
    # 2. Load FINE-TUNED Encoder Weights (OPTIONAL)
    if args.finetuned_encoder_path:
        print(f"Loading FINE-TUNED encoder from: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        
        # Handle the dictionary structure from your finetuning script
        if 'best_model_sd' in ft_weights:
            ft_weights = ft_weights['best_model_sd']
            
        # Load into the encoder submodule
        missing, unexpected = model.encoder.load_state_dict(ft_weights, strict=False)
        print(f"Fine-tuned weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("No fine-tuned path provided. Using ORIGINAL MassFormer weights.")

    # 3. Freeze/Unfreeze Logic
    if args.freeze_encoder:
        print("Freezing Encoder. Training only the Classification Head.")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("Encoder is UNFROZEN (End-to-End Training).")

    # 4. Optimizer & Loss
    # Use BCEWithLogitsLoss for numerical stability (requires raw logits, NO Sigmoid in model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # Load Validation
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=binary_collate_fn, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, num_workers=args.num_workers
    )

    # Setup Tracking
    best_val_acc = 0.0
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []
        
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader, desc="Training"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device) # Shape [Batch]
            
            optimizer.zero_grad()
            logits = model(batch_A, batch_B, batch_meta).squeeze() # Shape [Batch]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store for metrics (Sigmoid here just for metric calc)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(all_labels, all_preds)
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

        # --- Val ---
        model.eval()
        val_preds, val_probs, val_labels = [], [], []
        
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                
                logits = model(batch_A, batch_B, batch_meta).squeeze()
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Metrics
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        roc = roc_auc_score(val_labels, val_probs)
        
        print(f"Val Acc: {acc:.4f} | Val F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
        
        # --- Early Stopping & Saving ---
        if acc > best_val_acc:
            best_val_acc = acc
            early_stop_counter = 0 # Reset counter
            print(f"New Best Model ({acc:.4f})! Saving...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_binary_model_zoomin.pth"))
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stopping counter: {early_stop_counter}/{args.patience}")
            
        if early_stop_counter >= args.patience:
            print("Early stopping triggered. Terminating training.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data Paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Model Paths
    # OPTIONAL: If None, uses base MassFormer weights
    parser.add_argument("--finetuned_encoder_path", type=str, default=None, help="Path to your 'best_finetuned_encoder.pkl'. Leave empty for baseline.")
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Base MassFormer checkpoint")
    parser.add_argument("--output_dir", type=str, default="./binary_model")
    
    # Training Args
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3, help="Stop if validation acc doesn't improve for N epochs.")
    
    # System Args
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_encoder", action='store_true', help="If set, freezes the encoder and only trains the head.")

    args = parser.parse_args()
    train_binary(args)