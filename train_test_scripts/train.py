import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import argparse

import os
import sys
cwd = Path.cwd()

data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)

model_dir = os.path.join(cwd, 'model')
print(f"Adding {model_dir} to sys.path")
sys.path.insert(0, model_dir)

from siamesemodel import SiameseSpectralSimilarityModel

parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)

# Import your custom classes
from data_loader import SpectralSimilarityDataset, siamese_collate_fn


def main(args):
    # --- 1. Configuration from args ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 2. Data Loading ---
    print("Setting up DataLoaders...")
    train_dataset = SpectralSimilarityDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = SpectralSimilarityDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=siamese_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=siamese_collate_fn, num_workers=4)

    # --- 3. Model, Loss, and Optimizer ---
    print("Setting up Model, Loss, and Optimizer...")
    model = SiameseSpectralSimilarityModel(args.config_path, args.checkpoint_path).to(DEVICE)
    loss_fn = nn.MSELoss()

    # ** CRITICAL: Set up differential learning rates **
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
        {'params': model.similarity_head.parameters(), 'lr': args.lr_head}
    ])

    # --- 4. Training and Validation Loop ---
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Training
        model.train()
        total_train_loss = 0
        for i, (batch_A, batch_B, similarities) in enumerate(train_loader):
            # Move data to device
            batch_A = {k: v.to(DEVICE) for k, v in batch_A.items()}
            batch_B = {k: v.to(DEVICE) for k, v in batch_B.items()}
            similarities = similarities.to(DEVICE)
            
            # Forward pass
            predictions = model(batch_A, batch_B)
            
            # Calculate loss
            loss = loss_fn(predictions, similarities)
            total_train_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Current Loss: {loss.item():.6f}")
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss (MSE): {avg_train_loss:.6f}")

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_A, batch_B, similarities in val_loader:
                batch_A = {k: v.to(DEVICE) for k, v in batch_A.items()}
                batch_B = {k: v.to(DEVICE) for k, v in batch_B.items()}
                similarities = similarities.to(DEVICE)
                
                predictions = model(batch_A, batch_B)
                loss = loss_fn(predictions, similarities)
                total_val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(similarities.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        rmse = (avg_val_loss) ** 0.5
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation Pearson Correlation: {pearson_corr:.4f}")

if __name__ == '__main__':
    # --- MODIFICATION: Added argparse for command-line configuration ---
    parser = argparse.ArgumentParser(description="Train a Siamese MassFormer model for spectral similarity prediction.")
    
    # Path arguments
    parser.add_argument("--config_path", type=str, default="/data/nas-gpu/wang/tmach007/massformer/config/demo/demo_eval.yml", help="Path to the MassFormer model config file.")
    parser.add_argument("--checkpoint_path", type=str, default="/data/nas-gpu/wang/tmach007/massformer/checkpoints/demo.pkl", help="Path to the pre-trained MassFormer weights (.pkl file).")
    parser.add_argument("--train_pairs_path", type=str, required=True, help="Path to the training pairs feather file.")
    parser.add_argument("--val_pairs_path", type=str, required=True, help="Path to the validation pairs feather file.")
    parser.add_argument("--spec_data_path", type=str, default="/data/nas-gpu/wang/tmach007/massformer/data/proc/spec_df.pkl", help="Path to the processed spec_df.pkl file.")
    parser.add_argument("--mol_data_path", type=str, default="/data/nas-gpu/wang/tmach007/massformer/data/proc/mol_df.pkl", help="Path to the processed mol_df.pkl file.")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--lr_encoder", type=float, default=1e-5, help="Learning rate for the pre-trained encoder.")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for the new similarity head.")
    
    args = parser.parse_args()
    main(args)