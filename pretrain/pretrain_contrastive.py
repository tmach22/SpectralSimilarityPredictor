import argparse
import os
import time
import pandas as pd
import yaml
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

# --- Path setup to import MassFormer modules ---
from pathlib import Path
import sys
cwd = Path.cwd()
# This assumes your script is run from a directory where the parent contains the 'tmach007' folder
# Adjust this path if your project structure is different
parent_directory = os.path.dirname(cwd.parent)
massformer_script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
if massformer_script_dir not in sys.path:
    print(f"Adding {massformer_script_dir} to sys.path")
    sys.path.insert(0, massformer_script_dir)

data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)

model_dir = os.path.join(cwd, 'model')
print(f"Adding {model_dir} to sys.path")
sys.path.insert(0, model_dir)

# --- Import custom modules ---
# The new data loader components from your separate file
from data_loader_contrastive import ContrastiveDataset, siamese_collate_fn
# The MassFormerEncoder from your model file
from siamesemodel import MassFormerEncoder

def merge_configs(base_config, custom_config):
    """
    Recursively merges the custom config into the base config.
    """
    merged_config = copy.deepcopy(base_config)
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    return merged_config

# --- 1. Siamese Model and Contrastive Loss Definition ---

class SiameseMassFormer(nn.Module):
    """Siamese network using the MassFormer encoder for contrastive learning."""
    def __init__(self, model_config: dict, checkpoint_path: str):
        super(SiameseMassFormer, self).__init__()
        # This is the shared encoder for both inputs
        self.encoder = MassFormerEncoder(model_config, checkpoint_path)

    def forward(self, mol_a: dict, mol_b: dict) -> tuple:
        """
        Passes each molecule through the shared-weight encoder.
        The input is wrapped in a dictionary to match the encoder's expected format.
        """
        output_a = self.encoder({'gf_v2_data': mol_a})
        output_b = self.encoder({'gf_v2_data': mol_b})
        return output_a, output_b

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Pulls positive pairs (label=1) closer and pushes negative pairs (label=0) apart.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Loss for positive pairs (label=1) - pull them together
        loss_positive = label * torch.pow(euclidean_distance, 2)
        
        # Loss for negative pairs (label=0) - push them apart by at least the margin
        loss_negative = (1 - label) * torch.pow(F.relu(self.margin - euclidean_distance), 2)
        
        loss_contrastive = torch.mean(loss_positive + loss_negative)
        
        return loss_contrastive

# --- 2. Main Training Script ---

def main():
    parser = argparse.ArgumentParser(description="Pre-train MassFormer encoder with contrastive loss.")
    
    # --- Paths ---
    parser.add_argument("--train_pairs_path", type=str, required=True, help="Path to pretrain_contrastive_train.feather.")
    parser.add_argument("--val_pairs_path", type=str, required=True, help="Path to pretrain_contrastive_validation.feather.")
    parser.add_argument("--mol_df_path", type=str, required=True, help="Path to mol_df.pkl.")
    parser.add_argument("--template_config_path", type=str, required=True, help="Path to the MassFormer model YAML config file.")
    parser.add_argument("--custom_config_path", type=str, required=True, help="Path to the custom experiment config (e.g., demo_eval.yml).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the official MassFormer weights (e.g., demo.pkl).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new pre-trained encoder.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=50, help="Max number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=2.0, help="Margin for contrastive loss.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader.")
    parser.add_argument("--subset_size", type=float, default=None, help="Use a fraction of the dataset for quick testing (e.g., 0.1 for 10%).")

    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device: {device} ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data Loading ---
    print("\n--- 1. Loading Data ---")
    mol_df = pd.read_pickle(args.mol_df_path)
    mol_df['mol_id'] = mol_df['mol_id'].astype(int)
    
    train_dataset = ContrastiveDataset(args.train_pairs_path, mol_df, subset_size=args.subset_size)
    val_dataset = ContrastiveDataset(args.val_pairs_path, mol_df, subset_size=args.subset_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=siamese_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=siamese_collate_fn, pin_memory=True)

    # --- Model Initialization ---
    print("\n--- 2. Initializing Model ---")
    print(f"Loading template configuration from: {args.template_config_path}")
    with open(args.template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    with open(args.custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)

    # Merge the custom config on top of the template
    full_config = merge_configs(template_config, custom_config)
    
    model = SiameseMassFormer(full_config['model'], args.checkpoint_path).to(device)
    loss_function = ContrastiveLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Training Loop ---
    print("\n--- 3. Starting Pre-training ---")
    best_val_loss = float('inf')
    patience_counter = 0
    encoder_path = os.path.join(args.output_dir, 'massformer_encoder_contrastive_pretrained.pt')

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        for mol_a, mol_b, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            mol_a = {k: v.to(device, non_blocking=True) for k, v in mol_a.items()}
            mol_b = {k: v.to(device, non_blocking=True) for k, v in mol_b.items()}
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output_a, output_b = model(mol_a, mol_b)
            loss = loss_function(output_a, output_b, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for mol_a, mol_b, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                mol_a = {k: v.to(device, non_blocking=True) for k, v in mol_a.items()}
                mol_b = {k: v.to(device, non_blocking=True) for k, v in mol_b.items()}
                labels = labels.to(device, non_blocking=True)
                
                output_a, output_b = model(mol_a, mol_b)
                loss = loss_function(output_a, output_b, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s")

        # --- Early Stopping and Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save only the state_dict of the encoder
            torch.save(model.encoder.state_dict(), encoder_path)
            print(f"New best model saved to {encoder_path} (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
            break
            
    print("\n--- Pre-training Complete ---")
    print(f"Best pre-trained encoder saved at: {encoder_path}")

if __name__ == '__main__':
    main()