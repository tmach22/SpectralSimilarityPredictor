import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import copy

import os
import sys
cwd = Path.cwd()

data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)

model_dir = os.path.join(cwd, 'model')
print(f"Adding {model_dir} to sys.path")
sys.path.insert(0, model_dir)

# --- Import our custom classes ---
# Assuming this script is in the same directory as your model and data loader scripts
from updated_siamesemodel import SiameseSpectralSimilarityModel

parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)

from data_loader_v2 import SpectralSimilarityDataset, siamese_collate_fn

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

def train_model(args):
    """
    Main function to orchestrate the model training and validation process.
    """
    # --- 1. Setup and Configuration ---
    print("--- 1. Setting up training environment ---")
    
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CORRECTED CONFIG LOADING ---
    # Load the base template configuration
    print(f"Loading template configuration from: {args.template_config_path}")
    with open(args.template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    # Load the custom experiment configuration
    print(f"Loading custom configuration from: {args.custom_config_path}")
    with open(args.custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)

    # Merge the custom config on top of the template
    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
    # --- 2. Data Loading ---
    print("\n--- 2. Initializing Datasets and DataLoaders ---")
    
    train_dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        subset_size=args.subset_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=siamese_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True 
    )
    print(f"Training dataset contains {len(train_dataset)} pairs.")

    val_dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        subset_size=args.subset_size
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=siamese_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Validation dataset contains {len(val_dataset)} pairs.")

    # --- 3. Model, Loss, and Optimizer ---
    print("\n--- 3. Initializing Model, Loss Function, and Optimizer ---")

    print("Getting metadata dimension from dataset...")
    _, _, test_meta, _ = train_dataset[0]
    spec_meta_dim = test_meta.shape[1] # shape is [1, num_features]
    print(f"Detected spec_meta_dim: {spec_meta_dim}")
    
    model = SiameseSpectralSimilarityModel(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim
        # custom_encoder_weights_path=args.contrastive_checkpoint_path
    ).to(device)

    # --- Freeze the encoder for the first phase ---
    print("Freezing encoder weights for initial training phase.")
    for param in model.encoder.parameters():
        param.requires_grad = False

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- 4. Training Loop ---
    print("\n--- 4. Starting Model Training ---")
    best_val_loss = float('inf')

    early_stopping_patience = args.patience
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        model.train()
        running_train_loss = 0.0
        for batch_A, batch_B, batch_meta, similarities in tqdm(train_loader, desc="Training"):
            for key in batch_A: batch_A[key] = batch_A[key].to(device)
            for key in batch_B: batch_B[key] = batch_B[key].to(device)
            batch_meta = batch_meta.to(device)
            similarities = similarities.to(device)
            optimizer.zero_grad()
            predictions = model(batch_A, batch_B, batch_meta)
            loss = criterion(predictions.squeeze(), similarities)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.6f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, similarities in tqdm(val_loader, desc="Validating"):
                for key in batch_A: batch_A[key] = batch_A[key].to(device)
                for key in batch_B: batch_B[key] = batch_B[key].to(device)
                batch_meta = batch_meta.to(device)
                similarities = similarities.to(device)
                predictions = model(batch_A, batch_B, batch_meta)
                loss = criterion(predictions.squeeze(), similarities)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            print(f"New best validation loss: {best_val_loss:.6f}. Saving model...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Terminating training.")
            break  # Exit the training loop

    print("\n--- Training Complete ---")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Siamese Spectral Similarity Model.")
    
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # --- CORRECTED CONFIG PATHS ---
    parser.add_argument("--template_config_path", type=str, required=True, help="Path to the template.yml config file.")
    parser.add_argument("--custom_config_path", type=str, required=True, help="Path to the custom experiment config (e.g., demo_eval.yml).")
    
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--contrastive_checkpoint_path", type=str, default="path contrastive pretraining weights")
    parser.add_argument("--output_dir", type=str, default="./trained_model")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_size", type=float, default=1.0, help="Fraction of the dataset to use (for quick testing).")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for validation loss improvement before stopping.")

    args = parser.parse_args()
    train_model(args)