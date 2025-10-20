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
from torch.optim.lr_scheduler import LambdaLR

# --- (Your existing sys.path setup) ---
cwd = Path.cwd()
data_loader_dir = os.path.join(cwd, 'data_loaders')
sys.path.insert(0, data_loader_dir)
model_dir = os.path.join(cwd, 'model')
sys.path.insert(0, model_dir)
from siamesemodel import SiameseSpectralSimilarityModel
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)
from data_loader import SpectralSimilarityDataset, siamese_collate_fn

def merge_configs(base_config, custom_config):
    #... (this function remains the same)
    merged_config = copy.deepcopy(base_config)
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    return merged_config


def train_model(args):
    # --- 1. Setup and Configuration ---
    print("--- 1. Setting up Gradual Fine-Tuning Environment ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Config Loading ---
    #... (this section remains the same)
    print(f"Loading template configuration from: {args.template_config_path}")
    with open(args.template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)
    print(f"Loading custom configuration from: {args.custom_config_path}")
    with open(args.custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
    # --- 2. Data Loading ---
    #... (this section remains the same)
    print("\n--- 2. Initializing Datasets and DataLoaders ---")
    train_dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        subset_size=args.subset_size
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=siamese_collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        subset_size=args.subset_size
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=siamese_collate_fn, num_workers=args.num_workers, pin_memory=True)

    # --- 3. Model, Loss, and Optimizer ---
    print("\n--- 3. Initializing Model for Gradual Unfreezing ---")
    
    model = SiameseSpectralSimilarityModel(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path
    )

    print(f"Loading weights from checkpoint: {args.load_from_checkpoint}")
    model.load_state_dict(torch.load(args.load_from_checkpoint, map_location=device))
    
    # =============================================================================
    # *** NEW: Gradual Unfreezing and Differential Learning Rate Logic ***
    # =============================================================================
    
    # First, freeze all parameters in the entire model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the similarity head
    for param in model.similarity_head.parameters():
        param.requires_grad = True

    # Identify the encoder's transformer layers (there are 12 in graphormer_base)
    encoder_layers = model.encoder.encoder.encoder.graph_encoder.layers
    num_total_layers = len(encoder_layers)

    # Unfreeze the top N layers of the encoder
    if args.num_layers_to_unfreeze > 0:
        layers_to_unfreeze = encoder_layers[-args.num_layers_to_unfreeze:]
        print(f"Unfreezing the top {len(layers_to_unfreeze)} layers of the encoder (Layers {num_total_layers - args.num_layers_to_unfreeze} to {num_total_layers-1}).")
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
    
    model.to(device)

    # Create parameter groups for the optimizer
    print("Setting up differential learning rates:")
    print(f"  - Similarity Head LR: {args.head_lr}")
    print(f"  - Unfrozen Encoder LR: {args.unfrozen_encoder_lr}")
    
    optimizer_parameters = [
        {'params': model.similarity_head.parameters(), 'lr': args.head_lr}
    ]
    if args.num_layers_to_unfreeze > 0:
        optimizer_parameters.append(
            {'params': layers_to_unfreeze.parameters(), 'lr': args.unfrozen_encoder_lr}
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params:,} trainable parameters.")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(optimizer_parameters, weight_decay=args.weight_decay)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)
    print(f"Using a linear learning rate scheduler with {num_warmup_steps} warmup steps.")
    
    # --- 4. Training Loop (remains the same) ---
    #... (copy the entire training loop from your existing train.py, including early stopping)
    print("\n--- 4. Starting Model Training ---")
    best_val_loss = float('inf')
    early_stopping_patience = args.patience
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        model.train()
        running_train_loss = 0.0
        for i, (batch_A, batch_B, similarities) in enumerate(tqdm(train_loader, desc="Training")):
            for key in batch_A: batch_A[key] = batch_A[key].to(device)
            for key in batch_B: batch_B[key] = batch_B[key].to(device)
            similarities = similarities.to(device)
            optimizer.zero_grad()
            predictions = model(batch_A, batch_B)
            loss = criterion(predictions.squeeze(), similarities)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # =============================================================================
            # *** NEW: Detailed Debugging Logs (optional) ***
            # =============================================================================
            # Log gradients and weights every N batches to avoid slowing down training
            if (i + 1) % 1000 == 0: # Log every 1000 batches
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                # Get the L2 norm of the weights for the similarity head
                head_weight_norm = 0
                for p in model.similarity_head.parameters():
                    head_weight_norm += p.data.norm(2).item() ** 2
                head_weight_norm = head_weight_norm ** 0.5

                print(f"\n Grad Norm: {total_norm:.4f}, Head Weight Norm: {head_weight_norm:.4f}")
            # =============================================================================```

            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.6f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_A, batch_B, similarities in tqdm(val_loader, desc="Validating"):
                for key in batch_A: batch_A[key] = batch_A[key].to(device)
                for key in batch_B: batch_B[key] = batch_B[key].to(device)
                similarities = similarities.to(device)
                predictions = model(batch_A, batch_B)
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
    parser = argparse.ArgumentParser(description="Fine-tune the Siamese Model with Gradual Unfreezing.")
    
    # --- Data & Model Paths ---
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original MassFormer checkpoint (demo.pkl).")
    parser.add_argument("--output_dir", type=str, default="./trained_model_gradual")
    
    # =============================================================================
    # *** NEW: Arguments for controlling the gradual fine-tuning process ***
    # =============================================================================
    parser.add_argument("--load_from_checkpoint", type=str, required=True, 
                        help="Path to the model checkpoint to start from (e.g., best_model.pth).")
    parser.add_argument("--num_layers_to_unfreeze", type=int, default=3,
                        help="Number of encoder layers to unfreeze from the top down (0-12).")
    parser.add_argument("--head_lr", type=float, default=1e-5, 
                        help="Learning rate for the similarity head.")
    parser.add_argument("--unfrozen_encoder_lr", type=float, default=5e-6, 
                        help="Learning rate for the unfrozen encoder layers.")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay (L2 penalty) for the AdamW optimizer.")

    # --- Other Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--subset_size", type=float, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train_model(args)