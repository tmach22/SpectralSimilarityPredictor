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

# NEW: Import the necessary PEFT libraries from Hugging Face
from peft import get_peft_model, LoraConfig

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
    Main function to orchestrate the PEFT fine-tuning process.
    """
    # --- 1. Setup and Configuration ---
    print("--- 1. Setting up PEFT (LoRA) Fine-Tuning Environment ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Config and Data Loading (remains the same) ---
    print(f"Loading template configuration from: {args.template_config_path}")
    with open(args.template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)
    print(f"Loading custom configuration from: {args.custom_config_path}")
    with open(args.custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
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
    print(f"Training dataset contains {len(train_dataset)} pairs.")
    print(f"Validation dataset contains {len(val_dataset)} pairs.")

    # --- 3. Model Initialization and PEFT Configuration ---
    print("\n--- 3. Initializing Model for PEFT (LoRA) Fine-Tuning ---")
    
    model = SiameseSpectralSimilarityModel(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path
    )

    print(f"Loading weights from Phase 1 checkpoint: {args.load_from_checkpoint}")
    model.load_state_dict(torch.load(args.load_from_checkpoint, map_location=device))
    
    # =============================================================================
    # *** NEW: Configure and Apply LoRA using the PEFT library ***
    # =============================================================================
    print("\nConfiguring LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"], # Target the attention layers in Graphormer
        lora_dropout=0.1,
        bias="none",
    )

    # Wrap the model with the PEFT configuration. This freezes the base model
    # and injects the trainable LoRA adapters automatically.
    model = get_peft_model(model, lora_config)
    
    print("\n--- Model Architecture with LoRA ---")
    model.print_trainable_parameters()
    
    model.to(device)

    # The optimizer will now automatically see only the trainable parameters
    # (the LoRA adapters and the similarity head).
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # --- 4. Training Loop ---
    print("\n--- 4. Starting Model Training ---")
    best_val_loss = float('inf')
    early_stopping_patience = args.patience
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        model.train()
        running_train_loss = 0.0
        for batch_A, batch_B, similarities in tqdm(train_loader, desc="Training"):
            for key in batch_A: batch_A[key] = batch_A[key].to(device, non_blocking=True)
            for key in batch_B: batch_B[key] = batch_B[key].to(device, non_blocking=True)
            similarities = similarities.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            predictions = model(batch_A, batch_B)
            loss = criterion(predictions.squeeze(), similarities)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.6f}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_A, batch_B, similarities in tqdm(val_loader, desc="Validating"):
                for key in batch_A: batch_A[key] = batch_A[key].to(device, non_blocking=True)
                for key in batch_B: batch_B[key] = batch_B[key].to(device, non_blocking=True)
                similarities = similarities.to(device, non_blocking=True)
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
            # Save only the trained PEFT adapter, not the whole model
            model.save_pretrained(args.output_dir)
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Terminating training.")
            break

    print("\n--- Training Complete ---")
    print(f"Best PEFT adapter saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune the Siamese Model with PEFT (LoRA).")
    
    # --- Data & Model Paths ---
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original MassFormer checkpoint (demo.pkl).")
    parser.add_argument("--output_dir", type=str, default="./trained_model_peft")
    
    # --- PEFT & Fine-Tuning Arguments ---
    parser.add_argument("--load_from_checkpoint", type=str, required=True, 
                        help="Path to the Phase 1 model checkpoint (e.g., best_model.pth).")
    parser.add_argument("--lora_r", type=int, default=8, help="The rank of the LoRA update matrices.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="The scaling factor for the LoRA update.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the PEFT training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")

    # --- Other Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--subset_size", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train_model(args) 