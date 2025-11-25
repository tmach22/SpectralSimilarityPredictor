# Save this as 'finetune_model.py'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import sys

# --- 1. SETUP SYS.PATH ---
cwd = Path.cwd()
# (Add your sys.path logic here as in your train_model.py)
data_loader_dir = os.path.join(cwd, 'data_loaders')
sys.path.insert(0, data_loader_dir)
model_dir = os.path.join(cwd, 'model')
sys.path.insert(0, model_dir)
train_dir = os.path.join(cwd, 'train_test_scripts')
sys.path.insert(0, train_dir)
parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
sys.path.insert(0, script_dir)

# --- 2. LOCAL IMPORTS ---
try:
    from updated_siamesemodel import MassFormerEncoder # Your encoder model
    from updated_train import merge_configs # Your config helper
    from finetune_dataloader import TripletRankingDataset, triplet_ranking_collate_fn
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print("Please check your sys.path and file names.")
    print(f"Original error: {e}")
    sys.exit(1)

def finetune(args):
    print("--- 1. Setting up fine-tuning environment ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Config and Model ---
    print(f"Loading template configuration from: {args.template_config_path}")
    with open(args.template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)
    print(f"Loading custom configuration from: {args.custom_config_path}")
    with open(args.custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
    print("Initializing MassFormerEncoder...")
    model = MassFormerEncoder(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path,
    ).to(device)
    
    # --- Dynamic Unfreezing Logic (Copied from your script) ---
    print("Freezing all encoder parameters by default...")
    for param in model.parameters():
        param.requires_grad = False
    
    try:
        # model.encoder is MassFormerEncoder, model.encoder.encoder is GFv2Embedder
        encoder_layers = model.encoder.encoder.graph_encoder.layers
        num_total_layers = len(encoder_layers)
        
        if args.unfreeze_layers == -1:
            # Unfreeze the entire GFv2Embedder
            print(f"Unfreezing ALL {num_total_layers} transformer layers and all embeddings...")
            for param in model.encoder.encoder.graph_encoder.parameters():
                param.requires_grad = True
        
        elif args.unfreeze_layers > 0:
            # Unfreeze the last N layers
            num_to_unfreeze = min(args.unfreeze_layers, num_total_layers) # Cap at max layers
            print(f"Unfreezing the last {num_to_unfreeze} of {num_total_layers} transformer layers...")
            
            for layer in encoder_layers[-num_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # (Your commented-out final_ln logic was here)
        
        elif args.unfreeze_layers == 0:
            print("Freezing all transformer layers (unfreeze_layers=0).")
    
    except AttributeError as e:
        print(f"Error while trying to unfreeze layers: {e}")
        print(f"Warning: Could not find encoder layers. Is 'model.encoder.encoder.graph_encoder.layers' the correct path?")
        print("Falling back to unfreezing *all* parameters.")
        for param in model.parameters():
            param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded with {trainable_params} trainable parameters.")
    
    if trainable_params == 0:
        print("\nWARNING: 0 trainable parameters. The model will not learn.")
    # --- End Unfreezing Logic ---


    # --- 3. Data Loading ---
    print("\n--- 2. Initializing Datasets and DataLoaders ---")
    train_dataset = TripletRankingDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        num_samples_per_epoch=args.epoch_size,
        margin_per_bin=args.margin_per_bin
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=triplet_ranking_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Training dataset size: {len(train_dataset)} (epoch size)")

    # --- *** NEW: VALIDATION DATALOADER *** ---
    print("Initializing validation dataset...")
    val_dataset = TripletRankingDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        num_samples_per_epoch=args.val_epoch_size, # Use a separate size for validation
        margin_per_bin=args.margin_per_bin
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle validation
        collate_fn=triplet_ranking_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Validation dataset size: {len(val_dataset)} (epoch size)")
    # --- *** END NEW *** ---
    
    # --- 4. Loss and Optimizer ---
    print("\n--- 3. Initializing Optimizer (Loss is manual) ---")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )
    
    if not optimizer.param_groups[0]['params']:
        print("Optimizer has 0 parameters to train. Exiting.")
        return

    # --- 5. Training Loop ---
    print(f"\n--- 4. Starting Fine-Tuning for {args.epochs} Epochs ---")
    
    # --- *** NEW: BEST MODEL & EARLY STOPPING LOGIC *** ---
    best_val_loss = float('inf')
    early_stopping_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, 'best_finetuned_encoder_unfreeze2.pkl')
    # --- *** END NEW *** ---

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for batch_L1, batch_L2, batch_H1, batch_H2, targets, batch_margins in tqdm(train_loader, desc="Training"):
            
            # Move all tensors to device
            for batch in [batch_L1, batch_L2, batch_H1, batch_H2]:
                for key in batch: batch[key] = batch[key].to(device)
            targets = targets.to(device)
            batch_margins = batch_margins.to(device)
            
            optimizer.zero_grad()
            
            emb_L1 = model({'gf_v2_data': batch_L1})
            emb_L2 = model({'gf_v2_data': batch_L2})
            emb_H1 = model({'gf_v2_data': batch_H1})
            emb_H2 = model({'gf_v2_data': batch_H2})
            
            dist_LowSim = 1.0 - F.cosine_similarity(emb_L1, emb_L2)
            dist_HighSim = 1.0 - F.cosine_similarity(emb_H1, emb_H2)
            
            loss_per_item = F.relu(dist_HighSim - dist_LowSim + batch_margins)
            loss = loss_per_item.mean()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.6f}")
        
        # --- *** NEW: VALIDATION PHASE *** ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_L1, batch_L2, batch_H1, batch_H2, targets, batch_margins in tqdm(val_loader, desc="Validating"):
                
                for batch in [batch_L1, batch_L2, batch_H1, batch_H2]:
                    for key in batch: batch[key] = batch[key].to(device)
                targets = targets.to(device)
                batch_margins = batch_margins.to(device)

                emb_L1 = model({'gf_v2_data': batch_L1})
                emb_L2 = model({'gf_v2_data': batch_L2})
                emb_H1 = model({'gf_v2_data': batch_H1})
                emb_H2 = model({'gf_v2_data': batch_H2})

                dist_LowSim = 1.0 - F.cosine_similarity(emb_L1, emb_L2)
                dist_HighSim = 1.0 - F.cosine_similarity(emb_H1, emb_H2)

                loss_per_item = F.relu(dist_HighSim - dist_LowSim + batch_margins)
                loss = loss_per_item.mean()
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.6f}")
        
        # --- *** NEW: BEST MODEL & EARLY STOPPING LOGIC *** ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            print(f"New best validation loss! Saving model to {best_model_path}")
            
            finetuned_state_dict = model.state_dict()
            checkpoint_to_save = {
                'best_model_sd': finetuned_state_dict,
                'fine_tuned_epoch': epoch + 1,
                'fine_tuned_val_loss': avg_val_loss
            }
            torch.save(checkpoint_to_save, best_model_path)
            
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Counter: {early_stopping_counter}/{args.patience}")
            
        if early_stopping_counter >= args.patience:
            print("Early stopping triggered. Terminating training.")
            break # Exit the training loop
        # --- *** END NEW *** ---

    print("\n--- Fine-Tuning Complete ---")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")
    print(f"Best model saved to {best_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune the MassFormer encoder.")
    
    # Data paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True) # <-- NEW
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Config/Model paths
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the *base* MassFormer .pkl checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./finetuned_encoder")
    
    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Small LR for fine-tuning")
    parser.add_argument("--epochs", type=int, default=50) # Increased max epochs for early stopping
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--margin_per_bin", type=float, default=0.02, help="Dynamic margin to enforce per bin difference.")
    parser.add_argument("--epoch_size", type=int, default=100000, help="Number of random triplets per training epoch")
    parser.add_argument("--val_epoch_size", type=int, default=10000, help="Number of random triplets per validation epoch") # <-- NEW
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (e.g., 3)") # <-- NEW
    
    # Model Hyperparameters
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of *last* transformer layers to unfreeze. 0 = freeze all, 2 = unfreeze last 2 (default), -1 = unfreeze all.")
    
    # System
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    print(f"Arguments: {args}")
    finetune(args)