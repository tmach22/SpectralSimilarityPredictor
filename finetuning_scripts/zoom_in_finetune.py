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
    # Ensure we use the encoder wrapper that handles smart loading
    from updated_siamesemodel import MassFormerEncoder 
    from updated_train import merge_configs
    from zoom_in_data_loader import TripletRankingDataset, triplet_ranking_collate_fn
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    sys.exit(1)

def finetune(args):
    print("--- 1. Setting up Zoom-In Fine-Tuning ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Config and Model ---
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
    print(f"Initializing MassFormerEncoder from checkpoint: {args.checkpoint_path}")
    # Note: MassFormerEncoder will automatically detect if this is a 
    # 'demo.pkl' (original) or 'best_finetuned.pkl' (dictionary) and load correctly.
    model = MassFormerEncoder(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path,
    ).to(device)
    
    # --- Dynamic Unfreezing Logic ---
    print("Freezing all encoder parameters by default...")
    for param in model.parameters():
        param.requires_grad = False
    
    try:
        encoder_layers = model.encoder.encoder.graph_encoder.layers
        num_total_layers = len(encoder_layers)
        
        if args.unfreeze_layers == -1:
            print(f"Unfreezing ALL {num_total_layers} transformer layers...")
            for param in model.encoder.encoder.graph_encoder.parameters():
                param.requires_grad = True
        elif args.unfreeze_layers > 0:
            num_to_unfreeze = min(args.unfreeze_layers, num_total_layers)
            print(f"Unfreezing the last {num_to_unfreeze} layers...")
            for layer in encoder_layers[-num_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Unfreeze Final LayerNorm
            for param in model.encoder.encoder.graph_encoder.final_ln.parameters():
                param.requires_grad = True
    except AttributeError:
        print(f"Warning: Could not locate layers. Unfreezing ALL parameters.")
        for param in model.parameters():
            param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded with {trainable_params} trainable parameters.")

    # --- 3. Data Loading ---
    print("\n--- 2. Initializing Zoom-In Dataloaders ---")
    print(f"Using Margin Per Bin Base: {args.margin_per_bin}")
    
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

    val_dataset = TripletRankingDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        num_samples_per_epoch=args.val_epoch_size,
        margin_per_bin=args.margin_per_bin
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=triplet_ranking_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # --- 4. Optimizer ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )

    # --- 5. Training Loop ---
    print(f"\n--- 4. Starting 'Zoom-In' Fine-Tuning for {args.epochs} Epochs ---")
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, 'best_zoomin_encoder.pkl')

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            # UPDATED UNPACKING: Now receives 8 items
            # We ignore sim_low and sim_high (_) as they are for verification
            b_L1, b_L2, b_H1, b_H2, targets, batch_margins, _, _ = batch
            
            for b in [b_L1, b_L2, b_H1, b_H2]:
                for key in b: b[key] = b[key].to(device)
            
            # Move margin tensor to device (Crucial for dynamic margin loss)
            batch_margins = batch_margins.to(device)
            
            optimizer.zero_grad()
            
            emb_L1 = model({'gf_v2_data': b_L1})
            emb_L2 = model({'gf_v2_data': b_L2})
            emb_H1 = model({'gf_v2_data': b_H1})
            emb_H2 = model({'gf_v2_data': b_H2})
            
            # Cosine Distances (Range 0 to 2)
            dist_LowSim = 1.0 - F.cosine_similarity(emb_L1, emb_L2)
            dist_HighSim = 1.0 - F.cosine_similarity(emb_H1, emb_H2)
            
            # Manual Margin Loss: max(0, d_high - d_low + margin)
            # We want d_low > d_high + margin
            loss_per_item = F.relu(dist_HighSim - dist_LowSim + batch_margins)
            loss = loss_per_item.mean()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}")

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                b_L1, b_L2, b_H1, b_H2, targets, batch_margins, _, _ = batch

                for b in [b_L1, b_L2, b_H1, b_H2]:
                    for key in b: b[key] = b[key].to(device)
                batch_margins = batch_margins.to(device)

                emb_L1 = model({'gf_v2_data': b_L1})
                emb_L2 = model({'gf_v2_data': b_L2})
                emb_H1 = model({'gf_v2_data': b_H1})
                emb_H2 = model({'gf_v2_data': b_H2})

                dist_LowSim = 1.0 - F.cosine_similarity(emb_L1, emb_L2)
                dist_HighSim = 1.0 - F.cosine_similarity(emb_H1, emb_H2)

                loss_per_item = F.relu(dist_HighSim - dist_LowSim + batch_margins)
                running_val_loss += loss_per_item.mean().item()

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            print(f"New best model! Saving to {best_model_path}")
            
            checkpoint_to_save = {
                'best_model_sd': model.state_dict(),
                'zoomin_epoch': epoch + 1,
                'zoomin_val_loss': avg_val_loss
            }
            torch.save(checkpoint_to_save, best_model_path)
        else:
            early_stopping_counter += 1
            print(f"No improvement. Counter: {early_stopping_counter}/{args.patience}")
            
        if early_stopping_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print("\n--- Zoom-In Fine-Tuning Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Fine-tuning of MassFormer.")
    
    # Paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    
    # CRITICAL: Point this to your *fine-tuned* encoder!
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to 'best_finetuned_encoder.pkl' or demo.pkl")
    parser.add_argument("--output_dir", type=str, default="./zoomin_encoder")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--margin_per_bin", type=float, default=0.02, help="Base margin. Will be multiplied by weights (up to 3x).")
    parser.add_argument("--epoch_size", type=int, default=100000)
    parser.add_argument("--val_epoch_size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--unfreeze_layers", type=int, default=2)
    
    # System
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    finetune(args)