import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from pathlib import Path
import copy

# --- 1. SETUP SYS.PATH ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'triplet_loss'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# --- 2. LOCAL IMPORTS ---
try:
    from classifier_siamesemodel_new import MassFormerEncoder 
    from finetune_dataloader_triplet import TripletOnlineDataset, triplet_collate_fn
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    sys.exit(1)

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

# =============================================================================
# 3. ONLINE TRIPLET LOSS (Batch Hard Mining)
# =============================================================================

class ConstrainedTripletLoss(nn.Module):
    
    def __init__(self, margin=0.2, align_weight=1.0):
        super(ConstrainedTripletLoss, self).__init__()
        self.margin = margin
        self.align_weight = align_weight # <--- NEW PARAMETER

    def forward(self, embeddings, labels):
        # ... (Same Distance Matrix Calculation as before) ...
        dot_product = torch.matmul(embeddings, embeddings.t())
        dists = 2.0 - 2.0 * dot_product
        dists = torch.clamp(dists, min=0.0)

        # ... (Same Masking) ...
        labels = labels.unsqueeze(1)
        mask_pos = torch.eq(labels, labels.t()).float()
        mask_pos = mask_pos - torch.eye(labels.size(0)).to(labels.device) 
        mask_neg = 1.0 - torch.eq(labels, labels.t()).float()

        # MINING
        anchor_pos_dists = dists * mask_pos
        hardest_pos_dist, _ = anchor_pos_dists.max(dim=1) # Max dist positive
        
        max_dist = dists.max().detach()
        anchor_neg_dists = dists + max_dist * (1.0 - mask_neg)
        hardest_neg_dist, _ = anchor_neg_dists.min(dim=1) # Min dist negative
        
        # 1. Standard Triplet Loss
        triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin).mean()
        
        # 2. EXPLICIT ALIGNMENT PENALTY
        # We also want to minimize the absolute distance of the hardest positive
        alignment_loss = hardest_pos_dist.mean()
        
        # Combined
        total_loss = triplet_loss + (self.align_weight * alignment_loss)
        
        return total_loss

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_triplet(args):
    print("--- 1. Setting up Triplet Training ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Config ---
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    # --- Initialize Base Encoder ---
    print("Initializing MassFormerEncoder...")
    model = MassFormerEncoder(
        model_config=full_config.get('model', {}),
        checkpoint_path=None
    ).to(device)
    
    # --- Load Pre-trained Weights (SupCon) ---
    if args.supcon_encoder_path and os.path.exists(args.supcon_encoder_path):
        print(f"Loading SupCon weights from {args.supcon_encoder_path}...")
        checkpoint = torch.load(args.supcon_encoder_path, map_location=device)
        
        # Handle potentially nested keys
        if 'best_model_sd' in checkpoint:
            sd = checkpoint['best_model_sd']
        elif 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        else:
            sd = checkpoint
            
        # Clean keys if needed (remove 'encoder.' prefix if double wrapped)
        # Usually MassFormerEncoder keys start with 'encoder.' or 'projection.'
        # We assume strict compatibility with updated_siamesemodel.py
        try:
            model.encoder.load_state_dict(sd, strict=False)
            print("✅ Weights loaded successfully.")
        except RuntimeError as e:
            print(f"⚠️ Warning during loading: {e}")
    else:
        print("⚠️ No SupCon path provided or file not found. Starting from scratch (NOT RECOMMENDED).")

    # --- Data Loading ---
    print("\n--- 2. Initializing Triplet Data Loader ---")
    # Training Set
    train_dataset = TripletOnlineDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, # This is N pairs (2N graphs)
        shuffle=True, 
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        drop_last=True
    )
    print(f"Train Size: {len(train_dataset)} pairs")
    
    # Validation Set (Using Triplet Loader too to monitor metric learning)
    val_dataset = TripletOnlineDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers
    )

    # --- Optimizer & Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = ConstrainedTripletLoss(margin=args.margin).to(device)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # --- Loop ---
    print(f"\n--- 3. Starting Training (Margin={args.margin}, LR={args.learning_rate}) ---")
    
    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, 'best_triplet_encoder.pkl')

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # TRAIN
        model.train()
        running_loss = 0.0
        
        for batch_graphs, batch_labels in tqdm(train_loader, desc="Training"):
            # Move to device
            for k in batch_graphs: batch_graphs[k] = batch_graphs[k].to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward (Get Embeddings)
            embeddings = model({'gf_v2_data': batch_graphs})
            embeddings = F.normalize(embeddings, dim=1) # Vital for Triplet Loss
            
            # Loss (Automatic Mining)
            loss = criterion(embeddings, batch_labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # VALIDATION
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_graphs, batch_labels in tqdm(val_loader, desc="Validating"):
                for k in batch_graphs: batch_graphs[k] = batch_graphs[k].to(device)
                batch_labels = batch_labels.to(device)
                
                embeddings = model({'gf_v2_data': batch_graphs})
                embeddings = F.normalize(embeddings, dim=1)
                
                loss = criterion(embeddings, batch_labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        # Update Scheduler
        scheduler.step(avg_val_loss)

        # CHECKPOINT
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New Best Model! Saving to {best_model_path}")
            torch.save({'best_model_sd': model.encoder.state_dict(), 'epoch': epoch}, best_model_path)
        else:
            print("No improvement.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./triplet_encoder")
    
    # Pretrained Weights
    parser.add_argument("--supcon_encoder_path", type=str, required=True, 
                        help="Path to the best_supcon_encoder.pkl")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Number of PAIRS (Actual batch size will be 2x this)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train_triplet(args)