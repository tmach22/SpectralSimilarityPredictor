import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import copy
from sklearn.metrics import roc_auc_score

# Setup Paths
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Imports
from classifier_siamesemodel import SiameseSpectralSimilarityModel
from irm_data_loader import IRMBinaryDataset, irm_collate_fn

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
# IRM LOSS FUNCTION (The Gradient Norm Penalty)
# =============================================================================
def compute_irm_penalty(logits, y, device):
    """
    Computes the IRMv1 penalty: || grad(Loss, w) ||^2 where w is fixed at 1.0.
    This effectively measures how much the 'optimal' classifier would shift
    if trained only on this specific environment.
    """
    # 1. Create the dummy "linear classifier" w=1.0
    # We treat the logits as the features, and 'scale' as the classifier weight.
    scale = torch.tensor(1.).to(device).requires_grad_()
    
    # 2. Compute Loss with this scaled output
    # scaling logits is equivalent to scaling the last layer weights
    loss = nn.BCEWithLogitsLoss()(logits * scale, y)
    
    # 3. Compute Gradient of Loss w.r.t. Scale
    # create_graph=True is essential for higher-order derivatives
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    
    # 4. Penalty is the squared norm of that gradient
    return torch.sum(grad**2)

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_irm(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("--- Starting IRM Training (MassFormer Backbone) ---")
    print(f"Penalty Weight: {args.irm_penalty_weight} | Anneal Epochs: {args.irm_anneal_epochs}")

    # 1. Config & Data
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    print("Loading Data...")
    train_ds = IRMBinaryDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_ds = IRMBinaryDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # IRM benefits from larger batches to ensure environments are mixed
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=irm_collate_fn, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=irm_collate_fn, num_workers=args.num_workers)

    # 3rd item is spec_meta
    spec_meta_dim = train_ds[0][2].shape[1]
    print(f"Metadata Dimension: {spec_meta_dim}")

    # 2. Model Init
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)

    # Load Weights (Resume or Pretrained)
    if args.resume_path:
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
        print("Loaded resumed weights.")
    elif args.finetuned_encoder_path:
        ft = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft: ft = ft['best_model_sd']
        model.encoder.load_state_dict(ft, strict=False)
        print("Loaded fine-tuned encoder.")

    # 3. Optimizer
    # We use a lower LR for IRM to prevent instability
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_val_auc = 0.0
    save_name = "best_irm_model_5class.pth"
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_penalty = 0.0
        total_erm = 0.0
        
        # --- Penalty Annealing Schedule ---
        # ERM Warmup: Penalty is 0.0 for first N epochs
        if epoch < args.irm_anneal_epochs:
            penalty_weight = 0.0
        else:
            penalty_weight = args.irm_penalty_weight
            
        print(f"\nEpoch {epoch+1}/{args.epochs} | IRM Penalty: {penalty_weight:.1f}")

        # Unpack 5 items from IRM loader
        for batch_A, batch_B, batch_meta, labels, envs in tqdm(train_loader):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device).unsqueeze(1)
            envs = envs.to(device)

            optimizer.zero_grad()
            
            # Forward Pass (Standard)
            logits = model(batch_A, batch_B, batch_meta)
            
            # --- IRM Logic ---
            # 1. Compute Standard ERM Loss (Mean over batch)
            erm_loss = nn.BCEWithLogitsLoss()(logits, labels)
            
            # 2. Compute IRM Penalty per Environment
            penalty = torch.tensor(0.).to(device)
            valid_envs_count = 0
            
            # Iterate through unique environments present in this batch
            unique_envs = torch.unique(envs)
            for env_id in unique_envs:
                # Mask data for this environment
                mask = (envs == env_id)
                
                # Need at least 2 samples to compute a meaningful gradient variance/norm
                if mask.sum() > 1:
                    env_logits = logits[mask]
                    env_labels = labels[mask]
                    
                    # Compute Gradient Norm Penalty for this specific environment
                    penalty += compute_irm_penalty(env_logits, env_labels, device)
                    valid_envs_count += 1
            
            if valid_envs_count > 0:
                penalty = penalty / valid_envs_count
            
            # 3. Combine
            final_loss = erm_loss + (penalty_weight * penalty)
            
            # 4. Backward
            final_loss.backward()
            
            # Gradient Clipping is CRITICAL for IRM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += final_loss.item()
            total_erm += erm_loss.item()
            if isinstance(penalty, torch.Tensor):
                total_penalty += penalty.item()
            else:
                total_penalty += penalty
        
        avg_loss = total_loss / len(train_loader)
        avg_erm = total_erm / len(train_loader)
        avg_pen = total_penalty / len(train_loader)
        
        print(f"Train Loss: {avg_loss:.4f} | ERM: {avg_erm:.4f} | Penalty: {avg_pen:.6f}")

        # --- VALIDATION ---
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels, _ in tqdm(val_loader, desc="Val"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                
                logits = model(batch_A, batch_B, batch_meta)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(labels.numpy())

        try:
            auc = roc_auc_score(val_labels, val_probs)
        except:
            auc = 0.5
        
        print(f"Val AUC: {auc:.4f}")
        
        if auc > best_val_auc:
            best_val_auc = auc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
            print(f"New Best IRM Model Saved (AUC {auc:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # [Standard Args]
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--finetuned_encoder_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./irm_results")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5) # Lower LR
    parser.add_argument("--num_workers", type=int, default=8)
    
    # [IRM Specific Args]
    parser.add_argument("--irm_penalty_weight", type=float, default=100.0, help="Lambda for IRM penalty")
    parser.add_argument("--irm_anneal_epochs", type=int, default=5, help="Epochs to wait before applying penalty")

    args = parser.parse_args()
    train_irm(args)