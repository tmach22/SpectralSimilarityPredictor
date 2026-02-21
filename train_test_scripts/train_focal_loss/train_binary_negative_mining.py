import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model', 'focal_loss'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Ensure you are importing the UPDATED model class
from classifier_siamesemodel_new import SiameseSpectralSimilarityModel
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn

def merge_configs(base_config, custom_config):
    merged_config = copy.deepcopy(base_config)
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    return merged_config

# --- 1. NEW COMPONENT: HARD AWARE FOCAL LOSS ---
class HardAwareFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, hard_mining_weight=3.0, reduction='mean'):
        super(HardAwareFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.hard_weight = hard_mining_weight
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets, h_A, h_B):
        """
        Calculates Focal Loss weighted by geometric hardness.
        """
        # A. Base Focal Loss
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        # B. Geometric Hardness Weighting
        with torch.no_grad():
            # Normalize to Hypersphere (Critical for consistency)
            embed_A = F.normalize(h_A, p=2, dim=1)
            embed_B = F.normalize(h_B, p=2, dim=1)
            
            # Euclidean Distance: ||A - B||
            dists = (embed_A - embed_B).norm(dim=1, p=2)
            dists = dists.view(-1, 1) # Match shape of targets [Batch, 1]
            
            # Initialize weights as 1.0
            weights = torch.ones_like(targets)
            
            # CASE 1: Hard Negatives (Blue tail entering Green zone)
            # True Label = 0, but Distance < 1.0 (Too close)
            hard_neg_mask = (targets == 0) & (dists < 1.0)
            weights[hard_neg_mask] = self.hard_weight
            
            # CASE 2: Hard Positives (Green tail entering Blue zone)
            # True Label = 1, but Distance > 0.75 (Too far)
            hard_pos_mask = (targets == 1) & (dists > 0.75)
            weights[hard_pos_mask] = self.hard_weight
            
        # Apply weights
        weighted_loss = focal_loss * weights

        if self.reduction == 'mean': return torch.mean(weighted_loss)
        elif self.reduction == 'sum': return torch.sum(weighted_loss)
        else: return weighted_loss

# --- 2. COMPONENT: ROBUST SMART UNFREEZE ---
def setup_differential_fine_tuning(model, verbose=True):
    if verbose: print("\n--- Configuring Encoder Fine-Tuning (Smart Unfreeze) ---")
    
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    trainable_encoder_params = []
    
    try:
        gf_encoder = model.encoder.encoder 
        encoder_layers = gf_encoder.encoder.graph_encoder.layers
        
        total_layers = len(encoder_layers)
        if verbose: 
            print(f" -> Detected {total_layers} Graph Encoder layers.")
            print(f" -> Unfreezing layers {total_layers-2} and {total_layers-1}...")
        
        for i in range(total_layers - 2, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
                trainable_encoder_params.append(param)
        
        if hasattr(gf_encoder.encoder, 'output_layer'):
            if verbose: print(" -> Unfreezing Encoder Output Layer...")
            for param in gf_encoder.encoder.output_layer.parameters():
                param.requires_grad = True
                trainable_encoder_params.append(param)

    except AttributeError as e:
        print(f" -> WARNING: Smart unfreeze failed ({e}). Fallback: Unfreezing FULL encoder.")
        for param in model.encoder.parameters():
            param.requires_grad = True
            trainable_encoder_params.append(param)

    if verbose: print(" -> Unfreezing Similarity Head & Mass Gate")
    for param in model.head.parameters(): param.requires_grad = True
    for param in model.mass_gate.parameters(): param.requires_grad = True
        
    return trainable_encoder_params

# --- 3. TRAINING FUNCTION ---
def train_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Binary Training (Hard-Aware Focal Loss) ---")

    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    train_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    spec_meta_dim = train_dataset[0][2].shape[1]
    print(f"Metadata Dimension detected: {spec_meta_dim}")

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)
    
    if args.resume_path:
        print(f"Resuming FULL model weights from {args.resume_path}")
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict)
    elif args.finetuned_encoder_path:
        print(f"Loading Fine-Tuned Encoder: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft_weights: ft_weights = ft_weights['best_model_sd']
        model.encoder.load_state_dict(ft_weights, strict=False)

    if args.freeze_encoder:
        print("Mode: HEAD ONLY (Encoder Frozen)")
        for param in model.encoder.parameters(): param.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else:
        print("Mode: DIFFERENTIAL FINE-TUNING (Head + Last 2 Layers)")
        encoder_params = setup_differential_fine_tuning(model)
        head_params = list(model.head.parameters()) + list(model.mass_gate.parameters())
        optimizer = optim.AdamW([
            {'params': head_params, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': encoder_params, 'lr': 1e-5, 'weight_decay': 1e-5}
        ])

    # --- UPDATED LOSS FUNCTION ---
    # Weight=3.0 to strongly penalize the overlap region identified in your plots
    criterion = HardAwareFocalLoss(alpha=0.25, gamma=2.0, hard_mining_weight=3.0).to(device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    best_val_auc = 0.0
    save_name = "best_hard_aware_model.pth"
    early_stop_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        
        for batch_A, batch_B, batch_meta, batch_mass_diffs, labels in tqdm(train_loader):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            batch_mass_diffs = batch_mass_diffs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # UPDATED: Unpack 3 values (Logits, Emb_A, Emb_B)
            logits, h_A, h_B = model(batch_A, batch_B, batch_meta, batch_mass_diffs)
            
            # UPDATED: Pass embeddings to loss
            loss = criterion(logits, labels, h_A, h_B)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss/len(train_loader):.4f}")

        # --- VALIDATION ---
        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, batch_mass_diffs, labels in tqdm(val_loader):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                batch_mass_diffs = batch_mass_diffs.to(device)
                
                # UPDATED: Unpack here as well (ignore embeddings for inference)
                logits, _, _ = model(batch_A, batch_B, batch_meta, batch_mass_diffs)
                
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels_list.extend(labels.numpy())

        try:
            auc = roc_auc_score(val_labels_list, val_probs)
        except:
            auc = 0.5
        
        print(f"Val AUC: {auc:.4f}")
        
        if auc > best_val_auc:
            best_val_auc = auc
            early_stop_counter = 0 
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
            print(f"New Best Model Saved (AUC {auc:.4f})")
        else:
            early_stop_counter += 1
            print(f"Early Stop Counter: {early_stop_counter}/{args.patience}")
            if early_stop_counter >= args.patience:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--finetuned_encoder_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./hard_aware_model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze_encoder", action='store_true')
    parser.add_argument("--patience", type=int, default=5)
    
    args = parser.parse_args()
    train_binary(args)