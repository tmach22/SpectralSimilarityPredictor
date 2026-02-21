import torch
import torch.nn as nn
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
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Ensure you are importing the UPDATED model class
from classifier_siamesemodel_new import SiameseSpectralSimilarityModel
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn

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

# --- 1. COMPONENT: FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        f_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean': return torch.mean(f_loss)
        elif self.reduction == 'sum': return torch.sum(f_loss)
        else: return f_loss

# --- 2. COMPONENT: ROBUST SMART UNFREEZE ---
def setup_differential_fine_tuning(model, verbose=True):
    """
    Configures Option 3:
    - Freezes the bottom of the Graph Encoder.
    - Dynamically finds and unfreezes the Last 2 Layers of the Graph Encoder.
    - Unfreezes Similarity Head & Mass Gate.
    """
    if verbose: print("\n--- Configuring Encoder Fine-Tuning (Smart Unfreeze) ---")
    
    # 1. Freeze EVERYTHING in the encoder first
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    trainable_encoder_params = []
    
    # 2. Dynamic Navigation to Unfreeze Last 2 Layers
    try:
        # Navigate to the graph encoder layers (Adjust path if architecture differs slightly)
        # Structure based on MassFormer: model.encoder -> GFv2Embedder -> Encoder -> GraphEncoder -> Layers
        gf_encoder = model.encoder.encoder 
        encoder_layers = gf_encoder.encoder.graph_encoder.layers
        
        total_layers = len(encoder_layers)
        if verbose: 
            print(f" -> Detected {total_layers} Graph Encoder layers.")
            print(f" -> Unfreezing layers {total_layers-2} and {total_layers-1}...")
        
        # Unfreeze the last 2 layers
        for i in range(total_layers - 2, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
                trainable_encoder_params.append(param)
        
        # Also unfreeze the output/readout layer of the encoder if it exists
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

    # 3. Ensure Head and Mass Gate are ALWAYS trainable
    if verbose: print(" -> Unfreezing Similarity Head & Mass Gate")
    for param in model.head.parameters(): param.requires_grad = True
    for param in model.mass_gate.parameters(): param.requires_grad = True
        
    return trainable_encoder_params

# --- 3. TRAINING FUNCTION ---
def train_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Binary Training (Focal Loss + Diff. Fine-Tuning) ---")

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
    
    # --- Weight Loading Logic ---
    if args.resume_path:
        print(f"Resuming FULL model weights from {args.resume_path}")
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict)
    elif args.finetuned_encoder_path:
        print(f"Loading Fine-Tuned Encoder: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft_weights: ft_weights = ft_weights['best_model_sd']
        model.encoder.load_state_dict(ft_weights, strict=False)

    # --- OPTIMIZER SETUP ---
    if args.freeze_encoder:
        print("Mode: HEAD ONLY (Encoder Frozen)")
        for param in model.encoder.parameters(): param.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else:
        # OPTION 3: Simultaneous Differential Fine-Tuning
        print("Mode: DIFFERENTIAL FINE-TUNING (Head + Last 2 Layers)")
        encoder_params = setup_differential_fine_tuning(model)
        
        head_params = list(model.head.parameters()) + list(model.mass_gate.parameters())
        
        optimizer = optim.AdamW([
            # Group A: The Head (Train Fast - 1e-3)
            {'params': head_params, 'lr': 1e-3, 'weight_decay': 1e-4},
            # Group B: The Encoder (Train Slow - 1e-5)
            {'params': encoder_params, 'lr': 1e-5, 'weight_decay': 1e-5}
        ])

    # --- LOSS FUNCTION ---
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    best_val_auc = 0.0
    save_name = "best_focal_model_msgmona.pth"
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
            logits = model(batch_A, batch_B, batch_meta, batch_mass_diffs)
            loss = criterion(logits, labels)
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
                
                logits = model(batch_A, batch_B, batch_meta, batch_mass_diffs)
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
    parser.add_argument("--output_dir", type=str, default="./gated_focal_model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--freeze_encoder", action='store_true')
    parser.add_argument("--patience", type=int, default=5)
    
    args = parser.parse_args()
    train_binary(args)