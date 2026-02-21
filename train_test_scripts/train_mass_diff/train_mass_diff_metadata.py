import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import copy
from sklearn.metrics import roc_auc_score

from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'mass_gate'))
sys.path.insert(0, os.path.join(cwd, 'model', 'mass_gate'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Imports
from classifier_siamesemodel_md_metadata import SiameseSpectralSimilarityModel 
from binary_data_loader_metadata import BinaryClassificationDataset, binary_collate_fn

def merge_configs(base, custom):
    merged = copy.deepcopy(base)
    for k, v in custom.items():
        if isinstance(v, dict) and k in merged:
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        f_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(f_loss)
        return torch.sum(f_loss)

def train_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Training: Mass Diff as Feature (Focal Loss) ---")

    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    train_ds = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_ds = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # Check Meta Dim (Should include the +1 for mass diff now)
    spec_meta_dim = train_ds[0][2].shape[1]
    print(f"Metadata Dimension: {spec_meta_dim}")

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)
    
    if args.resume_path:
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
    elif args.finetuned_encoder_path:
        ft = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft: ft = ft['best_model_sd']
        model.encoder.load_state_dict(ft, strict=False)

    # Freeze Encoder, Train Head
    for param in model.encoder.parameters(): param.requires_grad = False
    
    # Only train head
    optimizer = optim.AdamW(model.similarity_head.parameters(), lr=args.learning_rate)
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    best_val_auc = 0.0
    save_name = "best_model_md_feature.pth"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(batch_A, batch_B, batch_meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                logits = model(batch_A, batch_B, batch_meta)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(labels.numpy())

        auc = roc_auc_score(val_labels, val_probs)
        print(f"Val AUC: {auc:.4f}")
        
        if auc > best_val_auc:
            best_val_auc = auc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
            print(f"Saved Best Model (AUC {auc:.4f})")

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
    parser.add_argument("--output_dir", type=str, default="./results_md_feature")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    
    args = parser.parse_args()
    train_binary(args)