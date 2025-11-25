import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Setup paths
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from multiclass_classifier_siamesemodel import SiameseSpectralSimilarityModel
from updated_train import merge_configs
from multiclass_data_loader import MulticlassDataset, multiclass_collate_fn

def train_multiclass(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Multiclass Training (3 Classes) ---")
    
    # 1. Config & Dataset
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    print("Initializing Dataset...")
    train_dataset = MulticlassDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = MulticlassDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # --- Calculate Class Weights (UPDATED FOR 3 CLASSES) ---
    print("Calculating class weights...")
    y_train_sims = train_dataset.pairs_df['cosine_similarity'].values
    y_train_labels = []
    for s in y_train_sims:
        if s < 0.65: y_train_labels.append(0)
        elif s < 0.85: y_train_labels.append(1)
        else: y_train_labels.append(2)
    
    class_weights = compute_class_weight('balanced', classes=[0,1,2], y=y_train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class Weights (0, 1, 2): {class_weights_tensor}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multiclass_collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multiclass_collate_fn, num_workers=4)

    # 2. Initialize Model
    spec_meta_dim = train_dataset[0][2].shape[1]
    print("Initializing Siamese Model...")
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim,
        num_classes=3  # <--- UPDATED TO 3 CLASSES
    ).to(device)
    
    # Load Fine-Tuned Encoder (Zoom-In Version)
    if args.finetuned_encoder_path:
        print(f"Loading 'Zoom-In' Encoder: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft_weights: ft_weights = ft_weights['best_model_sd']
        model.encoder.load_state_dict(ft_weights, strict=False)

    if args.freeze_encoder:
        print("Freezing Encoder.")
        for param in model.encoder.parameters(): param.requires_grad = False

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader, desc="Training"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_A, batch_B, batch_meta) 
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                
                logits = model(batch_A, batch_B, batch_meta)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New Best Model! Saving...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_multiclass_model_3class_v2.pth"))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--finetuned_encoder_path", type=str, required=True)
    
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./multiclass_model_3class")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--freeze_encoder", action='store_true')

    args = parser.parse_args()
    train_multiclass(args)