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
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from multitrain_siamesemodel import SiameseSpectralSimilarityModel
from updated_train import merge_configs
from multitrain_data_loader import MulticlassDataset, multiclass_collate_fn

def train_multitask(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Multi-Task Training (Spectral + Structural) ---")
    
    # 1. Config & Dataset
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    print("Initializing Dataset...")
    train_dataset = MulticlassDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = MulticlassDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    
    # Calculate Class Weights for Spectral Task
    print("Calculating class weights...")
    y_train_sims = train_dataset.pairs_df['cosine_similarity'].values
    y_train_labels = []
    for s in y_train_sims:
        if s < 0.65: y_train_labels.append(0)
        elif s < 0.85: y_train_labels.append(1)
        else: y_train_labels.append(2)
    
    class_weights = compute_class_weight('balanced', classes=[0,1,2], y=y_train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class Weights: {class_weights_tensor}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=multiclass_collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=multiclass_collate_fn, num_workers=4)

    # 2. Initialize Model
    spec_meta_dim = train_dataset[0][2].shape[1]
    print("Initializing Multi-Task Model...")
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim,
        num_classes=3 
    ).to(device)
    
    # Load Fine-Tuned Encoder
    if args.finetuned_encoder_path:
        print(f"Loading 'Zoom-In' Encoder: {args.finetuned_encoder_path}")
        ft_weights = torch.load(args.finetuned_encoder_path, map_location=device)
        if 'best_model_sd' in ft_weights: ft_weights = ft_weights['best_model_sd']
        model.encoder.load_state_dict(ft_weights, strict=False)

    # --- 3. SURGICAL UNFREEZING LOGIC ---
    print("Freezing all encoder parameters by default...")
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    if args.unfreeze_layers > 0:
        try:
            # Locate the layers list
            encoder_layers = model.encoder.encoder.encoder.graph_encoder.layers
            num_total_layers = len(encoder_layers)
            num_to_unfreeze = min(args.unfreeze_layers, num_total_layers)
            
            print(f"Unfreezing the last {num_to_unfreeze} of {num_total_layers} transformer layers...")
            
            # Unfreeze the specific layers
            for layer in encoder_layers[-num_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Always unfreeze the final LayerNorm
            print("Unfreezing the final LayerNorm...")
            for param in model.encoder.encoder.encoder.graph_encoder.emb_layer_norm.parameters():
                param.requires_grad = True
                
        except Exception as e:
            print("Warning: Could not locate layers by name. Falling back to unfreezing EVERYTHING (Risky).")
            print(f"Error: {e}")
            for param in model.encoder.parameters():
                param.requires_grad = True
    
    # Verify trainable params
    trainable_encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Encoder Params: {trainable_encoder_params}")
    print(f"Total Trainable Params: {total_trainable}")

    # --- 4. Loss & Optimizer ---
    criterion_spectral = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion_structure = nn.MSELoss() 
    
    # Use AdamW with Weight Decay for regularization
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay # Added Weight Decay
    )
    
    best_val_f1 = 0.0
    lambda_struct = args.lambda_structure 
    
    # --- 5. Training Loop ---
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        total_loss = 0.0
        spec_loss_sum = 0.0
        struct_loss_sum = 0.0
        
        for batch_A, batch_B, batch_meta, labels, tanimotos in tqdm(train_loader, desc="Training"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device)
            tanimotos = tanimotos.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward
            spec_logits, tanimoto_preds = model(batch_A, batch_B, batch_meta)
            
            # Losses
            loss_a = criterion_spectral(spec_logits, labels)
            loss_b = criterion_structure(tanimoto_preds, tanimotos)
            
            loss = loss_a + (lambda_struct * loss_b)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            spec_loss_sum += loss_a.item()
            struct_loss_sum += loss_b.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f} (Spec: {spec_loss_sum/len(train_loader):.4f}, Struct: {struct_loss_sum/len(train_loader):.4f})")

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels, tanimotos in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                
                spec_logits, _ = model(batch_A, batch_B, batch_meta)
                preds = torch.argmax(spec_logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Metrics
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New Best Model! Saving...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_multitask_model.pth"))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Model
    parser.add_argument("--finetuned_encoder_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./multitask_model")
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Use small LR for fine-tuning")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    # Regularization & Tuning
    parser.add_argument("--lambda_structure", type=float, default=0.5, help="Weight for Tanimoto loss")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of encoder layers to unfreeze")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")

    args = parser.parse_args()
    train_multitask(args)