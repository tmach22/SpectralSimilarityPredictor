import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

# Setup paths (EXACTLY mirroring train_binary.py)
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Import your custom modules
try:
    from classifier_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def train_hard_finetune(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 4: Hard Example Fine-Tuning ---")
    
    # 1. Config & Model
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    # Initialize dataset first to get meta dim (Consistent with train_binary)
    print("Loading Training Data...")
    full_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    spec_meta_dim = full_dataset[0][2].shape[1]
    
    print("Initializing Model...")
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, # Base MassFormer weights
        spec_meta_dim=spec_meta_dim
    ).to(device)
    
    # 2. Load Your BEST Binary Model (The Champion BCE Model)
    print(f"Loading Best Binary Model from: {args.best_bce_weights}")
    state_dict = torch.load(args.best_bce_weights, map_location=device)
    
    # Robust loading: Handle cases where weights might be wrapped in 'state_dict' or 'model' keys
    # Your train_binary saves pure state_dict, but this safety check costs nothing.
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'best_model_sd' in state_dict:
        state_dict = state_dict['best_model_sd']
        
    # Load weights
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded binary classifier weights.")
    except RuntimeError as e:
        print(f"WARNING: Slight mismatch in loading weights (common if freezing/unfreezing layers): {e}")

    # 3. Filter for "Hard" Examples (The Logic of Hard Mining)
    print("Filtering for Hard Examples...")
    df = full_dataset.pairs_df
    
    # Hard Positives: Looks different (Tanimoto < 0.5) but IS similar (Label=1 or Cosine > 0.7)
    hard_pos_mask = (df['tanimoto'] < 0.5) & ((df['label'] == 1) | (df['cosine_similarity'] >= 0.7))
    # Hard Negatives: Looks similar (Tanimoto > 0.6) but IS NOT similar (Label=0 or Cosine < 0.7)
    hard_neg_mask = (df['tanimoto'] > 0.6) & ((df['label'] == 0) | (df['cosine_similarity'] < 0.7))
    
    hard_indices = df.index[hard_pos_mask | hard_neg_mask].tolist()
    
    if len(hard_indices) == 0:
        print("Error: No hard examples found in training set. Check your Tanimoto column.")
        return

    print(f"Found {len(hard_indices)} Hard Examples (out of {len(df)} total).")
    
    # Create Subset & Loader
    hard_dataset = Subset(full_dataset, hard_indices)
    train_loader = DataLoader(
        hard_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=binary_collate_fn, num_workers=args.num_workers
    )
    
    # Load Validation (Standard Full Validation to ensure we don't regress)
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, num_workers=args.num_workers
    )

    # 4. Setup Training (Low LR, Unfreeze Head + Last Layer)
    print("Freezing lower layers, unfreezing Head + Last Graph Layer...")
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Unfreeze Classification Head
    for param in model.similarity_head.parameters(): 
        param.requires_grad = True
    
    # Unfreeze Last Graph Layer (Optional: gives the model 'wiggle room' to adapt features)
    try:
        last_layer = model.encoder.encoder.graph_encoder.layers[-1]
        for param in last_layer.parameters(): 
            param.requires_grad = True
    except AttributeError:
        print("Warning: Could not access last graph layer to unfreeze. Training Head only.")

    # VERY LOW LR (1e-6) to fine-tune gently
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6) 
    criterion = nn.BCEWithLogitsLoss() # Consistent with Champion Model

    # 5. Training Loop
    best_val_f1 = 0.0
    
    print(f"Fine-tuning for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} (Hard FT)"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device).float() # BCE needs Float

            optimizer.zero_grad()
            logits = model(batch_A, batch_B, batch_meta).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                labels = labels.to(device).float()

                logits = model(batch_A, batch_B, batch_meta).view(-1)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save (Consistent with train_binary style)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(args.output_dir, "best_hard_finetuned_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best F1! Saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Paths (Same as train_binary)
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Configs
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    # NEW: Path to your current best model (The 0.78 F1 model)
    parser.add_argument("--best_bce_weights", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, default="./hard_finetune")
    parser.add_argument("--epochs", type=int, default=15) 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    
    args = parser.parse_args()
    train_hard_finetune(args)