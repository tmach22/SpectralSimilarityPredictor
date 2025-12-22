import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import warnings
from pathlib import Path

# --- 0. SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")

# --- 1. SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
# Point to MassFormer source
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# --- 2. IMPORTS ---
try:
    from classifier_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 3. MIXUP HELPER ---
def mixup_data(z_a, z_b, meta, y, alpha=1.0, hard_targets=False, device='cuda'):
    '''
    Performs Manifold Mixup on the embeddings, metadata, and labels.
    
    Args:
        alpha (float): Beta distribution parameter. 
                       alpha=1.0 -> Uniform (0 to 1).
                       alpha=0.2 -> U-shaped (Mostly near 0 or 1, rarely 0.5).
        hard_targets (bool): If True, snaps label to the dominant parent.
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = z_a.size(0)
    index = torch.randperm(batch_size).to(device)

    # Mix Embeddings
    mixed_z_a = lam * z_a + (1 - lam) * z_a[index, :]
    mixed_z_b = lam * z_b + (1 - lam) * z_b[index, :]
    
    # Mix Metadata (Physics Injection)
    mixed_meta = lam * meta + (1 - lam) * meta[index, :]

    # Mix Labels
    y_a, y_b = y, y[index]
    
    if hard_targets:
        # Snap to dominant parent (no soft labels like 0.7)
        # If lam > 0.5, we are mostly A, so use label A. Else label B.
        if lam >= 0.5:
            mixed_y = y_a
        else:
            mixed_y = y_b
    else:
        # Standard Mixup (Soft labels, e.g. 0.7 * 1 + 0.3 * 0 = 0.7)
        mixed_y = lam * y_a + (1 - lam) * y_b
    
    return mixed_z_a, mixed_z_b, mixed_meta, mixed_y

def train_mixup(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 3: Manifold Mixup Fine-Tuning ---")
    print(f"Mode: {'Hard Targets' if args.hard_targets else 'Soft Targets'}")
    print(f"Alpha: {args.mixup_alpha}")
    print(f"Device: {device}")
    
    # 1. Config & Model Setup
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    train_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    val_dataset = BinaryClassificationDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    spec_meta_dim = train_dataset[0][2].shape[1]

    print("Initializing Siamese Model...")
    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)

    # 2. Load Phase 2 Weights
    if args.phase2_weights:
        print(f"Loading Phase 2 (Triplet) weights from: {args.phase2_weights}")
        state_dict = torch.load(args.phase2_weights, map_location=device)
        # Handle cases where state_dict might be nested under 'state_dict' or 'model' keys
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'best_model_sd' in state_dict: state_dict = state_dict['best_model_sd']
        
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully.")
        except RuntimeError as e:
            print(f"Weight loading warning (ignore if just head mismatch): {e}")
    else:
        print("WARNING: Starting from scratch (Not Recommended).")

    # 3. Smart Freezing Strategy
    print("\n--- Applying Freezing Strategy ---")
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze MLP Head
    for param in model.similarity_head.parameters():
        param.requires_grad = True
    print(" -> Unfrozen: MLP Similarity Head")

    # Unfreeze Last N Layers of Encoder
    try:
        encoder_layers = model.encoder.encoder.encoder.graph_encoder.layers
        total_layers = len(encoder_layers)
        
        params_unfrozen = 0
        for i in range(total_layers - 2, total_layers):
            print(f" -> Unfreezing Encoder Layer {i}")
            for param in encoder_layers[i].parameters():
                param.requires_grad = True
                params_unfrozen += param.numel()
        print(f" -> Total Unfrozen Encoder Params: {params_unfrozen}")
        
    except AttributeError as e:
        print(f"ERROR: Structure mismatch? {e}")
        return

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=binary_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=args.num_workers)

    # 5. Training Loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_A, batch_B, batch_meta, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Mixup]"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            # STEP A: Get Embeddings
            z_a = model.encoder({'gf_v2_data': batch_A})
            z_b = model.encoder({'gf_v2_data': batch_B})

            # STEP B: Manifold Mixup (with Tuning Options)
            mixed_z_a, mixed_z_b, mixed_meta, mixed_labels = mixup_data(
                z_a, z_b, batch_meta, labels, 
                alpha=args.mixup_alpha, 
                hard_targets=args.hard_targets, # <--- NEW FLAG
                device=device
            )

            # STEP C: Forward Pass (Head)
            sim_metric = F.cosine_similarity(mixed_z_a, mixed_z_b, dim=1).unsqueeze(1)
            combined_vector = torch.cat((sim_metric, mixed_meta), dim=1)
            logits = model.similarity_head(combined_vector).squeeze()

            # STEP D: Loss
            loss = criterion(logits, mixed_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # --- Validation (Standard) ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_A, batch_B, batch_meta, labels in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                batch_meta = batch_meta.to(device)
                labels = labels.to(device).float()

                logits = model(batch_A, batch_B, batch_meta).squeeze()
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # --- Save Best ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            save_path = os.path.join(args.output_dir, "best_mixup_model_v2.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model Saved! ({avg_val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Pre-trained Weights
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Base MassFormer Architecture")
    parser.add_argument("--phase2_weights", type=str, default=None, help="Path to best_finetuned_encoder_consensus.pkl")
    
    # Configs
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./mixup_model_v2")
    
    # Tuning Hyperparams
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Set to 0.2 for gentle mixing, 1.0 for strong mixing.")
    parser.add_argument("--hard_targets", action="store_true", help="Snap mixed targets to dominant parent (0 or 1) instead of soft interpolation.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train_mixup(args)