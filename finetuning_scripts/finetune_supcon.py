import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import sys
import copy

# --- 1. SETUP SYS.PATH ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'supcon'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# --- 2. LOCAL IMPORTS ---
try:
    # Your base encoder (MassFormer)
    from classifier_siamesemodel_new import MassFormerEncoder
    # The SupCon Data Loader
    from finetune_dataloader import SupConPairsDataset, supcon_pairs_collate_fn
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print("Please check your sys.path and file names.")
    print(f"Original error: {e}")
    sys.exit(1)

# =============================================================================
# 3. HELPER FUNCTIONS & CLASSES
# =============================================================================

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

def compute_uniformity_fast(x, t=2):
    """
    Calculates Uniformity metric on a batch of normalized vectors.
    L_unif = log( mean( exp( -t * ||x-y||^2 ) ) )
    Goal: < -2.0. If > -0.5, COLLAPSE detected.
    """
    if x.shape[0] > 2000: # Subsample for speed if batch is huge
        idx = torch.randperm(x.shape[0])[:2000]
        x = x[idx]
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return torch.log(torch.mean(torch.exp(-t * sq_pdist)))

class SupConLoss(nn.Module):
    """
    Robust Supervised Contrastive Loss with NaN protection.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # Flatten features if needed: [Batch, Views, Dim] -> [Batch*Views, Dim]
        if len(features.shape) > 2:
             features = features.view(features.shape[0] * features.shape[1], -1)

        batch_size = features.shape[0]
        
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            if mask is None:
                mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            else:
                mask = mask.float().to(device)

        # Compute Similarity Matrix (Logits)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask for self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        # Mask selects valid positives (excluding self)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # --- NAN PROTECTION START ---
        # Count positives per anchor
        mask_pos_pairs = mask.sum(1)
        
        # Identify valid anchors (those that have at least 1 positive pair)
        valid_anchors = mask_pos_pairs > 0
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Avoid division by zero for invalid anchors (we will filter them out later anyway)
        safe_divisor = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        
        # Calculate mean log_prob for positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / safe_divisor

        # Only average loss over VALID anchors
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid_anchors].mean()
        # --- NAN PROTECTION END ---

        return loss

class MassFormerSupCon(nn.Module):
    """
    Wrapper model with Projection Head.
    """
    def __init__(self, encoder, input_dim=768, feat_dim=128):
        super().__init__()
        self.encoder = encoder 
        
        # Projection Head: input_dim -> input_dim -> 128
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, feat_dim)
        )
        
    def forward(self, batch_data):
        # 1. Representation
        feat = self.encoder({'gf_v2_data': batch_data})
        feat = F.normalize(feat, dim=1)
        
        # 2. Projection
        z = self.head(feat)
        z = F.normalize(z, dim=1)
        
        return z

# =============================================================================
# 4. TRAINING FUNCTION
# =============================================================================

def train_supcon(args):
    print("--- 1. Setting up SupCon Environment ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Config & Base Encoder ---
    print(f"Loading configs...")
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    
    print("Initializing Base MassFormerEncoder...")
    base_encoder = MassFormerEncoder(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
    )
    
    # --- AUTO-DETECT DIMENSION ---
    # Try to grab the dimension from the first layer to avoid runtime errors
    try:
        # Navigate to GraphEncoder layer 0 FC1
        test_dim = base_encoder.encoder.encoder.graph_encoder.layers[0].fc1.in_features
        print(f"✅ Auto-detected Encoder Output Dimension: {test_dim}")
        enc_dim = test_dim
    except:
        print("⚠️ Could not auto-detect dimension. Defaulting to 768.")
        enc_dim = 768

    # --- Wrap in SupCon Model ---
    print(f"Wrapping with Projection Head (Input: {enc_dim})...")
    model = MassFormerSupCon(base_encoder, input_dim=enc_dim).to(device)

    # --- Unfreezing Logic ---
    print("Unfreezing all parameters for SupCon...")
    for param in model.parameters(): param.requires_grad = True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    # --- Data Loading ---
    print("\n--- 2. Initializing Datasets ---")
    train_dataset = SupConPairsDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=supcon_pairs_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    print(f"Train Size: {len(train_dataset)}")
    
    val_dataset = SupConPairsDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=supcon_pairs_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Val Size: {len(val_dataset)}")

    # --- Loss & Optimizer ---
    criterion = SupConLoss(temperature=args.temperature).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    # --- Training Loop ---
    print(f"\n--- 3. Starting Training (Temp={args.temperature}, LR={args.learning_rate}) ---")
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, 'best_supcon_encoder.pkl')

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # TRAINING
        model.train()
        running_loss = 0.0
        
        for batch_A, batch_B, labels_A, labels_B in tqdm(train_loader, desc="Training"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            labels_A = labels_A.to(device)
            labels_B = labels_B.to(device)
            
            optimizer.zero_grad()
            
            # 1. Forward
            z1 = model(batch_A) 
            z2 = model(batch_B)
            
            # 2. Combine for Loss [2*Batch, Dim]
            features_flat = torch.cat([z1, z2], dim=0).unsqueeze(1) # [2N, 1, 128]
            labels_flat = torch.cat([labels_A, labels_B], dim=0)    # [2N]
            
            # 3. Loss
            loss = criterion(features_flat, labels_flat)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # VALIDATION (With Uniformity Tracking)
        model.eval()
        running_val_loss = 0.0
        running_uniformity = 0.0
        
        with torch.no_grad():
            for batch_A, batch_B, labels_A, labels_B in tqdm(val_loader, desc="Validating"):
                for k in batch_A: batch_A[k] = batch_A[k].to(device)
                for k in batch_B: batch_B[k] = batch_B[k].to(device)
                labels_A = labels_A.to(device)
                labels_B = labels_B.to(device)

                z1 = model(batch_A)
                z2 = model(batch_B)
                
                features_flat = torch.cat([z1, z2], dim=0).unsqueeze(1)
                labels_flat = torch.cat([labels_A, labels_B], dim=0)
                
                # Loss
                loss = criterion(features_flat, labels_flat)
                running_val_loss += loss.item()
                
                # Uniformity Metric (Track Features in flattened form)
                # Squeeze the extra dim to get [2N, 128]
                feats_for_metric = features_flat.squeeze(1)
                unif = compute_uniformity_fast(feats_for_metric)
                running_uniformity += unif.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_uniformity = running_uniformity / len(val_loader)
        
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1} Val Uniformity: {avg_uniformity:.4f} (Goal: < -2.0, Collapse: > -0.5)")

        # CHECKPOINTING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            print(f"New Best Model! Saving Encoder to {best_model_path}")
            checkpoint_to_save = {
                'best_model_sd': model.encoder.state_dict(), # Strip head
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_uniformity': avg_uniformity
            }
            torch.save(checkpoint_to_save, best_model_path)
        else:
            early_stop_counter += 1
            print(f"No improvement. Counter: {early_stop_counter}/{args.patience}")
            if early_stop_counter >= args.patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SupCon Fine-tuning for MassFormer")
    
    # Paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./supcon_encoder")
    
    # Params (Updated Defaults for Collapse Prevention)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Lower LR to prevent collapse")
    parser.add_argument("--temperature", type=float, default=0.2, help="Higher Temp to prevent collapse")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128, help="Higher Batch Size is better")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()
    train_supcon(args)