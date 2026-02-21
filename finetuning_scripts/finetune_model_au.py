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
from pathlib import Path
import copy

# --- SETUP ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'AlignUniform'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from classifier_siamesemodel_new import MassFormerEncoder
    from finetune_dataloader_au import AlignUniformDataset, au_collate_fn
except ImportError as e:
    sys.exit(f"Error: {e}")

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
# EXPLICIT ALIGNMENT & UNIFORMITY LOSS (Strict Version)
# =============================================================================

class AlignUniformLoss(nn.Module):
    def __init__(self, t=2, alpha=2, lam_align=1.0, lam_unif=1.0):
        super(AlignUniformLoss, self).__init__()
        self.t = t             
        self.alpha = alpha     
        self.lam_align = lam_align
        self.lam_unif = lam_unif

    def forward(self, features):
        """
        features: [2N, Dim] (Normalized)
        Assumes alternating structure: [A1, P1, A2, P2, ...]
        """
        # --- 1. Alignment Loss (Pull Positive Pairs) ---
        anchors = features[0::2]
        positives = features[1::2]
        
        diff = anchors - positives
        l_align = diff.norm(dim=1).pow(self.alpha).mean()

        # --- 2. Uniformity Loss (Strict Push on Anchors Only) ---
        sq_pdist_anchors = torch.pdist(anchors, p=2).pow(2)
        l_unif = torch.log(torch.mean(torch.exp(-self.t * sq_pdist_anchors)))

        # --- 3. Total Loss ---
        loss = (self.lam_align * l_align) + (self.lam_unif * l_unif)
        
        return loss, l_align, l_unif

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_align_uniform(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("--- EXPLICIT ALIGNMENT & UNIFORMITY (FROM ORIGINAL WEIGHTS) ---")

    # 1. Config & Model
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    # Initialize Encoder
    model = MassFormerEncoder(full_config.get('model', {}), checkpoint_path=None).to(device)
    
    # 2. LOAD ORIGINAL WEIGHTS
    if os.path.exists(args.original_weights_path):
        print(f"Loading Original MassFormer weights from {args.original_weights_path}...")
        checkpoint = torch.load(args.original_weights_path, map_location=device)
        
        # Robust Loading Logic
        sd = checkpoint
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
            
        # Try loading into the encoder submodule
        try:
            # Case A: Keys match model.encoder (common in fine-tuning saves)
            model.encoder.load_state_dict(sd, strict=False)
            print("✅ Loaded weights directly into encoder.")
        except RuntimeError:
            print("⚠️ Direct load failed, attempting prefix matching...")
            # Case B: Original checkpoint might have 'network.0.' prefixes or similar
            # We strip prefixes to find matches
            model_dict = model.encoder.state_dict()
            pretrained_dict = {k: v for k, v in sd.items() if k in model_dict}
            
            # If that failed, try stripping 'module.' or 'encoder.' from the checkpoint keys
            if len(pretrained_dict) == 0:
                pretrained_dict = {}
                for k, v in sd.items():
                    new_key = k.replace("module.", "").replace("encoder.", "").replace("network.", "")
                    if new_key in model_dict:
                        pretrained_dict[new_key] = v
            
            model.encoder.load_state_dict(pretrained_dict, strict=False)
            print(f"✅ Loaded {len(pretrained_dict)}/{len(model_dict)} layers via flexible matching.")
            
    else:
        print(f"❌ Error: Weights file not found at {args.original_weights_path}")
        sys.exit(1)

    # 3. Dataset
    train_dataset = AlignUniformDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=au_collate_fn, num_workers=4, drop_last=True)
    
    val_dataset = AlignUniformDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=au_collate_fn, num_workers=4)

    # 4. Setup Loss & Optimizer
    criterion = AlignUniformLoss(lam_align=args.lam_align, lam_unif=args.lam_unif).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 5. Training Loop
    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        model.train()
        train_align = 0.0
        train_unif = 0.0
        
        for batch_graphs in tqdm(train_loader, desc="Training"):
            for k in batch_graphs: batch_graphs[k] = batch_graphs[k].to(device)
            
            optimizer.zero_grad()
            emb = model({'gf_v2_data': batch_graphs})
            emb = F.normalize(emb, dim=1)
            
            loss, l_a, l_u = criterion(emb)
            loss.backward()
            optimizer.step()
            
            train_align += l_a.item()
            train_unif += l_u.item()
            
        n = len(train_loader)
        print(f"Train Align: {train_align/n:.4f} | Train Unif: {train_unif/n:.4f}")

        # VALIDATION
        model.eval()
        val_loss_total = 0.0
        val_align = 0.0
        val_unif = 0.0
        
        with torch.no_grad():
            for batch_graphs in tqdm(val_loader, desc="Validating"):
                for k in batch_graphs: batch_graphs[k] = batch_graphs[k].to(device)
                emb = model({'gf_v2_data': batch_graphs})
                emb = F.normalize(emb, dim=1)
                loss, l_a, l_u = criterion(emb)
                
                val_loss_total += loss.item()
                val_align += l_a.item()
                val_unif += l_u.item()
        
        n_val = len(val_loader)
        avg_val_loss = val_loss_total / n_val
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"► Val Align: {val_align/n_val:.4f} (Goal: < 0.25)")
        print(f"► Val Unif:  {val_unif/n_val:.4f} (Goal: ~ -2.0)")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New Best Model! Saving...")
            torch.save({'best_model_sd': model.encoder.state_dict()}, 
                       os.path.join(args.output_dir, "best_au_msgmona.pkl"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    
    # REQUIRED: Path to original weights
    parser.add_argument("--original_weights_path", type=str, required=True, help="Path to original massformer.pkl")
    
    parser.add_argument("--output_dir", type=str, default="./au_original")
    parser.add_argument("--lam_align", type=float, default=1.0)
    parser.add_argument("--lam_unif", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5) # Slightly lower LR for fine-tuning
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    train_align_uniform(args)