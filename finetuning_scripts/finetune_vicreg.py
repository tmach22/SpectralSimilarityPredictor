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
# VICReg LOSS IMPLEMENTATION
# =============================================================================
class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, epsilon=1e-4):
        """
        Standard VICReg weights are 25-25-1.
        """
        super(VICRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def forward(self, features):
        """
        features: [2N, Dim] (UNNORMALIZED)
        Assumes alternating structure: [A1, B1, A2, B2, ...]
        """
        # Split alternating batch back into View A and View B
        z_a = features[0::2]
        z_b = features[1::2]
        
        N, D = z_a.size()
        
        # --- 1. INVARIANCE (Similarity) ---
        # Pull positive pairs together
        sim_loss = F.mse_loss(z_a, z_b)
        
        # --- 2. VARIANCE (Std) ---
        # Hinge loss to keep standard deviation > 1.0
        std_a = torch.sqrt(z_a.var(dim=0) + self.epsilon)
        std_b = torch.sqrt(z_b.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_a)) / 2 + torch.mean(F.relu(1 - std_b)) / 2
        
        # --- 3. COVARIANCE (Decorrelation) ---
        # Center the embeddings
        z_a_centered = z_a - z_a.mean(dim=0)
        z_b_centered = z_b - z_b.mean(dim=0)
        
        # Calculate covariance matrices
        cov_a = (z_a_centered.T @ z_a_centered) / (N - 1)
        cov_b = (z_b_centered.T @ z_b_centered) / (N - 1)
        
        # Zero out the diagonal (we only penalize off-diagonal correlations)
        diag_a = torch.diag(cov_a.diagonal())
        diag_b = torch.diag(cov_b.diagonal())
        cov_a_off = cov_a - diag_a
        cov_b_off = cov_b - diag_b
        
        # Sum of squared off-diagonal elements
        cov_loss = (cov_a_off.pow(2).sum() / D) + (cov_b_off.pow(2).sum() / D)
        
        # --- TOTAL LOSS ---
        loss = (self.sim_coeff * sim_loss) + (self.std_coeff * std_loss) + (self.cov_coeff * cov_loss)
        
        return loss, sim_loss, std_loss, cov_loss

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_vicreg(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("--- VICReg FINE-TUNING (Variance-Invariance-Covariance) ---")

    # 1. Config & Model
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = MassFormerEncoder(full_config.get('model', {}), checkpoint_path=None).to(device)
    
    # LOAD ORIGINAL WEIGHTS
    if os.path.exists(args.original_weights_path):
        print(f"Loading weights from {args.original_weights_path}...")
        checkpoint = torch.load(args.original_weights_path, map_location=device)
        sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        try:
            model.encoder.load_state_dict(sd, strict=False)
        except RuntimeError:
            model_dict = model.encoder.state_dict()
            pretrained_dict = {k.replace("module.", "").replace("encoder.", "").replace("network.", ""): v 
                               for k, v in sd.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model.encoder.load_state_dict(pretrained_dict, strict=False)
    else:
        sys.exit(f"❌ Error: Weights file not found.")

    # 3. Dataset (Using the Augmented Loader!)
    train_dataset = AlignUniformDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    # WARNING: VICReg needs large batches for stable Variance/Covariance calculations.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=au_collate_fn, num_workers=4, drop_last=True)
    
    val_dataset = AlignUniformDataset(args.val_pairs_path, args.spec_data_path, args.mol_data_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=au_collate_fn, num_workers=4, drop_last=True)

    # 4. Setup Loss & Optimizer
    criterion = VICRegLoss(
        sim_coeff=args.sim_coeff, 
        std_coeff=args.std_coeff, 
        cov_coeff=args.cov_coeff
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 5. Training Loop
    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        model.train()
        train_loss, train_inv, train_var, train_cov = 0.0, 0.0, 0.0, 0.0
        
        for batch_graphs in tqdm(train_loader, desc="Training"):
            for k in batch_graphs: batch_graphs[k] = batch_graphs[k].to(device)
            
            optimizer.zero_grad()
            emb = model({'gf_v2_data': batch_graphs})
            
            # 🔴 CRITICAL DIFFERENCE: NO F.normalize(emb) FOR VICReg!
            
            loss, l_i, l_v, l_c = criterion(emb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_inv += l_i.item()
            train_var += l_v.item()
            train_cov += l_c.item()
            
        n = len(train_loader)
        print(f"Train | Loss: {train_loss/n:.2f} | Inv(MSE): {train_inv/n:.4f} | Var: {train_var/n:.4f} | Cov: {train_cov/n:.4f}")

        # VALIDATION
        model.eval()
        val_loss, val_inv, val_var, val_cov = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_graphs in tqdm(val_loader, desc="Validating"):
                for k in batch_graphs: batch_graphs[k] = batch_graphs[k].to(device)
                emb = model({'gf_v2_data': batch_graphs})
                
                loss, l_i, l_v, l_c = criterion(emb)
                
                val_loss += loss.item()
                val_inv += l_i.item()
                val_var += l_v.item()
                val_cov += l_c.item()
        
        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        print(f"Val   | Loss: {avg_val_loss:.2f} | Inv(MSE): {val_inv/n_val:.4f} | Var: {val_var/n_val:.4f} | Cov: {val_cov/n_val:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New Best Model! Saving...")
            torch.save({'best_model_sd': model.encoder.state_dict()}, 
                       os.path.join(args.output_dir, "best_vicreg_prev.pkl"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--original_weights_path", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, default="./vicreg_output")
    
    # VICReg specific hyperparameters (Standard defaults)
    parser.add_argument("--sim_coeff", type=float, default=25.0, help="Weight for Invariance (Similarity)")
    parser.add_argument("--std_coeff", type=float, default=25.0, help="Weight for Variance")
    parser.add_argument("--cov_coeff", type=float, default=1.0, help="Weight for Covariance")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    
    # ⚠️ IMPORTANT: Keep batch size at 32 or higher. 64 is ideal. 
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    train_vicreg(args)