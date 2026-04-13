import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import numpy as np
from pathlib import Path

# --- SETUP ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model', 'bifurcate'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src'))

try:
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
    from classifier_siamese_model import OptimalTransportSiameseModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    with open(custom_path, 'r', encoding='utf-8') as f: custom_config = yaml.safe_load(f)
    for section, subdict in custom_config.items():
        if isinstance(subdict, dict):
            if section not in config: config[section] = {}
            for k, v in subdict.items(): config[section][k] = v
        else: config[section] = subdict
    return config

def dmon_pool_loss(S, A, mask):
    S = S * mask.unsqueeze(-1)
    A = A * mask.unsqueeze(1) * mask.unsqueeze(2)
    d = torch.sum(A, dim=-1, keepdim=True) 
    m = torch.sum(A, dim=(-2, -1), keepdim=True) / 2.0 
    m = torch.clamp(m, min=1e-8)
    d_dT = torch.bmm(d, d.transpose(1, 2)) 
    B = A - (d_dT / (2.0 * m))
    S_T_B_S = torch.bmm(torch.bmm(S.transpose(1, 2), B), S)
    modularity = torch.diagonal(S_T_B_S, dim1=-2, dim2=-1).sum(dim=-1)
    loss_mod = - (modularity / (2.0 * m.squeeze(-1)))
    K = S.size(-1)
    N_active = mask.sum(dim=-1)
    cluster_sizes = S.sum(dim=1) 
    cluster_norm = torch.norm(cluster_sizes, p=2, dim=-1)
    loss_col = (np.sqrt(K) / torch.clamp(N_active, min=1.0)) * cluster_norm - 1.0
    return (loss_mod + loss_col).mean()

# =============================================================================
# EXPERT FIX: GRAPH LAPLACIAN REGULARIZATION (DIRICHLET ENERGY)
# =============================================================================
def laplacian_regularization(X, A, mask):
    """
    Computes Tr(X^T L X) to penalize connected atoms that have different embeddings.
    """
    A = A * mask.unsqueeze(1) * mask.unsqueeze(2)
    X = X * mask.unsqueeze(-1)
    
    # Calculate Graph Laplacian: L = D - A
    D_vec = torch.sum(A, dim=-1)
    D = torch.diag_embed(D_vec)
    L = D - A
    
    # Calculate Dirichlet Energy: Tr(X^T L X)
    L_X = torch.bmm(L, X)
    X_T_L_X = torch.bmm(X.transpose(1, 2), L_X)
    energy = torch.diagonal(X_T_L_X, dim1=-2, dim2=-1).sum(dim=-1)
    
    # Normalize by active nodes to prevent penalty scaling with molecule size
    N_active = mask.sum(dim=-1).clamp(min=1.0)
    return (energy / N_active).mean()
# =============================================================================

def train_phase1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    num_workers = full_config.get('run', {}).get('num_workers', 6)
    use_pin_memory = full_config.get('run', {}).get('pin_memory', True)

    train_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=True 
    )
    
    val_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=False 
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=binary_collate_fn, pin_memory=use_pin_memory, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, pin_memory=use_pin_memory, num_workers=num_workers
    )

    model = OptimalTransportSiameseModel(
        model_config=full_config['model'], 
        stage1_checkpoint=args.stage1_checkpoint,
        num_motifs=args.num_motifs
    ).to(device)
    
    bce_criterion = nn.BCEWithLogitsLoss().to(device)
    
    # Ensure GCN parameters are included
    trainable_params = (
        list(model.kinetic_mlp.parameters()) + 
        list(model.gcn1.parameters()) + 
        list(model.gcn2.parameters()) + 
        list(model.assignment_head.parameters()) + 
        list(model.head.parameters()) + 
        list(model.ot_layer.parameters())
    )
    
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    
    best_val_bce = float('inf')
    patience_counter = 0

    eps_decay_rate = 0.90
    min_epsilon = 0.01

    start_tau = 3.0
    min_tau = 0.5
    tau_decay = 0.85
    static_lambda = 0.05 

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        
        current_tau = max(min_tau, start_tau * (tau_decay ** epoch))
        current_lambda = static_lambda 
            
        print(f"\n[*] Epoch {epoch+1} Setup -> Tau: {current_tau:.2f} | Struct Lambda: {current_lambda:.2f}")
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_A, batch_B, spec_meta, mass_A, mass_B, labels, A_brics_A, A_brics_B in train_bar:
            batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
            spec_meta = spec_meta.to(device, non_blocking=True)
            mass_A = mass_A.to(device, non_blocking=True)
            mass_B = mass_B.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1) 
            A_brics_A = A_brics_A.to(device, non_blocking=True)
            A_brics_B = A_brics_B.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # EXPERT FIX: Model now accepts A_brics directly and returns X_pool and sorted A matrices
            logits, S_heavy, S_light, mask_heavy, mask_light, X_pool_heavy, X_pool_light, A_heavy, A_light = model(
                batch_A, batch_B, spec_meta, mass_A, mass_B, A_brics_A, A_brics_B, tau=current_tau
            )
            
            loss_bce = bce_criterion(logits, labels)
            
            if current_lambda > 0:
                # Modularity Loss (DMoN)
                loss_dmon_heavy = dmon_pool_loss(S_heavy, A_heavy, mask_heavy)
                loss_dmon_light = dmon_pool_loss(S_light, A_light, mask_light)
                
                # Dirichlet Energy (Laplacian)
                loss_lap_heavy = laplacian_regularization(X_pool_heavy, A_heavy, mask_heavy)
                loss_lap_light = laplacian_regularization(X_pool_light, A_light, mask_light)
                
                loss = loss_bce + current_lambda * (loss_dmon_heavy + loss_dmon_light + loss_lap_heavy + loss_lap_light)
            else:
                loss = loss_bce
                loss_dmon_heavy = loss_lap_heavy = torch.tensor(0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({
                'BCE': f"{loss_bce.item():.3f}",
                'DMoN': f"{(loss_dmon_heavy.item()):.3f}",
                'Lap': f"{(loss_lap_heavy.item()):.3f}"
            })
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0.0
        total_val_bce = 0.0
        
        with torch.no_grad():
            for batch_A, batch_B, spec_meta, mass_A, mass_B, labels, A_brics_A, A_brics_B in val_loader:
                batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
                batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
                spec_meta = spec_meta.to(device, non_blocking=True)
                mass_A = mass_A.to(device, non_blocking=True)
                mass_B = mass_B.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1) 
                A_brics_A = A_brics_A.to(device, non_blocking=True)
                A_brics_B = A_brics_B.to(device, non_blocking=True)
                
                logits, S_heavy, S_light, mask_heavy, mask_light, X_pool_heavy, X_pool_light, A_heavy, A_light = model(
                    batch_A, batch_B, spec_meta, mass_A, mass_B, A_brics_A, A_brics_B, tau=current_tau
                )
                
                loss_bce = bce_criterion(logits, labels)
                
                if current_lambda > 0:
                    loss_dmon_heavy = dmon_pool_loss(S_heavy, A_heavy, mask_heavy)
                    loss_dmon_light = dmon_pool_loss(S_light, A_light, mask_light)
                    loss_lap_heavy = laplacian_regularization(X_pool_heavy, A_heavy, mask_heavy)
                    loss_lap_light = laplacian_regularization(X_pool_light, A_light, mask_light)
                    val_loss = loss_bce + current_lambda * (loss_dmon_heavy + loss_dmon_light + loss_lap_heavy + loss_lap_light)
                else:
                    val_loss = loss_bce
                    
                total_val_loss += val_loss.item()
                total_val_bce += loss_bce.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_bce = total_val_bce / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        current_eps = model.ot_layer.epsilon
        
        print(f"Epoch {epoch+1} Results | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce:.4f}) | LR: {current_lr:.2e} | Eps: {current_eps:.3f}")

        scheduler.step()
        
        if model.ot_layer.epsilon > min_epsilon:
            model.ot_layer.epsilon = max(min_epsilon, model.ot_layer.epsilon * eps_decay_rate)

        if avg_val_bce < best_val_bce:
            best_val_bce = avg_val_bce
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "massformer_phase1_best_ot.pt")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ New best validation BCE! Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f" -> ⚠️ Validation BCE did not improve. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}.")
                break

    print("\nPhase 1 Training Complete.")
    print(f"Best Validation BCE: {best_val_bce:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage3")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_motifs", type=int, default=15)
    
    args = parser.parse_args()
    train_phase1(args)