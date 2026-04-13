import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'flare'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from data_loader import CrossModalPretrainDataset, cross_modal_collate_fn
    from cross_modal_pretrainer import CrossModalFlarePretrainer, flare_contrastive_loss
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    sys.exit(1)

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    if os.path.exists(custom_path):
        with open(custom_path, 'r', encoding='utf-8') as f: custom_config = yaml.safe_load(f)
        for section, subdict in custom_config.items():
            if isinstance(subdict, dict):
                if section not in config: config[section] = {}
                for k, v in subdict.items(): config[section][k] = v
            else: config[section] = subdict
    return config

# =============================================================================
# EXPERT FIX: Orthogonality / Entropy Regularization Penalty
# =============================================================================
def orthogonality_penalty(S):
    """
    Forces the Assignment Matrix S to distribute atoms across all motif bins,
    preventing Mode Collapse.
    S shape: [Batch, Atoms, Motifs]
    """
    # 1. Compute S^T S (Shape: [Batch, Motifs, Motifs])
    S_t = S.transpose(1, 2)
    S_t_S = torch.bmm(S_t, S) 
    
    # 2. Normalize the matrix to prevent the loss from scaling with molecule size
    norm = torch.norm(S_t_S, p='fro', dim=(1, 2), keepdim=True)
    S_t_S_normalized = S_t_S / (norm + 1e-8)
    
    # 3. Create the Target Identity Matrix (Orthogonal ideal)
    num_motifs = S.size(-1)
    I = torch.eye(num_motifs, device=S.device).unsqueeze(0).expand_as(S_t_S)
    target = I / (num_motifs ** 0.5)
    
    # 4. Compute Frobenius distance from the ideal state
    penalty = torch.norm(S_t_S_normalized - target, p='fro', dim=(1, 2))
    return penalty.mean()

def train_flare_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # 1. Initialize Dataset
    print("\n[*] Initializing Cross-Modal Dataset...")
    full_dataset = CrossModalPretrainDataset(
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        max_peaks=args.max_peaks
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[+] Split: {train_size} Training | {val_size} Validation")
    
    num_workers = full_config.get('run', {}).get('num_workers', 6)
    use_pin_memory = full_config.get('run', {}).get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=cross_modal_collate_fn, num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=cross_modal_collate_fn, num_workers=num_workers, pin_memory=use_pin_memory
    )

    # 2. Initialize Model
    print("\n[*] Initializing Cross-Modal FLARE Pretrainer...")
    model = CrossModalFlarePretrainer(
        model_config=full_config['model'], 
        num_motifs=args.num_motifs
    ).to(device)
    
    # 3. Setup Optimizer
    trainable_params = (
        list(model.kinetic_mlp.parameters()) + 
        list(model.gcn1.parameters()) + 
        list(model.gcn2.parameters()) + 
        list(model.assignment_head.parameters()) + 
        list(model.spectral_encoder.parameters()) +
        [model.logit_scale]
    )
    
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # --- TRAINING LOOP ---
    print("\n[*] Commencing Stage 1 Pre-training with Orthogonality Penalty...")
    for epoch in range(args.epochs):
        model.train()
        total_flare_loss = 0.0
        total_ortho_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batched_graphs, A_brics, peaks, peak_mask in train_bar:
            
            if isinstance(batched_graphs, dict):
                batched_graphs = {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batched_graphs.items()}
            elif hasattr(batched_graphs, 'to'):
                batched_graphs = batched_graphs.to(device)
                
            A_brics = A_brics.to(device, non_blocking=True) if A_brics is not None else None
            peaks = peaks.to(device, non_blocking=True)
            peak_mask = peak_mask.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            current_tau = max(1.0 * (0.9 ** epoch), 0.5)
            Z_graph, Z_spec, S, X_pool = model(batched_graphs, A_brics, peaks, peak_mask, tau=current_tau)
            
            # --- CALCULATE COMBINED LOSS ---
            loss_flare = flare_contrastive_loss(Z_graph, Z_spec, peak_mask, model.logit_scale)
            loss_ortho = orthogonality_penalty(S)
            
            loss = loss_flare + (args.ortho_weight * loss_ortho)
            
            # Safety checks
            if torch.isnan(loss):
                print("\n[!] CRITICAL: NaN Loss Detected!")
                sys.exit(1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            
            total_flare_loss += loss_flare.item()
            total_ortho_loss += loss_ortho.item()
            
            train_bar.set_postfix({
                'FLARE': f"{loss_flare.item():.3f}", 
                'Ortho': f"{loss_ortho.item():.3f}", 
                'Tau': f"{current_tau:.2f}"
            })
            
        avg_flare_loss = total_flare_loss / len(train_loader)
        avg_ortho_loss = total_ortho_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for batched_graphs, A_brics, peaks, peak_mask in val_bar:
                if isinstance(batched_graphs, dict):
                    batched_graphs = {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batched_graphs.items()}
                elif hasattr(batched_graphs, 'to'):
                    batched_graphs = batched_graphs.to(device)
                    
                A_brics = A_brics.to(device, non_blocking=True) if A_brics is not None else None
                peaks = peaks.to(device, non_blocking=True)
                peak_mask = peak_mask.to(device, non_blocking=True)
                
                Z_graph, Z_spec, S, X_pool = model(batched_graphs, A_brics, peaks, peak_mask, tau=current_tau)
                
                v_loss_flare = flare_contrastive_loss(Z_graph, Z_spec, peak_mask, model.logit_scale)
                v_loss_ortho = orthogonality_penalty(S)
                
                v_loss = v_loss_flare + (args.ortho_weight * v_loss_ortho)
                total_val_loss += v_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train [FLARE: {avg_flare_loss:.3f}, Ortho: {avg_ortho_loss:.3f}] | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        scheduler.step()
        
        # --- EARLY STOPPING & SAVING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "massformer_flare_stage1_ortho_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ New best Val Loss! Motif weights saved.")
        else:
            patience_counter += 1
            print(f" -> ⚠️ Validation Loss did not improve. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trained_model/cross_modal_pretrain")
    
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    
    # Architecture & Penalties
    parser.add_argument("--max_peaks", type=int, default=60)
    parser.add_argument("--num_motifs", type=int, default=15)
    parser.add_argument("--ortho_weight", type=float, default=0.5, help="Weight of the assignment orthogonality penalty")
    
    args = parser.parse_args()
    train_flare_stage1(args)