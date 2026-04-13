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

# Import your newly created DataLoader and Model components
try:
    from data_loader import CrossModalPretrainDataset, cross_modal_collate_fn
    from cross_modal_pretrainer import CrossModalFlarePretrainer, flare_contrastive_loss
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    print("Ensure 'spectral_dataset.py' is in 'data_loaders/' and 'cross_modal_pretrainer.py' is in 'model/'")
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
    
    # Train/Val Split (80/20)
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
    print("\n[*] Commencing Stage 1 Pre-training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batched_graphs, A_brics, peaks, peak_mask in train_bar:
            
            # Move data to GPU
            # Added a safety check for dictionary collation
            if isinstance(batched_graphs, dict):
                batched_graphs = {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batched_graphs.items()}
            elif hasattr(batched_graphs, 'to'):
                batched_graphs = batched_graphs.to(device)
                
            A_brics = A_brics.to(device, non_blocking=True) if A_brics is not None else None
            peaks = peaks.to(device, non_blocking=True)
            peak_mask = peak_mask.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward Pass
            current_tau = max(1.0 * (0.9 ** epoch), 0.5)
            Z_graph, Z_spec, S, X_pool = model(batched_graphs, A_brics, peaks, peak_mask, tau=current_tau)
            loss = flare_contrastive_loss(Z_graph, Z_spec, peak_mask, model.logit_scale)
            
            # ====================================================================
            # DEBUGGING TRIPWIRE 1: Check Forward Pass outputs for NaNs
            # ====================================================================
            if torch.isnan(loss):
                print("\n[!] CRITICAL: NaN Loss Detected!")
                print(f" -> Are raw peaks NaN? {torch.isnan(peaks).any().item()}")
                print(f" -> Is A_brics NaN? {torch.isnan(A_brics).any().item() if A_brics is not None else 'None'}")
                print(f" -> Is Z_graph NaN? {torch.isnan(Z_graph).any().item()}")
                print(f" -> Is Z_spec NaN? {torch.isnan(Z_spec).any().item()}")
                print(f" -> Is S (assignments) NaN? {torch.isnan(S).any().item()}")
                print(f" -> Logit Scale (Temperature): {model.logit_scale.exp().item():.4f}")
                sys.exit(1)
            # ====================================================================
            
            # Backprop
            loss.backward()
            
            # ====================================================================
            # DEBUGGING TRIPWIRE 2: Check Gradients for Explosions before Step
            # ====================================================================
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"\n[!] CRITICAL: NaN Gradient detected in layer: {name}")
                        sys.exit(1)
            # ====================================================================
            
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'FLARE_Loss': f"{loss.item():.4f}", 'Tau': f"{current_tau:.2f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
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
                loss = flare_contrastive_loss(Z_graph, Z_spec, peak_mask, model.logit_scale)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Results | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Tau: {current_tau:.2f}")
        
        scheduler.step()
        
        # --- EARLY STOPPING & SAVING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "massformer_flare_stage1_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ New best Val Loss! Motif weights saved to {save_path}")
        else:
            patience_counter += 1
            print(f" -> ⚠️ Validation Loss did not improve. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}.")
            break

    print(f"\nStage 1 Pre-training Complete. Best Validation FLARE Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Paths
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to spec_df.pkl")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to mol_df.pkl")
    parser.add_argument("--output_dir", type=str, default="trained_model/cross_modal_pretrain", help="Directory to save checkpoint")
    
    # Config Paths
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    
    # Architecture Dimensions
    parser.add_argument("--max_peaks", type=int, default=60, help="Matches FLARE peak filtering limit")
    parser.add_argument("--num_motifs", type=int, default=15, help="Number of graph motifs/bins")
    
    args = parser.parse_args()
    train_flare_stage1(args)