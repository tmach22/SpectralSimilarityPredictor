import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import wandb
import re
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'flare'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from data_loader import CrossModalPretrainDataset, cross_modal_collate_fn
    # Use the correctly named updated loss function
    from cross_modal_pretrainer_new import BifurcatedCrossModalPretrainer, bifurcated_contrastive_loss
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

def setup_differential_fine_tuning(model, verbose=True):
    """Regex-based Smart Unfreeze for the MassFormer Backbone."""
    if verbose: print("\n--- Configuring Encoder Fine-Tuning (Smart Regex Unfreeze) ---")
    
    for param in model.graph_encoder.parameters():
        param.requires_grad = False
        
    trainable_encoder_params = []
    layer_indices = []
    
    for name, _ in model.graph_encoder.named_parameters():
        match = re.search(r'(?:layer|layers)\.(\d+)', name)
        if match: layer_indices.append(int(match.group(1)))
            
    if not layer_indices:
        print(" -> WARNING: Regex failed. Unfreezing FULL encoder.")
        for param in model.graph_encoder.parameters():
            param.requires_grad = True
            trainable_encoder_params.append(param)
        return trainable_encoder_params
        
    max_layer = max(layer_indices)
    target_layers = [max_layer - 1, max_layer]
    
    if verbose:
        print(f" -> Detected max layer index: {max_layer}")
        print(f" -> Unfreezing layer indices {target_layers[0]} and {target_layers[1]}...")
        
    unfrozen_count = 0
    for name, param in model.graph_encoder.named_parameters():
        is_top_layer = any(f"layer.{idx}." in name or f"layers.{idx}." in name for idx in target_layers)
        is_output_head = any(key in name for key in ['out_proj', 'final_proj', 'output_layer'])
        
        if is_top_layer or is_output_head:
            param.requires_grad = True
            trainable_encoder_params.append(param)
            unfrozen_count += 1
            
    if verbose: print(f" -> Successfully unfrozen {unfrozen_count} backbone parameters.")
    return trainable_encoder_params

def train_bifurcated_flare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Initialize wandb securely
    wandb.init(project="flare-bifurcated", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[*] Loading Model Configurations...")
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print("\n[*] Initializing Dataset...")
    full_dataset = CrossModalPretrainDataset(
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        max_peaks=args.max_peaks
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=cross_modal_collate_fn, num_workers=6, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=cross_modal_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing BIFURCATED FLARE Pretrainer...")
    model = BifurcatedCrossModalPretrainer(model_config=full_config['model'], num_motifs=args.num_motifs).to(device)
    
    # FIXED WANDB ERROR:
    # Removed `wandb.watch(model, log="all", log_freq=100)`. 
    # wandb's hook traces PyTorch graphs directly and fundamentally crashes when models process 
    # complex non-tensor structures like dict inputs `{'gf_v2_data': batched_data}`.

    encoder_params = setup_differential_fine_tuning(model)
    head_params = [p for n, p in model.named_parameters() if "graph_encoder" not in n]
    
    print(f" -> Router/Spectral Head parameters fully unfrozen: {len(head_params)}")

    optimizer = AdamW([
        # The new GNN Router and Spectral Head get the fast learning rate
        {'params': head_params, 'lr': args.learning_rate, 'weight_decay': 1e-4},
        # The Backbone gets the slow learning rate
        {'params': encoder_params, 'lr': args.learning_rate * 0.1, 'weight_decay': 1e-4}
    ])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n[*] Commencing Bifurcated Training Loop...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in train_bar:
            if batch[0] is None: continue

            batched_graphs, A_brics, _, peaks, peak_mask, _ = batch

            if isinstance(batched_graphs, dict):
                batched_graphs = {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batched_graphs.items()}
            elif hasattr(batched_graphs, 'to'):
                batched_graphs = batched_graphs.to(device)

            A_brics = A_brics.to(device, non_blocking=True)
            peaks = peaks.to(device, non_blocking=True)
            peak_mask = peak_mask.to(device, non_blocking=True)

            # Forward pass
            Z_graph, Z_spec, S = model(batched_graphs, A_brics, peaks, peak_mask)

            # Updated loss function usage
            loss = bifurcated_contrastive_loss(Z_graph, Z_spec, peak_mask, model.logit_scale)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch[0] is None: continue
                batched_graphs, A_brics, _, peaks, peak_mask, _ = batch

                if isinstance(batched_graphs, dict):
                    batched_graphs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batched_graphs.items()}
                elif hasattr(batched_graphs, 'to'): 
                    batched_graphs = batched_graphs.to(device)
                    
                A_brics = A_brics.to(device)
                peaks = peaks.to(device)
                peak_mask = peak_mask.to(device)

                Z_graph, Z_spec, S = model(batched_graphs, A_brics, peaks, peak_mask)
                
                # Updated loss function usage
                v_loss = bifurcated_contrastive_loss(Z_graph, Z_spec, peak_mask, model.logit_scale)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        
        wandb.log({"epoch": epoch, "val/loss": avg_val_loss, "train/loss": avg_train_loss})
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "bifurcated_flare_best.pt"))
            print(f" -> Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}.")
                break

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="trained_model/flare_bifurcated")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_peaks", type=int, default=60)
    parser.add_argument("--num_motifs", type=int, default=15)
    args = parser.parse_args()
    train_bifurcated_flare(args)