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
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'fiar')) 
sys.path.insert(0, os.path.join(cwd, 'model', 'fiar'))
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model', 'bifurcate'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src'))
sys.path.insert(0, os.path.join(str(cwd), 'model', 'flare'))

try:
    from phase2_dataloader import Phase2EdgeDataset, phase2_collate_fn
    from phase2_model_new import DESAFNet, calculate_cosine_reward
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

def train_phase2_reinforce(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    wandb.init(project="desaf-phase2-reinforce", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[*] Loading Model Configurations...")
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print("\n[*] Initializing Phase 2 Dataset...")
    full_dataset = Phase2EdgeDataset(processed_graphs_path=args.phase2_data_path)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=phase2_collate_fn, num_workers=6, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=phase2_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing DESAF-Net (Phase 2 Mode)...")
    model = DESAFNet(model_config=full_config['model']).to(device)
    
    # CRITICAL: Load the Phase 1 Anchor Weights
    if not os.path.exists(args.phase1_checkpoint):
        raise FileNotFoundError(f"[-] Phase 1 Checkpoint not found at {args.phase1_checkpoint}. Phase 1 is mandatory for DESAF-Net.")
    
    model.load_state_dict(torch.load(args.phase1_checkpoint, map_location=device), strict=False)
    print(f"[+] Phase 1 Anchor weights successfully loaded from {args.phase1_checkpoint}.")
    
    model.freeze_backbone()

    optimizer = AdamW(model.edge_head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_reward = 0.0
    patience_counter = 0

    print(f"\n[*] Commencing Phase 2 REINFORCE Training Loop...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_reward = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(train_bar):
            if batch is None: continue
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            
            # Forward Pass (Sampling)
            theoretical_mzs, log_probs, fragment_batch_idx = model.forward_phase2(batch)
            
            # Calculate Reward against empirical spectra
            rewards = calculate_cosine_reward(
                theoretical_mzs, fragment_batch_idx, batch['peaks'], batch['peak_mask']
            )
            
            # REINFORCE Math: Advantage Baseline Subtraction
            baseline = rewards.mean()
            advantage = rewards - baseline
            
            # Broadcast graph advantage to its edges
            edge_advantages = []
            current_edge_idx = 0
            for b in range(batch['x'].shape[0]):
                true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
                edges_in_graph = ((batch['edge_index'][0] >= current_edge_idx) & 
                                  (batch['edge_index'][0] < current_edge_idx + true_nodes)).sum().item()
                
                edge_advantages.append(torch.full((edges_in_graph,), advantage[b].item(), device=device))
                current_edge_idx += true_nodes
                
            edge_advantages = torch.cat(edge_advantages, dim=0)
            
            # Calculate Policy Gradient Loss
            loss = -(edge_advantages * log_probs).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.edge_head.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================================
            # CONTINUOUS RL TELEMETRY 
            # ==========================================
            total_molecules = batch['x'].shape[0]
            total_fragments_generated = len(theoretical_mzs)
            avg_frags_per_mol = total_fragments_generated / max(1, total_molecules)
            
            if wandb.run is not None:
                wandb.log({
                    "batch/pg_loss": loss.item(),
                    "batch/mean_reward": baseline.item(),
                    "batch/avg_fragments_per_mol": avg_frags_per_mol
                }, commit=False)
            
            if batch_idx % 100 == 0:
                print("\n" + "="*60)
                print(f"🚀 RL TELEMETRY | Epoch {epoch} | Batch {batch_idx}")
                print("-" * 60)
                print(f"Batch Mean Reward:      {baseline.item():.4f}")
                print(f"Policy Gradient Loss:   {loss.item():.4f}")
                print(f"Avg Fragments/Molecule: {avg_frags_per_mol:.2f} (Tracking Shatter Rate)")
                print("-" * 60)
                print(f"Sample Deep Dive: Molecule 0")
                
                mol_0_frags = theoretical_mzs[fragment_batch_idx == 0]
                print(f"-> Theoretical Fragments (m/z):")
                print(f"   {mol_0_frags.detach().cpu().numpy().round(2)}")
                
                valid_exp = batch['peaks'][0, ~batch['peak_mask'][0]]
                if len(valid_exp) > 0:
                    exp_mzs_real = valid_exp[:, 0] * 1000.0 
                    print(f"-> Actual Spectrum (m/z):")
                    print(f"   {exp_mzs_real.detach().cpu().numpy().round(2)}")
                else:
                    print(f"-> Actual Spectrum: [Empty/Filtered]")
                
                print(f"-> Local Reward for Mol 0: {rewards[0].item():.4f}")
                print("="*60 + "\n")
            # ==========================================

            total_train_loss += loss.item()
            total_train_reward += baseline.item()
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Reward': f"{baseline.item():.4f}"})

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_train_reward = total_train_reward / max(len(train_loader), 1)

        # --- Validation Pass ---
        model.eval()
        total_val_reward = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

                theoretical_mzs, _, fragment_batch_idx, _ = model.forward_phase2(batch)
                
                v_rewards = calculate_cosine_reward(
                    theoretical_mzs, fragment_batch_idx, batch['peaks'], batch['peak_mask']
                )
                total_val_reward += v_rewards.mean().item()

        avg_val_reward = total_val_reward / max(len(val_loader), 1)
        
        wandb.log({
            "epoch": epoch, 
            "train/pg_loss": avg_train_loss, 
            "train/cosine_reward": avg_train_reward,
            "val/cosine_reward": avg_val_reward
        })
        
        print(f"Epoch {epoch} | PG Loss: {avg_train_loss:.4f} | Train Reward: {avg_train_reward:.4f} | Val Reward: {avg_val_reward:.4f}")

        scheduler.step()

        # Early Stopping & Checkpointing (Maximizing Reward)
        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "desaf_phase2_best.pt"))
            print(f" -> Best Phase 2 model saved! (Reward: {best_val_reward:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}.")
                break

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_data_path", type=str, required=True, help="Path to graphs containing spectral peaks")
    parser.add_argument("--phase1_checkpoint", type=str, required=True, help="Path to desaf_phase1_best.pt (MANDATORY)")
    parser.add_argument("--output_dir", type=str, default="trained_model/desaf_phase2")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100) 
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    args = parser.parse_args()
    
    train_phase2_reinforce(args)