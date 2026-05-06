import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import wandb
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*nested tensors.*")

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
    from phase2_model_new import DESAFNet  # The RL Teacher
    from deterministic_phase2 import DeterministicPhase2  # The New Student
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

def extract_teacher_probs(teacher, batch, device):
    """
    Manually extracts the continuous cutting probabilities from the frozen RL Teacher
    without triggering the physical Bernoulli samplers or Scipy clustering.
    """
    B = batch['x'].shape[0]
    
    X_base = teacher.graph_encoder({'gf_v2_data': batch})
    X = X_base[0][:, 1:, :] if isinstance(X_base, tuple) else X_base[:, 1:, :]
    
    valid_X_list = []
    for b in range(B):
        true_nodes = (batch['x'][b, :, 0] != 0).sum().item()
        valid_X_list.append(X[b, :true_nodes, :])
        
    X_flat = torch.cat(valid_X_list, dim=0)
    
    # Predict Cut Probabilities
    break_logits = teacher.edge_head(X_flat, batch['edge_index'], batch['edge_attr_physics'])
    p_cut = torch.sigmoid(break_logits)
    
    # The Teacher predicts CUT. The Student predicts KEEP.
    p_keep_target = 1.0 - p_cut
    
    return p_keep_target.detach()

def train_dagger(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        print(f"[*] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")

    wandb.init(project="phase2-dagger-distillation", config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print("\n[*] Initializing Flat Phase 2 Dataset...")
    train_dataset = Phase2EdgeDataset(processed_graphs_path=args.graphs_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=phase2_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing RL Teacher (Frozen)...")
    teacher = DESAFNet(full_config['model']).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("\n[*] Initializing Deterministic Student (Trainable)...")
    student = DeterministicPhase2(full_config['model'], max_fragments=10).to(device)
    student.train()
    
    # [CRITICAL FIX] We now optimize ALL parameters in the Student model.
    # The GCN and Assignment MLP will receive their gradients directly from L_topo.
    trainable_params = list(student.parameters())
    
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Use Binary Cross Entropy for the edge distillation
    criterion = nn.BCELoss()

    print("\n[*] Commencing DAgger Distillation Loop with Topological Loss...")
    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0.0
        total_bce_loss = 0.0
        total_topo_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Distill]")
        
        for batch_idx, batch in enumerate(train_bar):
            if batch is None: continue
            
            # Send to GPU
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            
            # 1. Oracle Query (Get Teacher Targets)
            with torch.no_grad():
                teacher_p_keep = extract_teacher_probs(teacher, batch, device)
            
            # 2. Student Rollout (Extracting the new L_topo)
            _, student_p_keep, _, L_topo = student(batch)
            
            # 3. Soft-Target KD Loss
            loss_distill = criterion(student_p_keep, teacher_p_keep)
            
            # 4. Compute Dual Loss and Backpropagate
            # We equally weight the physics KD and the topological assignment
            total_loss = loss_distill + L_topo
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Metric tracking
            total_train_loss += total_loss.item()
            total_bce_loss += loss_distill.item()
            total_topo_loss += L_topo.item()
            
            train_bar.set_postfix({
                'Distill BCE': f"{loss_distill.item():.4f}",
                'Topo BCE': f"{L_topo.item():.4f}"
            })

            if batch_idx % 50 == 0 and wandb.run is not None:
                wandb.log({
                    "batch/total_loss": total_loss.item(),
                    "batch/distill_bce": loss_distill.item(),
                    "batch/topo_bce": L_topo.item()
                }, commit=False)

        avg_loss = total_train_loss / max(len(train_loader), 1)
        avg_bce = total_bce_loss / max(len(train_loader), 1)
        avg_topo = total_topo_loss / max(len(train_loader), 1)
        
        wandb.log({
            "epoch": epoch, 
            "train/total_loss": avg_loss, 
            "train/distill_bce": avg_bce,
            "train/topo_bce": avg_topo,
            "lr": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch} | Total: {avg_loss:.4f} | Distill: {avg_bce:.4f} | Topo: {avg_topo:.4f}")
        
        scheduler.step()

        ckpt_path = os.path.join(args.output_dir, f"deterministic_student_ep{epoch}.pt")
        torch.save(student.state_dict(), ckpt_path)
        torch.save(student.state_dict(), os.path.join(args.output_dir, "deterministic_student_best.pt"))

    wandb.finish()
    print(f"\n[+] Distillation Complete. Student weights saved to {args.output_dir}/deterministic_student_best.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, required=True, help="Path to phase3_graphs.pt")
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to the RL DESAFNet checkpoint")
    parser.add_argument("--output_dir", type=str, default="trained_model/phase2_distilled")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--epochs", type=int, default=5) 
    parser.add_argument("--learning_rate", type=float, default=1e-4) 
    
    args = parser.parse_args()
    train_dagger(args)