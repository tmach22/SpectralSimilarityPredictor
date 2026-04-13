import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup # <-- NEW EXPERT SCHEDULER
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path

# --- SETUP ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Import the 6-item dataloader and the new Phase 3 OT model
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
from peft_model import OptimalTransportSiameseModel

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(custom_path, 'r', encoding='utf-8') as f:
        custom_config = yaml.safe_load(f)
        
    for section, subdict in custom_config.items():
        if isinstance(subdict, dict):
            if section not in config:
                config[section] = {}
            for k, v in subdict.items():
                config[section][k] = v
        else:
            config[section] = subdict
    return config

def train_stage3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    num_workers = full_config.get('run', {}).get('num_workers', 6)
    use_pin_memory = full_config.get('run', {}).get('pin_memory', True)

    print("Loading Training Dataset...")
    train_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=False 
    )
    
    print("Loading Validation Dataset...")
    val_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=False 
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
        stage1_checkpoint=args.stage1_checkpoint
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    # --- EXPERT FIX: DIFFERENTIAL LEARNING RATES ---
    # 1. LoRA Parameters (Microscopic LR to prevent geometric collapse)
    lora_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    
    # 2. Latent Adapter (Low LR to slowly warp the PCQM4Mv2 space)
    adapter_params = [p for n, p in model.latent_adapter.named_parameters() if p.requires_grad]
    
    # 3. Physics Engine & MLP Head (Standard LR to drive the logic)
    ot_and_head_params = list(model.head.parameters()) + list(model.ot_layer.parameters())
    
    optimizer = AdamW([
        {'params': lora_params, 'lr': 1e-5},             
        {'params': adapter_params, 'lr': 1e-4},          
        {'params': ot_and_head_params, 'lr': args.learning_rate} 
    ], weight_decay=1e-4)
    
    total_trainable = sum(p.numel() for p in lora_params + adapter_params + ot_and_head_params)
    print(f"\nOptimizer loaded with {total_trainable} trainable parameters across 3 LR groups.")

    # --- EXPERT FIX: WARMUP + COSINE SCHEDULER ---
    # Graphormers require slow warmup to avoid destroying pre-trained representations early on
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps) # 10% of total training is just warming up
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # --- SINKHORN EPSILON ANNEALING PARAMS ---
    eps_decay_rate = 0.90 
    min_epsilon = 0.01    

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_A, batch_B, spec_meta, mass_A, mass_B, labels in train_bar:
            batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
            
            spec_meta = spec_meta.to(device, non_blocking=True)
            mass_A = mass_A.to(device, non_blocking=True)
            mass_B = mass_B.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1) 
            
            optimizer.zero_grad()
            
            logits = model(batch_A, batch_B, spec_meta, mass_A, mass_B)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # --- EXPERT FIX: GRADIENT CLIPPING ---
            # Protects the network from the RNN-like backprop through the Sinkhorn iterations
            all_params = lora_params + adapter_params + ot_and_head_params
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step() # Note: The scheduler now steps per batch, not per epoch!
            
            total_train_loss += loss.item()
            train_bar.set_postfix({
                'loss': loss.item(),
                'dustbin': model.ot_layer.dustbin_penalty.item(),
                'lr': optimizer.param_groups[2]['lr'] # Track the main head LR
            })
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_A, batch_B, spec_meta, mass_A, mass_B, labels in val_loader:
                batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
                batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
                
                spec_meta = spec_meta.to(device, non_blocking=True)
                mass_A = mass_A.to(device, non_blocking=True)
                mass_B = mass_B.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1) 
                
                logits = model(batch_A, batch_B, spec_meta, mass_A, mass_B)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[2]['lr']
        current_eps = model.ot_layer.epsilon
        current_dustbin = model.ot_layer.dustbin_penalty.item()
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} | Eps: {current_eps:.3f} | Dustbin: {current_dustbin:.4f}")
        
        # Anneal the Sinkhorn epsilon per epoch
        if model.ot_layer.epsilon > min_epsilon:
            model.ot_layer.epsilon = max(min_epsilon, model.ot_layer.epsilon * eps_decay_rate)

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            save_path = os.path.join(args.output_dir, "massformer_stage3_best_ot.pt")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ New best validation loss! Model saved.")
        else:
            patience_counter += 1
            print(f" -> ⚠️ Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}.")
                break

    print("\nPhase 3 Training Complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

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
    parser.add_argument("--patience", type=int, default=8) 
    
    args = parser.parse_args()
    train_stage3(args)