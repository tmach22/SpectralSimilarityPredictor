import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path

# --- SETUP ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'wide_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'wide_model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from finetuning_dataloader import StageOneDataset, stage1_collate_fn
from decon_encoder import StageOneMetricModel, OrderEmbeddingLoss

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    """Loads and merges the base template config with the specific run config."""
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(custom_path, 'r') as f:
        custom_config = yaml.safe_load(f)
        
    # Standard nested dictionary merge
    for section, subdict in custom_config.items():
        if isinstance(subdict, dict):
            if section not in config:
                config[section] = {}
            for k, v in subdict.items():
                config[section][k] = v
        else:
            config[section] = subdict
            
    return config

def verify_layer_resets(model, checkpoint_path):
    print("\n" + "="*50)
    print("DIAGNOSTIC: VERIFYING TRANSFORMER LAYER RESETS")
    print("="*50)
    
    # 1. Load the raw weights directly from the file
    raw_chkpt = torch.load(checkpoint_path, map_location="cpu")
    raw_sd = raw_chkpt.get('best_model_sd', raw_chkpt)
    
    # 2. Get the active model's weights
    current_sd = model.state_dict()
    
    try:
        # 3. Hunt down the specific tensor keys
        raw_atom_key = next(k for k in raw_sd.keys() if 'graph_node_feature.atom_encoder.weight' in k)
        curr_atom_key = next(k for k in current_sd.keys() if 'graph_node_feature.atom_encoder.weight' in k)
        
        raw_attn_key = next(k for k in raw_sd.keys() if 'layers.0.self_attn.q_proj.weight' in k)
        curr_attn_key = next(k for k in current_sd.keys() if 'layers.0.self_attn.q_proj.weight' in k)
        
        # 4. Perform the mathematical comparison
        atom_match = torch.equal(raw_sd[raw_atom_key].cpu(), current_sd[curr_atom_key].cpu())
        attn_match = torch.equal(raw_sd[raw_attn_key].cpu(), current_sd[curr_attn_key].cpu())
        
        print(f"1. Atom Embeddings Preserved?  {'✅ YES (Identical to checkpoint)' if atom_match else '❌ NO (Weights changed!)'}")
        print(f"2. Transformer Layers Reset?   {'✅ YES (Randomly re-initialized)' if not attn_match else '❌ NO (Still using pre-trained!)'}")
        
    except StopIteration:
        print("⚠️ Could not locate the exact tensor keys in the dictionaries to compare.")
        
    print("="*50 + "\n")

def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving checkpoints to: {args.output_dir}")

    # 1. Load configuration files
    print(f"Merging configs: {args.template_config} + {args.custom_config}")
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # 2. Init Data
    print("Loading Training Dataset...")
    train_dataset = StageOneDataset(
        pairs_feather_path=args.train_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=True
    )
    
    print("Loading Validation Dataset...")
    val_dataset = StageOneDataset(
        pairs_feather_path=args.val_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=False  # Do not skew the validation distribution
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=stage1_collate_fn, pin_memory=True, num_workers=8
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=stage1_collate_fn, pin_memory=True, num_workers=8
    )

    # 3. Init Model & Loss
    model = StageOneMetricModel(
        model_config=full_config['model'], 
        checkpoint_path=args.checkpoint_path
    ).to(device)

    # Diagnostic check
    verify_layer_resets(model, args.checkpoint_path)
    
    criterion = OrderEmbeddingLoss(margin=1.0).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # --- EARLY STOPPING TRACKERS ---
    best_val_loss = float('inf')
    patience_counter = 0

    # 4. Training Loop
    for epoch in range(args.epochs):
        # --- TRAIN PHASE ---
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_A, batch_B, mass_A, mass_B, labels in train_bar:
            # Move graph dictionaries and tensors to GPU asynchronously
            batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
            
            mass_A = mass_A.to(device, non_blocking=True)
            mass_B = mass_B.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            emb_A, emb_B = model(batch_A, batch_B)
            loss = criterion(emb_A, emb_B, mass_A, mass_B, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch_A, batch_B, mass_A, mass_B, labels in val_bar:
                batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
                batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
                
                mass_A = mass_A.to(device, non_blocking=True)
                mass_B = mass_B.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                emb_A, emb_B = model(batch_A, batch_B)
                val_loss = criterion(emb_A, emb_B, mass_A, mass_B, labels)
                
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
            save_path = os.path.join(args.output_dir, "massformer_stage1_best_metric.pt")
            torch.save(model.state_dict(), save_path)
            print(f" -> ⭐ New best validation loss! Geometry saved to {save_path}")
        else:
            patience_counter += 1
            print(f" -> ⚠️ Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}. No improvement for {args.patience} epochs.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Changed from --pairs_path to separate train/val paths
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Original Massformer weights (.pt)")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage1", help="Directory to save fine-tuned weights")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--patience", type=int, default=4, help="Epochs to wait before early stopping")
    
    args = parser.parse_args()
    train_stage1(args)