import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- SETUP PATHS ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model'))
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

def test_phase1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    num_workers = full_config.get('run', {}).get('num_workers', 6)
    use_pin_memory = full_config.get('run', {}).get('pin_memory', True)

    print("\n[*] Loading Test Dataset...")
    test_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.test_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=False # Never upsample during testing
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, pin_memory=use_pin_memory, num_workers=num_workers
    )

    print("\n[*] Initializing Model...")
    model = OptimalTransportSiameseModel(
        model_config=full_config['model'], 
        stage1_checkpoint=None, 
        num_motifs=args.num_motifs
    ).to(device)
    
    # Load the best trained weights
    if not os.path.exists(args.trained_checkpoint):
        print(f"[-] ERROR: Trained checkpoint not found at {args.trained_checkpoint}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(args.trained_checkpoint, map_location=device), strict=False)
    print(f"[+] Loaded trained weights from: {args.trained_checkpoint}")
    
    model.eval()
    bce_criterion = nn.BCEWithLogitsLoss().to(device)
    
    all_labels = []
    all_probs = []
    all_preds = []
    total_bce_loss = 0.0

    print("\n[*] Commencing Evaluation...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating Test Set")
        for batch_A, batch_B, spec_meta, mass_A, mass_B, labels, A_brics_A, A_brics_B in test_bar:
            
            batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
            
            spec_meta = spec_meta.to(device, non_blocking=True)
            mass_A = mass_A.to(device, non_blocking=True)
            mass_B = mass_B.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1) 
            
            # REVERT FIX: Model expects 5 positional args + tau. It returns 5 items.
            logits, S_heavy, S_light, mask_heavy, mask_light = model(
                batch_A, batch_B, spec_meta, mass_A, mass_B, tau=0.5
            )
            
            loss_bce = bce_criterion(logits, labels)
            total_bce_loss += loss_bce.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels_np = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # --- METRIC CALCULATION ---
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds).flatten()
    
    avg_bce_loss = total_bce_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float('nan')

    print("\n" + "="*50)
    print("PHASE 1 SINKHORN TEST RESULTS")
    print("="*50)
    print(f"BCE Loss:  {avg_bce_loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*50)

    # --- SAVE PREDICTIONS IN TARGET FORMAT ---
    print("\n[*] Saving detailed predictions...")
    
    try:
        test_pairs_df = pd.read_feather(args.test_pairs_path)
        if len(test_pairs_df) != len(all_labels):
            print("[-] WARNING: Length mismatch between dataloader and feather file.")
            
        results_df = pd.DataFrame({
            'name_main': test_pairs_df['name_main'],
            'name_sub': test_pairs_df['name_sub'],
            # Populates cosine_similarity if it exists, otherwise pads with probs
            'cosine_similarity': test_pairs_df.get('cosine_similarity', all_probs), 
            'true_label': all_labels,
            'prob_similarity': all_probs,
            'predicted_label': all_preds
        })
        
        results_df['true_label'] = results_df['true_label'].astype(float)
        results_df['predicted_label'] = results_df['predicted_label'].astype(float)
        
        save_path = os.path.join(args.output_dir, "phase1_test_predictions.csv")
        results_df.to_csv(save_path, index=False)
        print(f"[+] Predictions saved successfully to: {save_path}")
        
    except Exception as e:
        print(f"[-] ERROR saving predictions: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs_path", type=str, required=True, help="Path to the test feather file")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    parser.add_argument("--trained_checkpoint", type=str, required=True, help="Path to massformer_phase1_best_ot.pt")
    
    parser.add_argument("--output_dir", type=str, default="results/stage3")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--num_motifs", type=int, default=15)
    
    args = parser.parse_args()
    test_phase1(args)