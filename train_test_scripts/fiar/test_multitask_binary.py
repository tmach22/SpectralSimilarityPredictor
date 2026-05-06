import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    from phase3_multitask_dataloader import PairedSiameseDataset, siamese_collate_fn
    from phase3_multitask_siamese import Phase3_LinearProbe_SiameseNetwork
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

def test_linear_probe_classifier(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Native config loading
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print(f"\n[*] Initializing Test Dataset from {args.test_pairs}...")
    test_dataset = PairedSiameseDataset(feather_path=args.test_pairs, graphs_path=args.graphs_path)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, 
        shuffle=False, collate_fn=siamese_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing Phase 3 Strict Linear Probe Siamese Network...")
    model = Phase3_LinearProbe_SiameseNetwork(full_config, args.phase2_checkpoint, device, max_fragments=10).to(device)
    
    print(f"[*] Loading Trained Linear Probe Weights from: {args.model_ckpt}")
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.model_ckpt, map_location=device), strict=False)
    
    # =================================================================
    # [THE FIX] Final Weight Diagnostics (Everything should match perfectly)
    # =================================================================
    print("\n--- Final Weight Loading Diagnostics ---")
    print(f"Missing Keys (Should be empty): \n{missing_keys}")
    print(f"Unexpected Keys (Should be empty): \n{unexpected_keys}")
    print("----------------------------------------\n")

    if "binary_head.weight" in missing_keys:
        print("\n[!] FATAL ERROR: The 'binary_head' is missing from your checkpoint!")
        print("    -> You likely passed the Phase 2.5 weights instead of the final Phase 3 weights.")
        sys.exit(1)
        
    model.eval()

    all_continuous_preds = []
    all_binary_probs = []
    
    print("\n[*] Commencing Evaluation...")
    test_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch_idx, (batch_A, batch_B, targets_sim, targets_label) in enumerate(test_bar):
            if batch_A is None or batch_B is None: continue
            
            batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}

            # Unpack dual outputs
            continuous_pred, binary_logit = model(batch_A, batch_B, tau=0.01)
            
            # Convert raw focal logits into probabilities (0.0 to 1.0)
            binary_prob = torch.sigmoid(binary_logit)
            
            all_continuous_preds.extend(continuous_pred.cpu().numpy().flatten())
            all_binary_probs.extend(binary_prob.cpu().numpy().flatten())

    continuous_similarity = np.array(all_continuous_preds)
    prob_similarity = np.array(all_binary_probs)
    
    # Standard probability threshold of 0.5 since the logit was centered
    predicted_label = (prob_similarity >= 0.5).astype(float)

    print(f"\n[*] Loading original pairs data to map True Labels...")
    if str(args.test_pairs).endswith('.feather'):
        pairs_df = pd.read_feather(args.test_pairs)
    else:
        pairs_df = pd.read_csv(args.test_pairs)

    if 'label' in pairs_df.columns:
        true_label = pairs_df['label'].values.astype(float)
    else:
        raise KeyError("Could not find 'label' in the dataset.")

    # Calculate Classification Metrics
    acc = accuracy_score(true_label, predicted_label)
    prec = precision_score(true_label, predicted_label, zero_division=0)
    rec = recall_score(true_label, predicted_label, zero_division=0)
    f1 = f1_score(true_label, predicted_label, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(true_label, prob_similarity)
    except ValueError:
        roc_auc = float('nan') 

    print("\n" + "="*45)
    print(" 🏆 THERMODYNAMIC LINEAR PROBE METRICS 🏆")
    print("="*45)
    print(f" Accuracy        : {acc:.4f}")
    print(f" Precision       : {prec:.4f}")
    print(f" Recall          : {rec:.4f}")
    print(f" F1-Score        : {f1:.4f}")
    print(f" ROC-AUC Score   : {roc_auc:.4f}")
    print("="*45)

    print(f"[*] Compiling requested CSV format...")
    
    final_df = pd.DataFrame({
        'name_main': pairs_df['name_main'],
        'name_sub': pairs_df['name_sub'],
        'cosine_similarity': pairs_df.get('cosine_similarity', np.nan),
        'true_label': true_label,
        'prob_similarity': prob_similarity,
        'continuous_pred': continuous_similarity, 
        'predicted_label': predicted_label
    })
    
    # Check if testing NIST to change the output file name dynamically
    if "nist" in args.test_pairs.lower():
        results_path = os.path.join(args.output_dir, "test_thermodynamic_probe_predictions_nist.csv")
    else:
        results_path = os.path.join(args.output_dir, "test_thermodynamic_probe_predictions.csv")
        
    final_df.to_csv(results_path, index=False)
    print(f"[+] Formatted predictions successfully saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs", type=str, required=True)
    parser.add_argument("--graphs_path", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/binary_evaluation")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    test_linear_probe_classifier(args)