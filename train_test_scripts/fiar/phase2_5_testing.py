import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
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
    from phase2_5_dataloader import PairedSiameseDataset, siamese_collate_fn
    from phase2_5_model import Phase2_5_SiameseNetwork
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

def get_mass_regime(diff):
    if pd.isna(diff): return "Unknown"
    if diff < 0.01: return "0. Exact Isomers (<0.01 Da)"
    elif diff < 1.0: return "1. Isobaric (<1 Da)"
    elif diff < 10.0: return "2. Tiny (1-10 Da)"
    elif diff < 50.0: return "3. Medium (10-50 Da)"
    elif diff < 100.0: return "4. Large (50-100 Da)"
    else: return "5. Huge (>100 Da)"

def test_phase2_5(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[*] Using GPU {args.gpu_id}: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("[-] WARNING: CUDA is not available. Falling back to CPU.")

    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print(f"\n[*] Initializing Phase 2.5 Test Dataset from {args.test_pairs}...")
    test_dataset = PairedSiameseDataset(feather_path=args.test_pairs, graphs_path=args.graphs_path)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, 
        shuffle=False, collate_fn=siamese_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing Phase 2.5 Siamese Network...")
    if not os.path.exists(args.phase2_checkpoint):
        raise FileNotFoundError(f"[-] Phase 2 Checkpoint not found at {args.phase2_checkpoint}")
        
    model = Phase2_5_SiameseNetwork(full_config, args.phase2_checkpoint, device, max_fragments=10).to(device)
    
    print(f"[*] Loading Trained Phase 2.5 Weights from: {args.phase2_5_ckpt}")
    if not os.path.exists(args.phase2_5_ckpt):
        raise FileNotFoundError(f"[-] Phase 2.5 Checkpoint not found at {args.phase2_5_ckpt}")
        
    model.load_state_dict(torch.load(args.phase2_5_ckpt, map_location=device), strict=False)
    model.eval()

    all_targets = []
    all_preds = []
    
    print("\n[*] Commencing Evaluation...")
    test_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch_idx, (batch_A, batch_B, targets) in enumerate(test_bar):
            if batch_A is None or batch_B is None: continue
            
            batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}
            targets = targets.to(device, non_blocking=True)

            # tau=0.01 matches the hardened temperature from Epoch 3+
            pred_sims = model(batch_A, batch_B, tau=0.01)
            
            all_targets.extend(targets.cpu().numpy().flatten())
            all_preds.extend(pred_sims.cpu().numpy().flatten())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    print("\n" + "="*45)
    print(" 🐛 PREDICTION DIAGNOSTICS (ENTROPY) 🐛")
    print("="*45)
    print(f" True Sim Range : [{y_true.min():.4f}, {y_true.max():.4f}] (Mean: {y_true.mean():.4f}, Std: {y_true.std():.4f})")
    print(f" Pred Sim Range : [{y_pred.min():.4f}, {y_pred.max():.4f}] (Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f})")
    
    if y_pred.std() < 0.05:
        print("\n[!] WARNING: MODEL COLLAPSE DETECTED.")
    
    print("\n Sample Predictions (First 10):")
    for i in range(min(10, len(y_true))):
        print(f"   Pair {i:02d} -> True: {y_true[i]:.4f} | Predicted: {y_pred[i]:.4f}")

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_corr, _ = pearsonr(y_true, y_pred)
    else:
        pearson_corr = float('nan')

    print("\n" + "="*45)
    print(" 🏆 GLOBAL PHASE 2.5 REGRESSION METRICS 🏆")
    print("="*45)
    print(f" Mean Squared Error (MSE) : {mse:.4f}")
    print(f" Root Mean Sq. Err (RMSE) : {rmse:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" R-squared (R2) Score     : {r2:.4f}")
    print(f" Pearson Correlation (r)  : {pearson_corr:.4f}")
    print("="*45)
    
    print(f"\n[*] Loading original pairs data to map Mass Regimes...")
    if str(args.test_pairs).endswith('.feather'):
        pairs_df = pd.read_feather(args.test_pairs)
    else:
        pairs_df = pd.read_csv(args.test_pairs)

    print("\n📊 STRATIFIED PERFORMANCE BY MASS REGIME 📊")
    
    try:
        if len(pairs_df) != len(y_true):
            raise ValueError(f"Row count mismatch! pairs_df: {len(pairs_df)}, predictions: {len(y_true)}.")
            
        if 'mass_difference' in pairs_df.columns:
            all_mass_diffs = pairs_df['mass_difference'].values
        elif 'mass_diff' in pairs_df.columns:
            all_mass_diffs = pairs_df['mass_diff'].values
        else:
            all_mass_diffs = np.zeros(len(y_true))
            
        results_df = pd.DataFrame({
            'true_sim': y_true,
            'pred_sim': y_pred,
            'mass_diff': all_mass_diffs
        })
        
        results_df['mass_regime'] = results_df['mass_diff'].apply(get_mass_regime)
        regimes = sorted(results_df['mass_regime'].unique())
        
        if all_mass_diffs.sum() > 0:
            for regime in regimes:
                subset = results_df[results_df['mass_regime'] == regime]
                count = len(subset)
                if count > 0:
                    r_mse = mean_squared_error(subset['true_sim'], subset['pred_sim'])
                    r_mae = mean_absolute_error(subset['true_sim'], subset['pred_sim'])
                    print(f"[{regime}] -> MSE: {r_mse:.4f} | MAE: {r_mae:.4f} | Count: {count}")
        print()

        print(f"[*] Compiling aligned results into CSV...")
        final_df = pd.DataFrame({
            'name_main': pairs_df['name_main'],
            'name_sub': pairs_df['name_sub'],
            'true_entropy_sim': y_true,
            'pred_entropy_sim': y_pred,
            'mass_diff': all_mass_diffs,
            'mass_regime': results_df['mass_regime'],
            'entropy_label': pairs_df.get('entropy_label', np.zeros(len(y_true))) 
        })
        
    except Exception as e:
        print(f"\n[!] Error mapping original pair names ({e}).")
        final_df = pd.DataFrame({
            'true_entropy_sim': y_true,
            'pred_entropy_sim': y_pred
        })
    
    results_path = os.path.join(args.output_dir, "test_predictions_phase2_5_entropy.csv")
    final_df.to_csv(results_path, index=False)
    print(f"[+] Formatted predictions saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs", type=str, required=True)
    parser.add_argument("--graphs_path", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True)
    parser.add_argument("--phase2_5_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/phase2_5_evaluation")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    test_phase2_5(args)