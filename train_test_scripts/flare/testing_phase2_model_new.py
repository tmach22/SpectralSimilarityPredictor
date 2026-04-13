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

# --- SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
    from siamese_sinkhorn_new import SiameseSinkhornPredictor
    from cross_modal_pretrainer_new import CrossModalFlarePretrainer
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
# MASS REGIME HELPER
# =============================================================================
def get_mass_regime(diff):
    if pd.isna(diff): return "Unknown"
    if diff < 0.01: return "0. Exact Isomers (<0.01 Da)"
    elif diff < 1.0: return "1. Isobaric (<1 Da)"
    elif diff < 10.0: return "2. Tiny (1-10 Da)"
    elif diff < 50.0: return "3. Medium (10-50 Da)"
    elif diff < 100.0: return "4. Large (50-100 Da)"
    else: return "5. Huge (>100 Da)"

def test_stage2(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # 1. Initialize Test Dataset
    print(f"\n[*] Initializing Stage 2 Test Dataset from {args.test_pairs}...")
    test_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.test_pairs, 
        spec_data_path=args.spec_data_path, 
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=False # No upsampling during testing
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, 
        shuffle=False, collate_fn=binary_collate_fn, num_workers=4
    )

    # 2. Initialize Model & Load Weights
    print("\n[*] Loading Pretrained Stage 1 Encoder & Siamese Sinkhorn Predictor...")
    
    # Init Stage 1 backbone (needed to construct Stage 2)
    stage1_encoder = CrossModalFlarePretrainer(
        model_config=full_config['model'],
        num_motifs=15, 
        drop_edge_p=0.30 
    )
    
    model = SiameseSinkhornPredictor(
        stage1_encoder=stage1_encoder,
        meta_dim=args.meta_dim,
        epsilon=0.05,
        rho=1.0,
        num_iters=30
    ).to(device)
    
    print(f"[*] Loading Trained Stage 2 Weights from: {args.stage2_ckpt}")
    model.load_state_dict(torch.load(args.stage2_ckpt, map_location=device))
    model.eval()

    # 3. Evaluation Loop
    all_targets = []
    all_preds = []
    all_mass_diffs = []
    
    print("\n[*] Commencing Evaluation...")
    test_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch in test_bar:
            # Unpack full batch elements including physics constraints
            (b_A, b_B, b_meta, b_massA, b_massB, target_sim,
             brics_A, brics_B, bde_A, bde_B, shift_frac, ce_norm) = batch

            def to_dev(x):
                return {k: v.to(device) if hasattr(v, 'to') else v
                        for k, v in x.items()} if isinstance(x, dict) else x.to(device)
            
            # Device placement
            b_A = to_dev(b_A)
            b_B = to_dev(b_B)
            b_meta = b_meta.to(device)
            b_massA = b_massA.to(device)
            b_massB = b_massB.to(device)
            brics_A = brics_A.to(device)
            brics_B = brics_B.to(device)
            bde_A = bde_A.to(device) if bde_A is not None else None
            bde_B = bde_B.to(device) if bde_B is not None else None
            shift_frac = shift_frac.to(device)
            ce_norm = ce_norm.to(device)
            target_sim_tensor = target_sim.view(-1).float().to(device)
            
            # Forward Pass (Only grab the first prediction output, ignore intermediate physics)
            pred_sim, _, _, _, _, _, _, _ = model(
                b_A, b_B, brics_A, brics_B,
                b_massA, b_massB, b_meta,
                A_bde_A=bde_A, A_bde_B=bde_B,
                shift_frac=shift_frac, ce_norm=ce_norm
            )
            
            # Calculate Absolute Mass Difference
            mass_diffs = torch.abs(b_massA - b_massB).cpu().numpy().flatten()
            
            # Store for metrics
            all_targets.extend(target_sim_tensor.cpu().numpy().flatten())
            all_preds.extend(pred_sim.cpu().numpy().flatten())
            all_mass_diffs.extend(mass_diffs)

    # 4. Calculate Final Metrics
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    # Regression Metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson Correlation (handles cases where variance might be zero)
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_corr, _ = pearsonr(y_true, y_pred)
    else:
        pearson_corr = float('nan')

    # 5. Print Global Metrics
    print("\n" + "="*45)
    print(" 🏆 GLOBAL STAGE 2 REGRESSION METRICS 🏆")
    print("="*45)
    print(f" Mean Squared Error (MSE) : {mse:.4f}")
    print(f" Root Mean Sq. Err (RMSE) : {rmse:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" R-squared (R2) Score     : {r2:.4f}")
    print(f" Pearson Correlation (r)  : {pearson_corr:.4f}")
    print("="*45)
    
    # 6. Stratified Performance by Mass Regime
    print("\n📊 STRATIFIED PERFORMANCE BY MASS REGIME 📊")
    
    results_df = pd.DataFrame({
        'true_sim': y_true,
        'pred_sim': y_pred,
        'mass_diff': all_mass_diffs
    })
    
    results_df['mass_regime'] = results_df['mass_diff'].apply(get_mass_regime)
    regimes = sorted(results_df['mass_regime'].unique())
    
    for regime in regimes:
        subset = results_df[results_df['mass_regime'] == regime]
        count = len(subset)
        if count > 0:
            r_mse = mean_squared_error(subset['true_sim'], subset['pred_sim'])
            r_mae = mean_absolute_error(subset['true_sim'], subset['pred_sim'])
            print(f"[{regime}] -> MSE: {r_mse:.4f} | MAE: {r_mae:.4f} | Count: {count}")
    print()

    # 7. Format and Save Results CSV
    print(f"[*] Compiling aligned results into CSV...")
    try:
        if str(args.test_pairs).endswith('.feather'):
            pairs_df = pd.read_feather(args.test_pairs)
        else:
            pairs_df = pd.read_csv(args.test_pairs)
            
        if len(pairs_df) == len(y_true):
            final_df = pd.DataFrame({
                'name_main': pairs_df['name_main'],
                'name_sub': pairs_df['name_sub'],
                'true_cosine_sim': y_true,
                'pred_cosine_sim': y_pred,
                'mass_diff': all_mass_diffs,
                'mass_regime': results_df['mass_regime']
            })
        else:
            raise ValueError(f"Row count mismatch! pairs_df: {len(pairs_df)}, predictions: {len(y_true)}")
            
    except Exception as e:
        print(f"[!] Error mapping original pair names ({e}). Saving raw predictions instead.")
        final_df = pd.DataFrame({
            'true_cosine_sim': y_true,
            'pred_cosine_sim': y_pred,
            'mass_diff': all_mass_diffs,
            'mass_regime': results_df['mass_regime']
        })
    
    results_path = os.path.join(args.output_dir, "test_predictions_stage2.csv")
    final_df.to_csv(results_path, index=False)
    print(f"[+] Formatted predictions saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Path to best Stage 1 weights")
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to best Stage 2 Sinkhorn weights")
    parser.add_argument("--test_pairs", type=str, required=True, help="Path to feather or csv file with testing pairs")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/stage2_evaluation")
    
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--meta_dim", type=int, default=81)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for testing (default: 0)")
    
    args = parser.parse_args()
    test_stage2(args)