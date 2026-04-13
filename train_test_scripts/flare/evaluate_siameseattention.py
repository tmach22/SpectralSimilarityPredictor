import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path

# --- SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'transport_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
    from siamese_cross_attention import SiameseCrossAttentionPredictor
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

def evaluate_attention_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # 1. Initialize Test Dataset
    print(f"\n[*] Initializing Test Dataset from: {args.test_pairs}")
    test_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.test_pairs, 
        spec_data_path=args.spec_data_path, 
        mol_data_path=args.mol_data_path,
        upsample_hard_negatives=False # STRICTLY FALSE FOR EVALUATION
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn, num_workers=4)

    # 2. Initialize Model and Load Weights
    print(f"[*] Loading Stage 2 Cross-Attention Weights from: {args.stage2_ckpt}")
    model = SiameseCrossAttentionPredictor(
        model_config=full_config['model'], 
        stage1_ckpt_path=args.stage1_ckpt, 
        meta_dim=args.meta_dim
    ).to(device)
    
    model.load_state_dict(torch.load(args.stage2_ckpt, map_location=device))
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    all_probs = []
    all_labels = []
    all_mass_diffs = []
    
    print("\n[*] Commencing Inference on Held-Out Test Set...")
    with torch.no_grad():
        for b_A, b_B, b_meta, b_massA, b_massB, labels, brics_A, brics_B in tqdm(test_loader, desc="Evaluating"):
            
            b_A = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b_A.items()} if isinstance(b_A, dict) else b_A.to(device)
            b_B = {k: v.to(device) if hasattr(v, 'to') else v for k, v in b_B.items()} if isinstance(b_B, dict) else b_B.to(device)
            
            b_meta, b_massA, b_massB = b_meta.to(device), b_massA.to(device), b_massB.to(device)
            brics_A = brics_A.to(device) if brics_A is not None else None
            brics_B = brics_B.to(device) if brics_B is not None else None
            
            # Forward Pass
            logits = model(b_A, b_B, brics_A, brics_B, b_massA, b_massB, b_meta)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(float) 
            
            # Compute Mass Differences for ICEBERG stratification
            mass_diffs = torch.abs(b_massA - b_massB).squeeze().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_mass_diffs.extend(mass_diffs)

    # 4. Enforce the Exact Output Structure
    results_df = test_dataset.pairs_df.copy()
    
    if 'label' in results_df.columns:
        results_df = results_df.rename(columns={'label': 'true_label'})
    
    results_df['prob_similarity'] = all_probs
    results_df['predicted_label'] = all_preds
    results_df['mass_diff'] = all_mass_diffs
    
    if 'cosine_similarity' not in results_df.columns:
        print("[!] Warning: 'cosine_similarity' not found in test pairs. Filling with 0.0.")
        results_df['cosine_similarity'] = 0.0
        
    cols_to_keep = ['name_main', 'name_sub', 'cosine_similarity', 'true_label', 'prob_similarity', 'predicted_label']
    final_output_df = results_df[cols_to_keep]

    # 5. Terminal Metrics (ICEBERG Stratification)
    print("\n" + "="*40)
    print(" 🏆 CROSS-ATTENTION GLOBAL TEST METRICS 🏆")
    print("="*40)
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(all_labels, all_probs):.4f}")
    print("="*40)

    results_df['mass_regime'] = results_df['mass_diff'].apply(get_mass_regime)
    print("\n📊 STRATIFIED PERFORMANCE BY MASS REGIME 📊")
    regimes = sorted(results_df['mass_regime'].unique())
    for regime in regimes:
        subset = results_df[results_df['mass_regime'] == regime]
        if len(subset) > 0:
            reg_f1 = f1_score(subset['true_label'], subset['predicted_label'], zero_division=0)
            reg_acc = accuracy_score(subset['true_label'], subset['predicted_label'])
            print(f"[{regime}] -> F1: {reg_f1:.4f} | Acc: {reg_acc:.4f} | Count: {len(subset)}")

    # 6. Save the Strictly Formatted CSV
    csv_path = os.path.join(args.output_dir, "siamese_attention_test_predictions.csv")
    final_output_df.to_csv(csv_path, index=False)
    print(f"\n[+] Detailed predictions saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Path to FLARE motif weights")
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Path to best Siamese Attention weights")
    parser.add_argument("--test_pairs", type=str, required=True, help="Path to held-out test feather file")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/msg_iceberg_predictions")
    
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--meta_dim", type=int, default=81)
    
    args = parser.parse_args()
    evaluate_attention_model(args)