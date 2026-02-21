import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score
from pathlib import Path
import copy

# --- 1. SETUP SYS.PATH ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, "train_test_scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from classifier_siamesemodel_new import SiameseSpectralSimilarityModel
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def merge_configs(base_config, custom_config):
    """
    Recursively merges the custom config into the base config.
    """
    merged_config = copy.deepcopy(base_config)
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    return merged_config

def test_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Dynamic Output Naming ---
    input_filename = Path(args.test_pairs_path).stem 
    output_filename = f"results_{input_filename}_uncertainty.csv"
    save_path = os.path.join(args.output_dir, output_filename)
    
    print(f"--- Starting Binary Model Testing (With Uncertainty Awareness) ---")
    print(f"Target Dataset: {input_filename}")
    print(f"Output will be saved to: {save_path}")
    print(f"Confidence Thresholds: Low <= {args.conf_low} | High >= {args.conf_high}")

    # --- 2. Load Data ---
    print(f"Loading Dataset and Metadata...")
    
    # A. Load the PyTorch Dataset (For Model)
    test_dataset = BinaryClassificationDataset(
        args.test_pairs_path, 
        args.spec_data_path, 
        args.mol_data_path
    )
    
    # B. Load the Raw DataFrame (For IDs)
    df_meta = pd.read_feather(args.test_pairs_path)
    
    # SAFETY CHECK: 
    if len(test_dataset) != len(df_meta):
        print(f"CRITICAL WARNING: Dataset length ({len(test_dataset)}) != File length ({len(df_meta)}).")
        print("Trimming metadata to match the Dataset...")
        df_meta = df_meta.iloc[:len(test_dataset)]

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # <--- MUST BE FALSE TO KEEP ALIGNMENT
        collate_fn=binary_collate_fn, 
        num_workers=4
    )
    
    spec_meta_dim = test_dataset[0][2].shape[1]
    print(f"Dataset size: {len(test_dataset)} pairs")

    # --- 3. Initialize Model ---
    print("Initializing Model...")
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim
    ).to(device)

    # --- 4. Load Weights ---
    print(f"Loading Model Weights: {os.path.basename(args.binary_model_path)}")
    state_dict = torch.load(args.binary_model_path, map_location=device)
    
    # Handle possible nested state_dict keys
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'best_model_sd' in state_dict:
        state_dict = state_dict['best_model_sd']
        
    model.load_state_dict(state_dict)
    model.eval()

    # --- 5. Inference Loop ---
    print("Running Inference...")
    all_probs = []
    all_preds = []      # Standard threshold (0.5)
    all_trust_preds = [] # 3-class: 0, 1, -1 (Uncertain)
    all_labels = []
    
    with torch.no_grad():
        for batch_A, batch_B, batch_meta, mass_diffs, labels in tqdm(test_loader, desc="Testing"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            mass_diffs = mass_diffs.to(device) 
            
            logits = model(batch_A, batch_B, batch_meta, mass_diffs).view(-1)
            probs = torch.sigmoid(logits)
            
            # A. Standard Prediction (Threshold 0.5)
            preds = (probs > 0.5).float()
            
            # B. Uncertainty-Aware Prediction
            # Initialize with -1 (Uncertain)
            trust_preds = torch.full_like(probs, -1).int()
            
            # Mark Confident Negatives (Prob <= Low)
            trust_preds[probs <= args.conf_low] = 0
            
            # Mark Confident Positives (Prob >= High)
            trust_preds[probs >= args.conf_high] = 1
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_trust_preds.extend(trust_preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 6. METRICS: STANDARD (Legacy) ---
    print("\n" + "="*40)
    print("  STANDARD METRICS (Threshold 0.5)  ")
    print("="*40)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")

    # --- 7. METRICS: UNCERTAINTY-AWARE (Selective Prediction) ---
    print("\n" + "="*40)
    print(f"  TRUST METRICS (Excluding {args.conf_low} < P < {args.conf_high})  ")
    print("="*40)
    
    # Convert lists to numpy for masking
    np_labels = np.array(all_labels)
    np_trust_preds = np.array(all_trust_preds)
    
    # Filter: Keep only confident predictions (0 or 1)
    mask_confident = (np_trust_preds != -1)
    
    n_total = len(np_labels)
    n_confident = np.sum(mask_confident)
    n_uncertain = n_total - n_confident
    abstention_rate = n_uncertain / n_total
    
    print(f"Total Predictions: {n_total}")
    print(f"Confident Predictions: {n_confident}")
    print(f"Uncertain Predictions: {n_uncertain}")
    print(f"Abstention Rate:       {abstention_rate:.2%}")
    
    if n_confident > 0:
        y_true_conf = np_labels[mask_confident]
        y_pred_conf = np_trust_preds[mask_confident]
        
        trust_acc = accuracy_score(y_true_conf, y_pred_conf)
        trust_prec = precision_score(y_true_conf, y_pred_conf)
        trust_rec = recall_score(y_true_conf, y_pred_conf)
        trust_f1 = f1_score(y_true_conf, y_pred_conf)
        
        print(f"\n--- Performance on Confident Data ---")
        print(f"Trust Accuracy:  {trust_acc:.4f}  (vs {acc:.4f})")
        print(f"Trust Precision: {trust_prec:.4f}")
        print(f"Trust Recall:    {trust_rec:.4f}")
        print(f"Trust F1 Score:  {trust_f1:.4f}")
    else:
        print("\nWARNING: Model was uncertain about ALL predictions!")

    # --- 8. Save Results WITH IDs ---
    print(f"\nSaving results to CSV...")
    
    results_df = pd.DataFrame()
    results_df['name_main'] = df_meta['name_main'].values
    results_df['name_sub'] = df_meta['name_sub'].values
    if 'cosine_similarity' in df_meta.columns:
        results_df['cosine_similarity'] = df_meta['cosine_similarity'].values
    
    results_df['true_label'] = all_labels
    results_df['prob_similarity'] = all_probs
    results_df['predicted_label'] = all_preds # Standard
    results_df['trust_label'] = all_trust_preds # -1 for Uncertain
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(save_path, index=False)
    
    print(f"Done. Detailed results saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs_path", type=str, required=True, help="Path to ANY binary dataset feather file")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--binary_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    # New Confidence Arguments
    parser.add_argument("--conf_low", type=float, default=0.35, help="Probability threshold for confident NEGATIVE")
    parser.add_argument("--conf_high", type=float, default=0.7, help="Probability threshold for confident POSITIVE")

    args = parser.parse_args()
    test_binary(args)