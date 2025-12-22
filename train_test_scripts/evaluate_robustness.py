import pandas as pd
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def load_pickle(path):
    with open(path, 'rb') as f:
        return pd.read_pickle(f)

def compute_metrics_safe(y_true, y_pred, y_prob):
    """
    Computes metrics safely, handling edge cases.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan 
            
        return acc, prec, rec, f1, auc
    except Exception as e:
        return 0, 0, 0, 0, 0

def standardize_columns(df):
    """
    Renames columns to standard 'true_label', 'pred_label', 'pred_prob'.
    """
    # 1. Normalize 'True Label'
    if 'true_label' not in df.columns:
        if 'y_true' in df.columns: df.rename(columns={'y_true': 'true_label'}, inplace=True)
        elif 'label' in df.columns: df.rename(columns={'label': 'true_label'}, inplace=True)

    # 2. Normalize 'Predicted Label' (The binary 0/1)
    if 'pred_label' not in df.columns:
        if 'predicted_label' in df.columns: df.rename(columns={'predicted_label': 'pred_label'}, inplace=True)
        elif 'y_pred' in df.columns: df.rename(columns={'y_pred': 'pred_label'}, inplace=True)
        elif 'preds' in df.columns: df.rename(columns={'preds': 'pred_label'}, inplace=True)

    # 3. Normalize 'Predicted Probability' (The float 0.0-1.0)
    if 'pred_prob' not in df.columns:
        if 'prob_similarity' in df.columns: df.rename(columns={'prob_similarity': 'pred_prob'}, inplace=True)
        elif 'y_prob' in df.columns: df.rename(columns={'y_prob': 'pred_prob'}, inplace=True)
        elif 'probs' in df.columns: df.rename(columns={'probs': 'pred_prob'}, inplace=True)

    # Validate
    required = ['true_label', 'pred_label', 'pred_prob']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"DEBUG: Current columns: {df.columns.tolist()}")
        raise KeyError(f"Could not find columns for: {missing}. Please check CSV headers.")
        
    return df

def evaluate_robustness(args):
    print("--- Starting Robustness Stress Test (V4) ---")
    
    # 1. Load Predictions
    if not os.path.exists(args.predictions_csv):
        print(f"Error: File {args.predictions_csv} not found.")
        return

    print(f"Loading predictions from: {args.predictions_csv}")
    res_df = pd.read_csv(args.predictions_csv)
    
    # --- AUTO-FIX COLUMN NAMES ---
    res_df = standardize_columns(res_df)
    
    # 2. Load Metadata & Source Pairs
    print("Loading Metadata...")
    spec_df = load_pickle(args.spec_data_path)
    
    print(f"Loading Source Test Pairs from: {args.test_pairs_path}")
    test_pairs = pd.read_feather(args.test_pairs_path)
    
    # Merge
    if len(res_df) != len(test_pairs):
        print(f"Warning: Length mismatch! Preds: {len(res_df)}, Pairs: {len(test_pairs)}. Truncating.")
        min_len = min(len(res_df), len(test_pairs))
        res_df = res_df.iloc[:min_len]
        test_pairs = test_pairs.iloc[:min_len]
    
    full_df = pd.concat([test_pairs.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
    
    # --- HELPER: Print Row ---
    def print_header(title):
        print(f"\n=== {title} ===")
        print(f"{'Stratum':<20} | {'N':<6} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6}")
        print("-" * 75)

    def print_row(label, n, acc, prec, rec, f1, auc):
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"{label:<20} | {n:<6} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc_str}")

    # ==========================================
    # STRATIFICATION 1: Instrument Type
    # ==========================================
    print_header("Instrument Robustness")
    
    spec_id_to_inst = {}
    if 'inst_type' in spec_df.columns:
        use_index = not ('spec_id' in spec_df.columns)
        for idx, row in tqdm(spec_df.iterrows(), total=len(spec_df), desc="Mapping Inst", leave=False):
            sid = idx if use_index else row['spec_id']
            spec_id_to_inst[sid] = row.get('inst_type', 'Unknown')
    else:
        print("Warning: 'inst_type' column not found.")

    full_df['instrument'] = full_df['name_main'].map(spec_id_to_inst).fillna('Unknown')
    
    for inst, group in full_df.groupby('instrument'):
        if len(group) < 20: continue 
        acc, prec, rec, f1, auc = compute_metrics_safe(group['true_label'], group['pred_label'], group['pred_prob'])
        print_row(inst, len(group), acc, prec, rec, f1, auc)

    # ==========================================
    # STRATIFICATION 2: Structural Difficulty
    # ==========================================
    if 'tanimoto' in full_df.columns:
        print_header("Structural Difficulty (Hard Pairs)")
        bins = [0.0, 0.4, 0.6, 0.85, 1.0]
        labels = ["Low (0-0.4)", "Med (0.4-0.6)", "High (0.6-0.85)", "V.High (>0.85)"]
        full_df['struct_bin'] = pd.cut(full_df['tanimoto'], bins=bins, labels=labels)
        
        for bin_label in labels:
            group = full_df[full_df['struct_bin'] == bin_label]
            if len(group) == 0: continue
            acc, prec, rec, f1, auc = compute_metrics_safe(group['true_label'], group['pred_label'], group['pred_prob'])
            print_row(bin_label, len(group), acc, prec, rec, f1, auc)
    else:
        print("\nSkipping Structural Stratification (Tanimoto missing).")

    # ==========================================
    # STRATIFICATION 3: Precursor Charge
    # ==========================================
    if 'prec_type' in spec_df.columns:
        print_header("Precursor Charge Robustness")
        
        spec_id_to_prec = {}
        use_index = not ('spec_id' in spec_df.columns)
        for idx, row in tqdm(spec_df.iterrows(), total=len(spec_df), desc="Mapping Prec", leave=False):
            sid = idx if use_index else row['spec_id']
            spec_id_to_prec[sid] = row.get('prec_type', 'Unknown')

        full_df['prec_type'] = full_df['name_main'].map(spec_id_to_prec).fillna('Unknown')
        top_charges = full_df['prec_type'].value_counts().nlargest(5).index
        
        for charge in top_charges:
            group = full_df[full_df['prec_type'] == charge]
            acc, prec, rec, f1, auc = compute_metrics_safe(group['true_label'], group['pred_label'], group['pred_prob'])
            print_row(charge, len(group), acc, prec, rec, f1, auc)

    # ==========================================
    # STRATIFICATION 4: "Surprise" Pairs
    # ==========================================
    # Defined as: Structurally Distinct (Tanimoto < 0.5) BUT Spectrally Similar (Label=1)
    if 'tanimoto' in full_df.columns:
        surprise_mask = (full_df['tanimoto'] < 0.5) & (full_df['true_label'] == 1)
        surprise_group = full_df[surprise_mask]
        
        if len(surprise_group) > 0:
            print("\n=== 'Surprise' Pairs (Low Struct / High Spec) ===")
            acc, prec, rec, f1, _ = compute_metrics_safe(surprise_group['true_label'], surprise_group['pred_label'], surprise_group['pred_prob'])
            print(f"Count: {len(surprise_group)}")
            print(f"Recall (Accuracy on Positives): {rec:.4f}")
            print("Interpretation: How often does the model correctly identify spectral similarity when structure suggests otherwise?")
            print("-" * 75)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    
    args = parser.parse_args()
    evaluate_robustness(args)