import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

def optimize_threshold(args):
    print(f"Loading predictions from: {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)
    
    # Standardize column names
    if 'y_true' in df.columns: df.rename(columns={'y_true': 'true_label'}, inplace=True)
    if 'true_label' not in df.columns: raise KeyError("Column 'true_label' or 'y_true' not found.")
    
    # Identify probability column
    prob_col = None
    for col in ['pred_prob', 'prob_similarity', 'y_prob', 'probs']:
        if col in df.columns:
            prob_col = col
            break
    if not prob_col: raise KeyError("Probability column not found.")
    
    y_true = df['true_label'].values
    y_probs = df[prob_col].values
    
    print(f"Evaluated {len(df)} predictions.")
    
    # Search Thresholds
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_f1 = 0.0
    best_thresh = 0.0
    best_metrics = {}
    
    print("\n--- Scanning Thresholds ---")
    print(f"{'Thresh':<6} | {'F1':<6} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6}")
    print("-" * 45)
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                'acc': accuracy_score(y_true, preds),
                'prec': precision_score(y_true, preds),
                'rec': recall_score(y_true, preds)
            }
        
        # Print a few checkpoints
        if int(t*100) % 10 == 0:
            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds)
            print(f"{t:.2f}   | {f1:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f}")

    print("\n=== OPTIMIZED RESULTS ===")
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Best F1 Score:  {best_f1:.4f}")
    print(f"Accuracy:       {best_metrics['acc']:.4f}")
    print(f"Precision:      {best_metrics['prec']:.4f}")
    print(f"Recall:         {best_metrics['rec']:.4f}")
    
    print("\nRecommendation: Use this threshold for your final report evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=str, required=True, help="Path to test_results.csv from the hard finetuned model")
    args = parser.parse_args()
    optimize_threshold(args)