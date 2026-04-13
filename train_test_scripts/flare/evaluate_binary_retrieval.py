import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_binary_retrieval(args):
    print(f"[*] Loading regression results from: {args.input_csv}")
    
    if not os.path.exists(args.input_csv):
        print(f"[!] Error: Could not find {args.input_csv}")
        return
        
    # Read the Stage 2 regression output
    df = pd.read_csv(args.input_csv)
    
    # Check if necessary columns exist (based on our Stage 2 testing script)
    required_cols = ['name_main', 'name_sub', 'true_cosine_sim', 'pred_cosine_sim']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in input CSV: {col}")

    print(f"[*] Applying binarization threshold: {args.threshold}")
    
    # 1. Generate Binary Labels based on the Threshold (e.g., >= 0.7)
    df['cosine_similarity'] = df['true_cosine_sim']
    df['prob_similarity'] = df['pred_cosine_sim']
    
    df['true_label'] = (df['cosine_similarity'] >= args.threshold).astype(float)
    df['predicted_label'] = (df['prob_similarity'] >= args.threshold).astype(float)
    
    # 2. Extract final DataFrame matching the requested structure
    final_df = df[[
        'name_main', 
        'name_sub', 
        'cosine_similarity', 
        'true_label', 
        'prob_similarity', 
        'predicted_label'
    ]]
    
    # Save the formatted CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)
    print(f"[+] Binary classification results saved to: {args.output_csv}")
    
    # 3. Calculate Performance Metrics
    y_true = final_df['true_label'].values
    y_prob = final_df['prob_similarity'].values
    y_pred = final_df['predicted_label'].values
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        # ROC-AUC uses the continuous probabilities, not the hard predictions
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')
        
    # Calculate Confusion Matrix elements for deeper insight
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 4. Print the Metrics
    print("\n" + "="*45)
    print(" 🎯 BINARY CLASSIFICATION METRICS (Threshold: {:.2f}) 🎯".format(args.threshold))
    print("="*45)
    print(f" Accuracy        : {accuracy:.4f}")
    print(f" Precision       : {precision:.4f}  (When it predicts match, is it right?)")
    print(f" Recall          : {recall:.4f}  (Did it find all the true matches?)")
    print(f" F1-Score        : {f1:.4f}")
    print(f" ROC-AUC         : {roc_auc:.4f}")
    print("-" * 45)
    print(f" True Positives  : {tp} (Correctly predicted match)")
    print(f" True Negatives  : {tn} (Correctly predicted non-match)")
    print(f" False Positives : {fp} (Falsely predicted match - Type I Error)")
    print(f" False Negatives : {fn} (Missed a true match - Type II Error)")
    print("="*45)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert regression similarities to binary classification metrics.")
    
    # Default paths assume you run this from the same directory as your training scripts
    parser.add_argument("--input_csv", type=str, 
                        default="results/stage2_evaluation/test_predictions_stage2.csv",
                        help="Path to the regression predictions CSV")
                        
    parser.add_argument("--output_csv", type=str, 
                        default="results/stage2_evaluation/binary_classification_results.csv",
                        help="Path to save the binary formatted CSV")
                        
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Cosine similarity threshold for a positive match (default: 0.7)")
                        
    args = parser.parse_args()
    evaluate_binary_retrieval(args)