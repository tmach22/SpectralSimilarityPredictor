import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
import argparse
import os
import matplotlib.pyplot as plt

def find_optimal_threshold(args):
    """
    Analyzes prediction probabilities to find the optimal decision threshold 
    that maximizes the F1 Score.
    """
    
    # 1. Load Data
    try:
        df_results = pd.read_csv(args.results_csv_path)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Prediction results CSV not found. Error: {e}")
        return

    # Extract true labels and predicted probabilities
    y_true = df_results['true_label'].values
    y_scores = df_results['prob_similarity'].values

    # 2. Sweep Thresholds and Calculate Metrics
    
    # Generate 100 thresholds between 0.01 and 0.99
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    optimal_threshold = 0.5
    optimal_metrics = {}
    
    print("\n--- Sweeping Thresholds to Maximize F1 Score ---")

    for T in thresholds:
        # Convert probabilities to binary predictions based on current threshold T
        y_pred = (y_scores >= T).astype(int)
        
        # Calculate metrics for the current threshold
        # We only care about the positive class (label 1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, warn_for=tuple())
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = T
            
            # Recalculate all required metrics for the optimal point
            optimal_metrics = {
                'Accuracy': np.mean(y_true == y_pred),
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }

    # Calculate ROC-AUC once (it's threshold-independent)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 3. Report Results
    print(f"\n--- Optimal Threshold Found ---")
    print(f"Optimal Probability Threshold (T_opt): {optimal_threshold:.4f}")
    print(f"Maximize F1 Score: {best_f1:.4f}")
    print("-" * 40)
    print(f"Performance at T = {optimal_threshold:.4f}:")
    for name, value in optimal_metrics.items():
        print(f"  {name:<10}: {value:.4f}")
    print(f"  {'ROC-AUC':<10}: {roc_auc:.4f} (Threshold Independent)")
    print("-" * 40)

    # 4. Optional: Plot Precision-Recall Curve (for visualization)
    if args.plot_pr:
        # Use precision_recall_curve which handles continuous scores (y_scores)
        # Note: precision_recall_curve returns precision, recall, and thresholds
        precision_points, recall_points, pr_thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
        
        # We need to find the specific point that corresponds to the optimal_threshold
        # Find the index of the threshold closest to the T_opt found earlier
        idx_optimal = np.argmin(np.abs(pr_thresholds - optimal_threshold))

        plt.figure(figsize=(8, 6))
        
        # Plot the curve
        plt.plot(recall_points, precision_points, 
                 label=f'PR Curve (Max F1: {best_f1:.4f})')
                 
        # Mark the optimal F1 point
        plt.scatter(optimal_metrics['Recall'], optimal_metrics['Precision'], 
                    color='red', marker='o', s=100, 
                    label=f'Optimal F1 Point (T={optimal_threshold:.2f})')
        
        plt.xlabel('Recall (True Positive Rate)')
        plt.ylabel('Precision (Positive Predictive Value)')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.4)
        
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "precision_recall_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Precision-Recall plot saved to: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find optimal decision threshold for binary classifier.")
    parser.add_argument("--results_csv_path", type=str, required=True, help="Path to the model's test_results.csv file (must contain 'true_label' and 'prob_similarity').")
    parser.add_argument("--output_dir", type=str, default="./analysis_plots", help="Directory to save the resulting plot.")
    parser.add_argument("--plot_pr", action='store_true', help="Flag to generate and save the Precision-Recall curve plot.")

    args = parser.parse_args()
    find_optimal_threshold(args)