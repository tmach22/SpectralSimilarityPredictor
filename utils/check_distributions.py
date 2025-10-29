import pandas as pd
import argparse
import os
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr

def analyze_distribution(df, split_name, summary_file):
    """Calculates and reports the internal distribution of a golden dataset."""
    
    header = f"\n==================================================\n"
    header += f"DISTRIBUTION ANALYSIS FOR: {split_name}\n"
    header += f"==================================================\n"
    
    if df.empty:
        report = "  Count: 0\n"
        print(header + report)
        summary_file.write(header + report)
        return

    total_count = len(df)
    
    # Re-calculate the two sub-groups
    df_high = df.query("cosine_similarity >= 0.8")
    df_low = df.query("cosine_similarity < 0.8")
    
    count_high = len(df_high)
    count_low = len(df_low)
    
    # --- The Most Important Metric: The Ratio ---
    pct_high = (count_high / total_count) * 100
    pct_low = (count_low / total_count) * 100
    
    report = f"  Total Pairs: {total_count:,}\n"
    report += "  ------------------------------------------------\n"
    report += f"  Positive/Negative Ratio:\n"
    report += f"    - High-Signal (Positives, Cos>=0.8): {count_high:10,} ({pct_high:5.1f}%)\n"
    report += f"    - Clean-Signal (Negatives, Cos<0.8): {count_low:10,} ({pct_low:5.1f}%)\n"
    report += "  ------------------------------------------------\n"

    # --- Overall Stats ---
    try:
        # Clean data for stats
        clean_df = df[['tanimoto', 'cosine_similarity']].replace([np.inf, -np.inf], np.nan).dropna()
        
        mean_cosine = clean_df['cosine_similarity'].mean()
        mean_tanimoto = clean_df['tanimoto'].mean()
        pearson_corr, _ = pearsonr(clean_df['tanimoto'], clean_df['cosine_similarity'])
        spearman_corr, _ = spearmanr(clean_df['tanimoto'], clean_df['cosine_similarity'])

        report += f"  Overall Stats (for this split):\n"
        report += f"    - Mean Cosine:     {mean_cosine:.4f}\n"
        report += f"    - Mean Tanimoto:   {mean_tanimoto:.4f}\n"
        report += f"    - Pearson (r):     {pearson_corr:.4f}\n"
        report += f"    - Spearman (r):    {spearman_corr:.4f}\n"

    except Exception as e:
        report += f"  Could not calculate stats: {e}\n"

    print(header + report)
    summary_file.write(header + report)

def main(args):
    """
    Loads all three golden datasets (train, val, test) and
    compares their internal distributions side-by-side.
    """
    print(f"--- Starting Distribution Comparison ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "distribution_comparison_report.txt")
    
    try:
        summary_file = open(summary_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file {summary_path} for writing. Error: {e}")
        return

    datasets_to_check = {
        "Train Set": args.train_path,
        "Validation Set": args.val_path,
        "Test Set": args.test_path
    }
    
    for split_name, file_path in datasets_to_check.items():
        try:
            print(f"Loading {split_name} from {file_path}...")
            df = pd.read_feather(file_path)
            analyze_distribution(df, split_name, summary_file)
        except Exception as e:
            error_msg = f"\n==================================================\n"
            error_msg += f"ERROR LOADING: {split_name}\n"
            error_msg += f"File not found or corrupt: {file_path}\n"
            error_msg += f"Error: {e}\n"
            error_msg += f"==================================================\n"
            print(error_msg)
            summary_file.write(error_msg)
    
    summary_file.close()
    print(f"\n--- Analysis Complete ---")
    print(f"Comparison report saved to: {summary_path}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare the internal distributions of train, val, and test 'golden' datasets.")
    
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to your 'golden_train_pairs.feather'.")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to your 'golden_val_pairs.feather'.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to your 'golden_test_pairs.feather'.")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the 'distribution_comparison_report.txt'.")
    
    args = parser.parse_args()
    main(args)