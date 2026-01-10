import pandas as pd
import argparse
import os

def check_abundance(file_path):
    print(f"--- Analyzing Data Abundance for Retraining ---")
    print(f"Loading file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load the Feather file
    try:
        df = pd.read_feather(file_path)
    except Exception as e:
        print(f"Error loading feather file: {e}")
        return

    total_pairs = len(df)
    print(f"Total Pairs in Dataset: {total_pairs:,}")

    # Verify columns exist
    if 'cosine_similarity' not in df.columns:
        print("Error: Column 'cosine_similarity' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # --- Analysis ---
    
    # 1. Current State (Threshold 0.7)
    # We check the actual distribution based on the raw similarity score
    current_positives = df[df['cosine_similarity'] >= 0.7]
    n_current = len(current_positives)
    
    # 2. Proposed State (Threshold 0.8)
    proposed_positives = df[df['cosine_similarity'] >= 0.8]
    n_proposed = len(proposed_positives)
    
    # 3. Very Strict State (Threshold 0.85)
    strict_positives = df[df['cosine_similarity'] >= 0.85]
    n_strict = len(strict_positives)

    print("\n--- Impact of Changing Threshold ---")
    print(f"{'Threshold':<20} | {'Positive Pairs':<15} | {'% of Total':<15} | {'Retention Rate'}")
    print("-" * 75)
    
    # Calculate percentages
    pct_total_curr = (n_current / total_pairs) * 100
    pct_total_prop = (n_proposed / total_pairs) * 100
    pct_total_strict = (n_strict / total_pairs) * 100
    
    retention_prop = (n_proposed / n_current) * 100 if n_current > 0 else 0
    retention_strict = (n_strict / n_current) * 100 if n_current > 0 else 0

    print(f"{'>= 0.70 (Current)':<20} | {n_current:<15,} | {pct_total_curr:.1f}%           | 100.0% (Baseline)")
    print(f"{'>= 0.80 (Proposed)':<20} | {n_proposed:<15,} | {pct_total_prop:.1f}%           | {retention_prop:.1f}% of baseline")
    print(f"{'>= 0.85 (Strict)':<20} | {n_strict:<15,} | {pct_total_strict:.1f}%           | {retention_strict:.1f}% of baseline")

    print("\n--- Strategic Recommendation ---")
    if n_proposed > 100000:
        print("✅ GREEN LIGHT: You have >100k positive pairs at the 0.8 threshold.")
        print("   Action: Retrain immediately. This will significantly reduce false positives without data starvation.")
    elif n_proposed > 50000:
        print("⚠️ YELLOW LIGHT: You have 50k-100k positive pairs.")
        print("   Action: Safe to retrain, but you might want to upsample the positives or use class weighting.")
    else:
        print("❌ RED LIGHT: Fewer than 50k positive pairs.")
        print("   Action: Do not retrain with a hard 0.8 cutoff. You will starve the model.")
        print("   Alternative: Keep 0.7 threshold for training, but enforce 0.8 threshold only during inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check positive pair abundance for retraining.")
    parser.add_argument("--file_path", type=str, help="Path to your training .feather file")
    
    args = parser.parse_args()
    check_abundance(args.file_path)