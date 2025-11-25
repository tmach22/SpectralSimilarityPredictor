import pandas as pd
import argparse
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split

def main(args):
    """
    Splits the balanced 1:1:1 dataset into train, validation, and test sets
    while preserving the 1:1:1 ratio via stratification.
    """
    print(f"--- Starting Stratified 1:1:1 Split ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "split_report.txt")
    
    try:
        summary_file = open(report_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file {report_path} for writing. Error: {e}")
        return

    print(f"Loading balanced dataset from: {args.input_file}")
    start_time = time.time()
    try:
        df = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        summary_file.close()
        return
        
    print(f"Loaded {len(df):,} total pairs in {time.time() - start_time:.2f}s")
    
    # --- 2. Create the Stratification Column ---
    print("Creating temporary stratification bins...")
    
    # Define the bin edges and labels
    # Bins: (-inf, 0.1), [0.1, 0.8), [0.8, inf)
    bin_edges = [-np.inf, args.negative_threshold, args.positive_threshold, np.inf]
    bin_labels = ["low", "medium", "high"]
    
    # Create the new 'cosine_bin' column
    df['cosine_bin'] = pd.cut(df['cosine_similarity'], bins=bin_edges, labels=bin_labels, right=False)
    
    # Check the initial distribution
    print("\nInitial Data Distribution:")
    print(df['cosine_bin'].value_counts(normalize=True).sort_index())

    # --- 3. Perform Stratified Splits ---
    # We will do two splits:
    # 1. Split df into train_df (e.g., 80%) and temp_df (e.g., 20%)
    # 2. Split temp_df into val_df (e.g., 10%) and test_df (e.g., 10%)
    
    # Calculate the size of the first split (train vs. the rest)
    train_frac = 1.0 - (args.val_size + args.test_size)
    # Calculate the size of the second split (test vs. val, relative to temp_df)
    test_frac_of_temp = args.test_size / (args.val_size + args.test_size)

    print(f"\nSplitting data: {train_frac*100:.0f}% Train, {args.val_size*100:.0f}% Val, {args.test_size*100:.0f}% Test")

    # Split 1: Train vs. Temp (Val+Test)
    print("Performing split 1 (Train vs. Temp)...")
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        shuffle=True,
        stratify=df['cosine_bin'], # <-- THIS IS THE CRITICAL STEP
        random_state=args.seed
    )

    # Split 2: Val vs. Test
    print("Performing split 2 (Val vs. Test)...")
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac_of_temp,
        shuffle=True,
        stratify=temp_df['cosine_bin'], # <-- MUST STRATIFY AGAIN
        random_state=args.seed
    )

    # Drop the temporary 'cosine_bin' column before saving
    train_df = train_df.drop(columns=['cosine_bin'])
    val_df = val_df.drop(columns=['cosine_bin'])
    test_df = test_df.drop(columns=['cosine_bin'])
    
    print("\nSplits created successfully.")

    # --- 4. Verify Distributions and Save ---
    
    report = "--- Stratified Split Report ---\n\n"
    report += f"Original Dataset: {len(df):,} pairs\n"
    report += f"{df['cosine_bin'].value_counts(normalize=True).sort_index()}\n\n"

    report += f"Train Dataset: {len(train_df):,} pairs\n"
    # We check the distribution by re-creating the bin on the fly for the report
    report += f"{pd.cut(train_df['cosine_similarity'], bins=bin_edges, labels=bin_labels, right=False).value_counts(normalize=True).sort_index()}\n\n"

    report += f"Validation Dataset: {len(val_df):,} pairs\n"
    report += f"{pd.cut(val_df['cosine_similarity'], bins=bin_edges, labels=bin_labels, right=False).value_counts(normalize=True).sort_index()}\n\n"
    
    report += f"Test Dataset: {len(test_df):,} pairs\n"
    report += f"{pd.cut(test_df['cosine_similarity'], bins=bin_edges, labels=bin_labels, right=False).value_counts(normalize=True).sort_index()}\n\n"
    
    print(report)
    summary_file.write(report)
    summary_file.close()

    # --- 5. Save Files ---
    try:
        train_path = os.path.join(args.output_dir, "train_pairs.feather")
        val_path = os.path.join(args.output_dir, "val_pairs.feather")
        test_path = os.path.join(args.output_dir, "test_pairs.feather")
        
        train_df.reset_index(drop=True).to_feather(train_path)
        val_df.reset_index(drop=True).to_feather(val_path)
        test_df.reset_index(drop=True).to_feather(test_path)
        
        print(f"Successfully saved train, val, and test files to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError saving split files: {e}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split the balanced dataset into stratified train/val/test sets.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your BALANCED 1:1:1 dataset (e.g., balanced_111_dataset.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the new 'train_pairs.feather', 'val_pairs.feather', and 'test_pairs.feather'.")
    
    # --- Split Ratios ---
    parser.add_argument("--val_size", type=float, default=0.1,
                        help="Fraction of the total data to use for validation (e.g., 0.1 for 10%).")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Fraction of the total data to use for testing (e.g., 0.1 for 10%).")
    
    # --- Bin Thresholds (Must match previous script) ---
    parser.add_argument("--positive_threshold", type=float, default=0.8,
                        help="Cosine similarity threshold for 'Positive' bin.")
    parser.add_argument("--negative_threshold", type=float, default=0.1,
                        help="Cosine similarity threshold for 'Negative' bin.")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits.")

    args = parser.parse_args()
    
    if (args.val_size + args.test_size) >= 1.0:
        print("Error: val_size and test_size must sum to less than 1.0")
    else:
        main(args)