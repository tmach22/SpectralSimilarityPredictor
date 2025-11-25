import pandas as pd
import argparse
import os
import time
import numpy as np

def main(args):
    """
    ### MODIFIED ###
    Creates a new, 10-bin balanced dataset from the heavily skewed
    brute-force file by sampling equally from 10 bins of 0.1 width.
    """
    print(f"--- Starting 10-Bin Balanced Dataset Creation (for Regression) ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    ### MODIFIED ###
    report_path = os.path.join(args.output_dir, "balanced_10bin_dataset_report.txt")
    output_file_path = os.path.join(args.output_dir, "balanced_10bin_dataset.feather")
    ### END MODIFIED ###
    
    try:
        summary_file = open(report_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file {report_path} for writing. Error: {e}")
        return

    print(f"Loading full (skewed) dataset from: {args.input_file}")
    print("This may take a significant amount of time and RAM...")
    start_time = time.time()
    try:
        df = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        summary_file.close()
        return
        
    print(f"Loaded {len(df):,} total pairs in {time.time() - start_time:.2f}s")
    
    ### MODIFIED ###
    # --- 2. Define 10 Populations ---
    print(f"\nDefining 10 populations based on 0.1-width bins:")
    
    # Define bin edges and labels
    bins_defs = [
        {"label": "0.0-0.1", "query": "cosine_similarity >= 0.0 and cosine_similarity < 0.1"},
        {"label": "0.1-0.2", "query": "cosine_similarity >= 0.1 and cosine_similarity < 0.2"},
        {"label": "0.2-0.3", "query": "cosine_similarity >= 0.2 and cosine_similarity < 0.3"},
        {"label": "0.3-0.4", "query": "cosine_similarity >= 0.3 and cosine_similarity < 0.4"},
        {"label": "0.4-0.5", "query": "cosine_similarity >= 0.4 and cosine_similarity < 0.5"},
        {"label": "0.5-0.6", "query": "cosine_similarity >= 0.5 and cosine_similarity < 0.6"},
        {"label": "0.6-0.7", "query": "cosine_similarity >= 0.6 and cosine_similarity < 0.7"},
        {"label": "0.7-0.8", "query": "cosine_similarity >= 0.7 and cosine_similarity < 0.8"},
        {"label": "0.8-0.9", "query": "cosine_similarity >= 0.8 and cosine_similarity < 0.9"},
        {"label": "0.9-1.0", "query": "cosine_similarity >= 0.9 and cosine_similarity <= 1.0"} # Include 1.0
    ]
    ### END MODIFIED ###
    
    dataframes = {}
    counts = {}

    for bin_def in bins_defs:
        label = bin_def["label"]
        query_str = bin_def["query"]
        print(f"  Populating Bin '{label}' ({query_str})...")
        
        df_bin = df.query(query_str).copy()
        dataframes[label] = df_bin
        counts[label] = len(df_bin)
        print(f"    Found {counts[label]:,} pairs.")

    ### MODIFIED ###
    # --- 3. Perform 1:1:...:1 Sampling (10 bins) ---
    
    # Set the anchor bin to the highest similarity bin, as requested
    anchor_bin_label = "0.9-1.0"
    n_sample_size = counts[anchor_bin_label]
    ### END MODIFIED ###
    
    if n_sample_size == 0:
        print(f"\nFatal Error: Anchor bin ({anchor_bin_label}) has 0 samples. Cannot proceed.")
        summary_file.close()
        return
        
    print(f"\nUsing anchor bin '{anchor_bin_label}' as sample size: {n_sample_size:,} pairs.")

    sampled_dataframes = []
    
    for bin_def in bins_defs:
        label = bin_def["label"]
        df_bin = dataframes[label]
        n_bin = counts[label]

        if n_bin == 0:
            print(f"Warning: Bin '{label}' has 0 samples. Final dataset will be missing this bin.")
            continue
        
        # This is the anchor bin, as specified. We take all its samples.
        if label == anchor_bin_label:
            print(f"Using all {n_bin:,} pairs from anchor bin '{label}'.")
            sampled_dataframes.append(df_bin)
            continue
            
        # For all other bins, sample to match the anchor size.
        # This requires OVERSAMPLING (replace=True) if the bin is smaller than the anchor.
        if n_bin < n_sample_size:
            print(f"Warning: Bin '{label}' ({n_bin:,}) is smaller than anchor ({n_sample_size:,}). Oversampling with replacement.")
            df_sampled = df_bin.sample(n=n_sample_size, random_state=args.seed, replace=True)
        else:
            # Undersample (no replacement)
            print(f"Sampling {n_sample_size:,} pairs from bin '{label}' ({n_bin:,} available)...")
            df_sampled = df_bin.sample(n=n_sample_size, random_state=args.seed, replace=False)
        
        sampled_dataframes.append(df_sampled)

    # --- 4. Combine and Save ---
    print("Combining and shuffling final dataset...")
    df_balanced = pd.concat(sampled_dataframes)
    
    # Shuffle the final combined dataset
    df_balanced = df_balanced.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    total_balanced_count = len(df_balanced)
    
    try:
        df_balanced.to_feather(output_file_path)
        ### MODIFIED ###
        print(f"\nSuccessfully saved balanced 10-bin dataset to: {output_file_path}")
        ### END MODIFIED ###
    except Exception as e:
        print(f"\nError saving final file: {e}")
        summary_file.close()
        return

    ### MODIFIED ###
    # --- 5. Write Final Report ---
    report = "--- Balanced 10-Bin Dataset Report ---\n"
    report += f"Source file: {args.input_file}\n"
    report += f"Total Original Pairs: {len(df):,}\n"
    report += "\n--- New Balanced Dataset ---\n"
    report += f"Anchor Bin: '{anchor_bin_label}' (Target Size: {n_sample_size:,})\n"
    report += f"Total New Pairs: {total_balanced_count:,}\n\n"
    
    # Calculate the final distribution for the report
    final_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01] # Use 1.01 to include 1.0
    final_labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", 
                    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    df_balanced['temp_bin'] = pd.cut(df_balanced['cosine_similarity'], bins=final_bins, labels=final_labels, right=False)
    
    report += "Final Distribution in Saved File:\n"
    report += f"{df_balanced['temp_bin'].value_counts().sort_index()}\n"
    
    report += f"\nFile saved to: {output_file_path}\n"
    
    print(f"\n{report}")
    summary_file.write(report)
    summary_file.close()
    ### END MODIFIED ###


# --- Standalone Execution Block ---
if __name__ == '__main__':
    ### MODIFIED ###
    parser = argparse.ArgumentParser(description="Create a 10-bin balanced dataset (1:1:...:1) for regression.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your FULL, SKEWED combined feather file (e.g., brute_force_combined.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the new 'balanced_10bin_dataset.feather' and report.")
    # Removed the threshold arguments as they are no longer needed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling.")
    ### END MODIFIED ###

    args = parser.parse_args()
    main(args)