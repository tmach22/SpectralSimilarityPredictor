import pandas as pd
import argparse
import os
import time
import numpy as np

def main(args):
    """
    Creates a Balanced Binary Dataset with STRATIFIED sampling for the negative class.
    
    1. Positive Class (>= threshold): Keeps ALL available samples (or downsamples to match negative if needed).
    2. Negative Class (< threshold): Samples EQUALLY from 0.1-width bins to remove skew.
       (Uses oversampling with replacement for rare negative bins).
    """
    print(f"--- Starting Stratified Binary Dataset Creation (Threshold={args.threshold}) ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    safe_threshold_str = str(args.threshold).replace('.', '')
    report_path = os.path.join(args.output_dir, f"stratified_binary_{safe_threshold_str}_report.txt")
    output_file_path = os.path.join(args.output_dir, f"stratified_binary_{safe_threshold_str}_dataset.feather")
    
    try:
        summary_file = open(report_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file. Error: {e}")
        return

    print(f"Loading full dataset from: {args.input_file}")
    start_time = time.time()
    try:
        df = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file. Error: {e}")
        summary_file.close()
        return
    print(f"Loaded {len(df):,} total pairs in {time.time() - start_time:.2f}s")
    
    # --- 2. Separate High (Positive) Class ---
    # We treat the High class as the "Anchor" for size.
    df_high = df[df['cosine_similarity'] >= args.threshold].copy()
    n_high = len(df_high)
    print(f"\nPositive Class (>= {args.threshold}): Found {n_high:,} pairs.")
    
    if n_high == 0:
        print("Fatal Error: No positive pairs found.")
        return

    # --- 3. Stratify Negative Class ---
    print(f"\nStratifying Negative Class (< {args.threshold})...")
    
    # Generate bins: 0.0, 0.1, ... up to threshold
    # e.g. for 0.7: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    bin_edges = np.arange(0.0, args.threshold + 0.0001, 0.1)
    num_neg_bins = len(bin_edges) - 1
    
    # Calculate target per bin to achieve 50/50 balance with High class
    target_neg_total = n_high
    target_per_bin = int(target_neg_total / num_neg_bins)
    
    print(f"Target Negative Total: {target_neg_total:,}")
    print(f"Number of Negative Bins: {num_neg_bins}")
    print(f"Target Per Negative Bin: {target_per_bin:,}")
    
    sampled_neg_dfs = []
    
    for i in range(num_neg_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        
        # Query specific bin
        # Note: We use inclusive lower, exclusive upper, except for the last bin logic if needed
        query_mask = (df['cosine_similarity'] >= low) & (df['cosine_similarity'] < high)
        df_bin = df[query_mask]
        count = len(df_bin)
        
        bin_label = f"{low:.1f}-{high:.1f}"
        
        if count == 0:
            print(f"  [Warning] Bin {bin_label} is EMPTY. Skipping.")
            continue
            
        if count >= target_per_bin:
            # Undersample abundant bins (e.g., 0.0-0.1)
            print(f"  Bin {bin_label}: Found {count:,} -> Undersampling to {target_per_bin:,}")
            df_sampled = df_bin.sample(n=target_per_bin, random_state=args.seed, replace=False)
        else:
            # Oversample rare bins (e.g., 0.5-0.6)
            print(f"  Bin {bin_label}: Found {count:,} -> Oversampling to {target_per_bin:,}")
            df_sampled = df_bin.sample(n=target_per_bin, random_state=args.seed, replace=True)
            
        sampled_neg_dfs.append(df_sampled)
        
    if not sampled_neg_dfs:
        print("Fatal Error: No negative samples could be generated.")
        return

    df_neg_balanced = pd.concat(sampled_neg_dfs)
    
    # --- 4. Combine and Finalize ---
    # Note: We might have slightly fewer negatives due to integer division or empty bins,
    # or slightly more/less depending on rounding.
    # To ensure strict 50/50, we trim the larger class to match the smaller one.
    
    n_final_high = len(df_high)
    n_final_neg = len(df_neg_balanced)
    n_balanced = min(n_final_high, n_final_neg)
    
    print(f"\nBalancing Final Classes...")
    print(f"  High Available: {n_final_high:,}")
    print(f"  Low Generated:  {n_final_neg:,}")
    print(f"  Final Count Per Class: {n_balanced:,}")
    
    df_high = df_high.sample(n=n_balanced, random_state=args.seed)
    df_neg_balanced = df_neg_balanced.sample(n=n_balanced, random_state=args.seed)
    
    # Assign Labels
    df_high['label'] = 1.0
    df_neg_balanced['label'] = 0.0
    
    # Concat and Shuffle
    df_final = pd.concat([df_high, df_neg_balanced])
    df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # --- 5. Save and Report ---
    try:
        df_final.to_feather(output_file_path)
        print(f"\nSuccessfully saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return

    # Generate Report
    report = f"--- Stratified Binary Dataset Report ---\n"
    report += f"Source: {args.input_file}\n"
    report += f"Threshold: {args.threshold}\n"
    report += f"Final Total Pairs: {len(df_final):,}\n"
    report += f"  - Class 1 (>= {args.threshold}): {n_balanced:,}\n"
    report += f"  - Class 0 (< {args.threshold}):  {n_balanced:,} (Stratified)\n\n"
    
    report += "Class 0 Internal Distribution:\n"
    # Verify the stratification
    neg_only = df_final[df_final['label'] == 0.0]
    neg_bins = pd.cut(neg_only['cosine_similarity'], bins=bin_edges)
    report += f"{neg_bins.value_counts().sort_index()}\n"
    
    print(f"\n{report}")
    summary_file.write(report)
    summary_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a Stratified Binary Classification Dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to FULL feather file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for similarity.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    main(args)