# import pandas as pd
# import argparse
# import os
# import time
# import numpy as np

# def main(args):
#     """
#     Main function to analyze the final combined dataset by
#     binning it into a 2D Tanimoto vs. Cosine matrix of counts.
#     """
#     print(f"--- Starting 2D Binning Analysis ---")
    
#     # --- 1. Setup ---
#     os.makedirs(args.output_dir, exist_ok=True)
#     report_path_1d = os.path.join(args.output_dir, "cosine_bin_counts.csv")
#     report_path_2d = os.path.join(args.output_dir, "tanimoto_cosine_2D_counts.csv")

#     print(f"Loading data from: {args.input_file}")
#     start_time = time.time()
#     try:
#         df = pd.read_feather(args.input_file)
#     except Exception as e:
#         print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
#         return
        
#     print(f"Loaded {len(df):,} records in {time.time() - start_time:.2f}s")
    
#     # --- 2. Create Bins ---
#     print(f"Creating {args.num_bins}x{args.num_bins} bins...")
    
#     # Define the bin edges
#     bins = np.linspace(0, 1, args.num_bins + 1)
    
#     # Create labels for the bins (e.g., "[0.0-0.1)", "[0.1-0.2)")
#     labels = [f"[{bins[i]:.1f}-{bins[i+1]:.1f})" for i in range(args.num_bins)]
    
#     # Use pandas.cut to assign each row to a bin
#     df['cosine_bin'] = pd.cut(df['cosine_similarity'], bins=bins, labels=labels, right=False, include_lowest=True)
#     df['tanimoto_bin'] = pd.cut(df['tanimoto'], bins=bins, labels=labels, right=False, include_lowest=True)

#     # --- 3. Analyze 1D Cosine Distribution (Your Original Request) ---
#     print("\nCalculating 1D Cosine Bin Counts...")
#     cosine_counts = df.groupby('cosine_bin').size()
#     cosine_counts.name = "count"
    
#     print(cosine_counts)
#     try:
#         cosine_counts.to_csv(report_path_1d, header=True)
#         print(f"Successfully saved 1D Cosine counts to: {report_path_1d}")
#     except Exception as e:
#         print(f"Error saving 1D report: {e}")

#     # --- 4. Analyze 2D Tanimoto vs. Cosine Distribution ---
#     print("\nCalculating 2D Tanimoto vs. Cosine Bin Counts...")
#     # Group by both bins and count the size of each group
#     grouped_counts = df.groupby(['tanimoto_bin', 'cosine_bin']).size()
    
#     # Convert from a multi-index Series to a 2D pivot table
#     heatmap_table = grouped_counts.unstack(level='cosine_bin', fill_value=0)
    
#     print(heatmap_table)
    
#     try:
#         heatmap_table.to_csv(report_path_2d)
#         print(f"Successfully saved 2D Heatmap counts to: {report_path_2d}")
#     except Exception as e:
#         print(f"Error saving 2D report: {e}")

#     print("\n--- Analysis Complete ---")

# # --- Standalone Execution Block ---
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Analyze the final dataset with 2D binning.")
#     parser.add_argument("--input_file", type=str, required=True,
#                         help="Path to your final combined feather file (e.g., brute_force_combined.feather).")
#     parser.add_argument("--output_dir", type=str, required=True,
#                         help="Directory to save the CSV reports.")
#     parser.add_argument("--num_bins", type=int, default=10,
#                         help="Number of bins for each dimension (e.g., 10 creates 10x10 grid).")

#     args = parser.parse_args()
#     main(args)

import pandas as pd

df = pd.read_csv('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/results/eda/new_dataset/brute_force/2d_plots/cosine_bin_counts.csv')
print(df)