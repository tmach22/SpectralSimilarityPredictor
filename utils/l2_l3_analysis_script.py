import pandas as pd
import argparse
import time
from pathlib import Path

import os
import sys
cwd = Path.cwd()

utils_dir = os.path.join(cwd, 'utils')
print(f"Adding {utils_dir} to sys.path")
sys.path.insert(0, utils_dir)

try:
    from eda_helper import calculate_and_log_metrics
    # We don't need plotting for this, just the metrics
except ImportError:
    print("\n--- FATAL ERROR ---")
    print("Could not import 'eda_utils.py'.")
    print(f"Please make sure 'eda_utils.py' is in the same directory as this script:")
    print(f"{utils_dir}")
    print("-------------------\n")
    sys.exit(1)
# -----------------------------------------------------------

def main(args):
    """
    Main function to run a surgical L2 (Instrument) and L3 (Adduct)
    analysis inside our new 'Low-Cosine' and 'High-Cosine' groups.
    """
    print(f"--- Starting Surgical L2/L3 EDA ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "Surgical_L2_L3_report.txt")
    
    try:
        summary_file = open(summary_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file {summary_path} for writing. Error: {e}")
        return

    print(f"Loading data from: {args.input_file}")
    start_time = time.time()
    try:
        all_data = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        summary_file.close()
        return
    print(f"Loaded {len(all_data):,} total records in {time.time() - start_time:.2f}s")
    
    # Filter for only the 'Different-Structure' pairs first
    print("Filtering for 'Different-Structure' pairs...")
    l1_data = all_data.query("taxonomy_L1 == 'Different-Structure'").copy()
    print(f"Found {len(l1_data):,} 'Different-Structure' pairs to analyze.")

    # --- 2. Bimodal Split ---
    threshold = 0.8 # Based on your last report
    print(f"Splitting data on cosine_similarity = {threshold}")
    
    df_low_cosine = l1_data.query("cosine_similarity < @threshold").copy()
    df_high_cosine = l1_data.query("cosine_similarity >= @threshold").copy()

    # --- 3. Run Surgical Analysis ---
    
    # Define the two main groups we will iterate through
    bimodal_groups = [
        ("Bimodal: Low-Cosine (Cosine < 0.8)", df_low_cosine),
        ("Bimodal: High-Cosine (Cosine >= 0.8)", df_high_cosine)
    ]

    # This loop will run the L2 and L3 splits *inside* each bimodal group
    for bimodal_name, bimodal_df in bimodal_groups:
        
        # Log metrics for the top-level bimodal group
        calculate_and_log_metrics(bimodal_df, bimodal_name, summary_file, is_inter=True)
        
        # --- L2 Split (Instrument) ---
        df_l2_same_inst = bimodal_df.query("taxonomy_L2 == 'Same-Instrument'").copy()
        df_l2_cross_inst = bimodal_df.query("taxonomy_L2 == 'Cross-Instrument'").copy()
        
        l2_groups = [
            (f"{bimodal_name} → Same-Instrument", df_l2_same_inst),
            (f"{bimodal_name} → Cross-Instrument", df_l2_cross_inst)
        ]
        
        for l2_name, l2_df in l2_groups:
            # Log metrics for the L2 split
            calculate_and_log_metrics(l2_df, l2_name, summary_file, is_inter=True)
            
            # --- L3 Split (Adduct) ---
            df_l3_same_adduct = l2_df.query("taxonomy_L3 == 'Same-Adduct'").copy()
            df_l3_cross_adduct = l2_df.query("taxonomy_L3 == 'Cross-Adduct'").copy()
            
            l3_groups = [
                (f"{l2_name} → Same-Adduct", df_l3_same_adduct),
                (f"{l2_name} → Cross-Adduct", df_l3_cross_adduct)
            ]
            
            for l3_name, l3_df in l3_groups:
                # Log metrics for the L3 split
                calculate_and_log_metrics(l3_df, l3_name, summary_file, is_inter=True)

    # --- 4. Cleanup ---
    summary_file.close()
    print(f"\n--- Surgical L2/L3 EDA Complete ---")
    print(f"Summary report saved to: {summary_path}")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Surgical L2/L3 EDA inside Bimodal splits.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your FULL taxonomy file (e.g., taxanomy_file.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the surgical text report.")
    
    args = parser.parse_args()
    main(args)