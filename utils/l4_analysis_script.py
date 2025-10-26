import pandas as pd
import argparse
import os
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
    # We only need metrics for this report
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
    Main function to run a surgical L4 (Collision Energy) analysis
    inside our bimodal groups.
    """
    print(f"--- Starting Surgical L4 (Collision Energy) EDA ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "Surgical_L4_report.txt")
    
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
    
    # Filter for only the 'Different-Structure', 'Same-Instrument', 'Same-Adduct'
    # This is based on our last report, which showed all data fits this
    print("Filtering for 'Different-Structure', 'Same-Instrument', 'Same-Adduct' pairs...")
    l3_data = all_data.query(
        "taxonomy_L1 == 'Different-Structure' and "
        "taxonomy_L2 == 'Same-Instrument' and "
        "taxonomy_L3 == 'Same-Adduct'"
    ).copy()
    print(f"Found {len(l3_data):,} pairs to analyze.")

    # --- 2. Bimodal Split ---
    threshold = 0.8 # Based on your last report
    print(f"Splitting data on cosine_similarity = {threshold}")
    
    df_low_cosine = l3_data.query("cosine_similarity < @threshold").copy()
    df_high_cosine = l3_data.query("cosine_similarity >= @threshold").copy()

    # --- 3. Run Surgical L4 Analysis ---
    
    bimodal_groups = [
        ("Bimodal: Low-Cosine (Cosine < 0.8)", df_low_cosine),
        ("Bimodal: High-Cosine (Cosine >= 0.8)", df_high_cosine)
    ]

    for bimodal_name, bimodal_df in bimodal_groups:
        
        # Log metrics for the top-level bimodal group
        calculate_and_log_metrics(bimodal_df, bimodal_name, summary_file, is_inter=True)
        
        # --- L4 Split (Collision Energy) ---
        df_l4_same_energy = bimodal_df.query("taxonomy_L4 == 'Same-Energy'").copy()
        df_l4_cross_energy = bimodal_df.query("taxonomy_L4 == 'Cross-Energy'").copy()
        
        l4_groups = [
            (f"{bimodal_name} → Same-Energy", df_l4_same_energy),
            (f"{bimodal_name} → Cross-Energy", df_l4_cross_energy)
        ]
        
        for l4_name, l4_df in l4_groups:
            # Log metrics for the L4 split
            calculate_and_log_metrics(l4_df, l4_name, summary_file, is_inter=True)

    # --- 4. Cleanup ---
    summary_file.close()
    print(f"\n--- Surgical L4 EDA Complete ---")
    print(f"Summary report saved to: {summary_path}")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Surgical L4 (Energy) EDA inside Bimodal splits.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your FULL taxonomy file (e.g., taxanomy_file.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the surgical text report.")
    
    args = parser.parse_args()
    main(args)