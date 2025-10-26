import pandas as pd
import argparse
import time
from pathlib import Path
import numpy as np

import os
import sys
cwd = Path.cwd()

utils_dir = os.path.join(cwd, 'utils')
print(f"Adding {utils_dir} to sys.path")
sys.path.insert(0, utils_dir)

try:
    # Import our helper functions
    from eda_helper import plot_distributions, calculate_and_log_metrics
except ImportError:
    print("\n--- FATAL ERROR ---")
    print("Could not import 'eda_utils.py'.")
    print(f"Please make sure 'eda_utils.py' is in the same directory as this script:")
    print(f"{script_dir}")
    print("-------------------\n")
    sys.exit(1)
# -----------------------------------------------------------

def main(args):
    """
    Main function to analyze the bimodal (Low vs. High Cosine) split
    by loading the *full* taxonomy file and filtering first.
    """
    print(f"--- Starting Bimodal Split EDA ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "Bimodal_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    summary_path = os.path.join(args.output_dir, "Bimodal_report.txt")
    
    try:
        summary_file = open(summary_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file {summary_path} for writing. Error: {e}")
        return

    print(f"Loading data from: {args.input_file}")
    start_time = time.time()
    try:
        # Load the FULL taxonomy file
        all_data = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        summary_file.close()
        return
        
    print(f"Loaded {len(all_data):,} total records in {time.time() - start_time:.2f}s")
    
    # -----------------------------------------------------------------
    # --- THIS IS THE FIX YOU SUGGESTED ---
    # Filter for only the 'Different-Structure' pairs first
    print("Filtering for 'Different-Structure' pairs...")
    l1_data = all_data.query("taxonomy_L1 == 'Different-Structure'").copy()
    print(f"Found {len(l1_data):,} 'Different-Structure' pairs to analyze.")
    # Now the rest of the script works on 'l1_data'
    # -----------------------------------------------------------------

    
    # --- 2. Bimodal Analysis ---
    threshold = 0.8
    print(f"\nSplitting 'Different-Structure' data on cosine_similarity = {threshold}")

    # Group 1: Low-Cosine ("Haystack")
    df_low_cosine = l1_data.query("cosine_similarity < @threshold").copy()
    taxon_name_low = f"Bimodal: Low-Cosine (Cosine < {threshold})"
    calculate_and_log_metrics(df_low_cosine, taxon_name_low, summary_file, is_inter=True)
    plot_distributions(df_low_cosine, taxon_name_low, plot_dir, is_inter=True)
    
    # Group 2: High-Cosine ("Isomer/Analog Spike")
    df_high_cosine = l1_data.query("cosine_similarity >= @threshold").copy()
    taxon_name_high = f"Bimodal: High-Cosine (Cosine >= {threshold})"
    calculate_and_log_metrics(df_high_cosine, taxon_name_high, summary_file, is_inter=True)
    plot_distributions(df_high_cosine, taxon_name_high, plot_dir, is_inter=True)

    # --- 3. Deeper Dive: Are the High-Cosine pairs isomers? ---
    print("\n--- Deeper Dive on High-Cosine Group ---")
    summary_file.write("\n\n--- Deeper Dive on High-Cosine Group ---\n")
    
    # Check if formulas or parent masses are identical
    if not df_high_cosine.empty:
        # Check for matching formulas
        try:
            # Fill NaNs in formula columns *locally* for this comparison
            formula_main_clean = df_high_cosine['formula_main'].fillna('NO_FORMULA_1')
            formula_sub_clean = df_high_cosine['formula_sub'].fillna('NO_FORMULA_2')
            
            formula_match = (formula_main_clean == formula_sub_clean)
            formula_match_pct = formula_match.mean() * 100
            report = f"  Formula Match %: {formula_match_pct:.2f}% of high-cosine pairs have identical formulas.\n"
            print(report)
            summary_file.write(report)
        except Exception as e:
            report = f"  Could not compare formulas: {e}\n"
            print(report)
            summary_file.write(report)
            
        # Check for matching parent masses (as a fallback)
        try:
            # Fill NaNs with non-matching values
            mass_main_clean = df_high_cosine['parent_mass_main'].fillna(-1.0)
            mass_sub_clean = df_high_cosine['parent_mass_sub'].fillna(-2.0)
            
            # Use np.isclose for robust float comparison
            mass_match = np.isclose(mass_main_clean, mass_sub_clean, atol=0.0001)
            mass_match_pct = mass_match.mean() * 100
            report = f"  Parent Mass Match %: {mass_match_pct:.2f}% of high-cosine pairs have parent mass within 0.0001 Da.\n"
            print(report)
            summary_file.write(report)
        except Exception as e:
            report = f"  Could not compare parent masses: {e}\n"
            print(report)
            summary_file.write(report)
    else:
        report = "  High-cosine group is empty, skipping deeper dive.\n"
        print(report)
        summary_file.write(report)

    # --- 4. Cleanup ---
    summary_file.close()
    print(f"\n--- Bimodal EDA Complete ---")
    print(f"Summary report saved to: {summary_path}")
    print(f"Plots saved to directory: {plot_dir}")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Bimodal (Low vs. High Cosine) EDA on 'Different-Structure' data.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your FULL taxonomy file (e.g., taxanomy_file.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the bimodal text report and plots.")
    
    args = parser.parse_args()

    print(f"Running Bimodal EDA with arguments: {args}")
    main(args)