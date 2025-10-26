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

# Import our helper functions
from eda_helper import plot_distributions, calculate_and_log_metrics

def main(args):
    """
    Main function to run L0 and L1 analysis using the
    new InChIKey-based taxonomy.
    """
    print(f"--- Starting L0 and L1 EDA (v2, InChIKey-based) ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "L1_plots_v2")
    os.makedirs(plot_dir, exist_ok=True)
    
    summary_path = os.path.join(args.output_dir, "L1_report_v2.txt")
    
    # Open summary file in 'w' (write) mode to create a new report
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
        
    print(f"Loaded {len(all_data):,} records in {time.time() - start_time:.2f}s")
    
    # --- 2. L0 Analysis (All Data) ---
    taxon_name_l0 = "L0: All Data"
    calculate_and_log_metrics(all_data, taxon_name_l0, summary_file, is_inter=True)
    plot_distributions(all_data, taxon_name_l0, plot_dir, is_inter=True)

    # --- 3. L1 Analysis (NEW Molecular Identity) ---
    
    # L1.1: Identical-Molecule (Our true "Intra-Molecule")
    print("\nAnalyzing L1: Identical-Molecule...")
    df_l1_identical = all_data.query("taxonomy_L1 == 'Identical-Molecule'").copy()
    taxon_name_l1_identical = "L1: Identical-Molecule"
    # is_inter=False because Tanimoto is meaningless (should be 1.0)
    calculate_and_log_metrics(df_l1_identical, taxon_name_l1_identical, summary_file, is_inter=False)
    plot_distributions(df_l1_identical, taxon_name_l1_identical, plot_dir, is_inter=False)
    
    # L1.2: Stereoisomer
    print("\nAnalyzing L1: Stereoisomer...")
    df_l1_stereo = all_data.query("taxonomy_L1 == 'Stereoisomer'").copy()
    taxon_name_l1_stereo = "L1: Stereoisomer"
    # is_inter=True because they are different molecules, and Tanimoto vs Cosine is interesting
    calculate_and_log_metrics(df_l1_stereo, taxon_name_l1_stereo, summary_file, is_inter=True)
    plot_distributions(df_l1_stereo, taxon_name_l1_stereo, plot_dir, is_inter=True)

    # L1.3: Different-Structure (Our true "Inter-Molecule")
    print("\nAnalyzing L1: Different-Structure...")
    df_l1_different = all_data.query("taxonomy_L1 == 'Different-Structure'").copy()
    taxon_name_l1_different = "L1: Different-Structure"
    # is_inter=True, this is our main "signal" group
    calculate_and_log_metrics(df_l1_different, taxon_name_l1_different, summary_file, is_inter=True)
    plot_distributions(df_l1_different, taxon_name_l1_different, plot_dir, is_inter=True)

    # L1.4: Unknown
    print("\nAnalyzing L1: Unknown/NoInChI...")
    df_l1_unknown = all_data.query("taxonomy_L1 == 'Unknown/NoInChI'").copy()
    if len(df_l1_unknown) > 0:
        taxon_name_l1_unknown = "L1: Unknown/NoInChI"
        calculate_and_log_metrics(df_l1_unknown, taxon_name_l1_unknown, summary_file, is_inter=True)
        plot_distributions(df_l1_unknown, taxon_name_l1_unknown, plot_dir, is_inter=True)
    else:
        print("No 'Unknown/NoInChI' records found. Skipping.")

    # --- 4. Cleanup ---
    summary_file.close()
    print(f"\n--- L0 & L1 EDA (v2) Complete ---")
    print(f"Summary report saved to: {summary_path}")
    print(f"Plots saved to directory: {plot_dir}")
    
    # Save intermediate files for L2
    print("Saving L1 splits for next step...")
    df_l1_identical.reset_index(drop=True).to_feather(os.path.join(args.output_dir, "L1_identical_molecule.feather"))
    df_l1_stereo.reset_index(drop=True).to_feather(os.path.join(args.output_dir, "L1_stereoisomer.feather"))
    df_l1_different.reset_index(drop=True).to_feather(os.path.join(args.output_dir, "L1_different_structure.feather"))
    print("Saved intermediate files.")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run L0 and L1 EDA (v2, InChIKey-based) on Paired Spectra Data.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your *new* v2 taxonomy file (e.g., taxanomy_file.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the L1 v2 text report and plots.")
    
    args = parser.parse_args()
    main(args)