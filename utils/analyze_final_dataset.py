import pandas as pd
import argparse
import os
import time
from pathlib import Path

# --- This block helps Python find the 'eda_utils.py' file ---
import sys
try:
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
except NameError:
    # This handles cases where __file__ is not defined (e.g., in a notebook)
    script_dir = Path.cwd()
    
# Add this directory to the Python path
sys.path.append(str(script_dir))

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
    Main function to run a full analysis on the final,
    combined, brute-force dataset.
    """
    print(f"--- Starting Analysis of Final Combined Dataset ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    summary_path = os.path.join(args.output_dir, "final_dataset_report.txt")
    
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
    taxon_name = "Final Combined Dataset"
    # We set is_inter=True to get all stats, including Tanimoto and Correlation
    calculate_and_log_metrics(all_data, taxon_name, summary_file, is_inter=True)
    plot_distributions(all_data, taxon_name, plot_dir, is_inter=True)

    # --- 3. Cleanup ---
    summary_file.close()
    print(f"\n--- Analysis Complete ---")
    print(f"Summary report saved to: {summary_path}")
    print(f"Plots saved to directory: {plot_dir}")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the final combined (brute-force) dataset.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your final combined feather file (e.g., brute_force_combined.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the text report and plots.")
    
    args = parser.parse_args()
    main(args)