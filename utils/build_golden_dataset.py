import pandas as pd
import argparse
import os
import time
from pathlib import Path

def main(args):
    """
    Cleans a full "final" taxonomy file by filtering for
    the 'High-Signal' and 'Clean-Signal' groups identified
    during our Exploratory Data Analysis (EDA).
    
    This version includes all L1-L4 filters for maximum robustness.
    """
    print(f"--- Starting 'Golden' Dataset Creation/Cleaning (Robust v2) ---")
    
    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = Path(args.input_file).stem
    output_file_name = f"golden_{base_name}.feather"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    summary_path = os.path.join(args.output_dir, f"golden_{base_name}_report.txt")

    try:
        summary_file = open(summary_path, "w")
    except Exception as e:
        print(f"Fatal Error: Could not open report file {summary_path} for writing. Error: {e}")
        return

    print(f"Loading full taxonomy file from: {args.input_file}")
    start_time = time.time()
    try:
        all_data = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        summary_file.close()
        return
    
    total_loaded = len(all_data)
    print(f"Loaded {total_loaded:,} total records in {time.time() - start_time:.2f}s")
    
    # --- 2. Define and Filter for Golden Datasets ---
    
    # Define our base filter (L1, L2, L3) based on EDA findings
    # All our "signal" data is in this group
    base_filter = (
        "(taxonomy_L1 == 'Different-Structure') and "
        "(taxonomy_L2 == 'Same-Instrument') and "
        "(taxonomy_L3 == 'Same-Adduct')"
    )
    
    # Rule 1: The "High-Signal" Analog Spike
    # Base Filter + High Cosine + Cross Energy
    print("Filtering for 'High-Signal' group (positives)...")
    filter_high_signal = (
        f"{base_filter} and "
        "(cosine_similarity >= 0.8) and "
        "(taxonomy_L4 == 'Cross-Energy')"
    )
    df_high_signal = all_data.query(filter_high_signal).copy()
    count_high = len(df_high_signal)
    print(f"Found {count_high:,} 'High-Signal' pairs.")

    # Rule 2: The "Clean-Signal" Haystack
    # Base Filter + Low Cosine + Same Energy
    print("Filtering for 'Clean-Signal' group (negatives)...")
    filter_clean_signal = (
        f"{base_filter} and "
        "(cosine_similarity < 0.8) and "
        "(taxonomy_L4 == 'Same-Energy')"
    )
    df_clean_signal = all_data.query(filter_clean_signal).copy()
    count_clean = len(df_clean_signal)
    print(f"Found {count_clean:,} 'Clean-Signal' pairs.")

    # --- 3. Combine and Save ---
    print("\nCombining 'High-Signal' and 'Clean-Signal' datasets...")
    df_golden = pd.concat([df_high_signal, df_clean_signal])
    total_count = len(df_golden)
    
    if total_count == 0:
        print("\n--- WARNING: No pairs found matching the criteria. Output file will be empty. ---")
    else:
        print(f"Total pairs in 'Golden' dataset: {total_count:,}")
    
    # Important: Reset index for clean saving to feather
    df_golden = df_golden.reset_index(drop=True)
    
    try:
        df_golden.to_feather(output_file_path)
        print(f"\nSuccessfully saved golden (clean) dataset to: {output_file_path}")
    except Exception as e:
        print(f"\nError saving golden dataset: {e}")
        summary_file.close()
        return

    # --- 4. Write Final Report ---
    report = f"--- Golden Dataset Report for {base_name} ---\n"
    report += f"Original (dirty) file pair count: {total_loaded:,}\n"
    report += f"Cleaned (golden) file pair count: {total_count:,}\n"
    
    if total_count > 0:
        report += f"  - 'High-Signal' (Positives): {count_high:,} ({count_high/total_count*100:.1f}%)\n"
        report += f"  - 'Clean-Signal' (Negatives): {count_clean:,} ({count_clean/total_count*100:.1f}%)\n"
    
    report += f"\nFiltered {total_loaded - total_count:,} 'noisy' or 'out-of-scope' pairs.\n"
    
    print(report)
    summary_file.write(report)
    summary_file.close()


# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build the 'Golden' dataset from EDA findings.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your FULL taxonomy file to be cleaned (e.g., validation_pairs_final.feather).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the new 'golden_...' feather file and report.")
    
    args = parser.parse_args()
    main(args)