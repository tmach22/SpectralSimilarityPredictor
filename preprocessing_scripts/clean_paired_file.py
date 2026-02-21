import pandas as pd
import os
import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- GLOBAL SHARED MEMORY ---
# This allows workers to read the set without duplicating it in RAM
VALID_IDS = set()

def load_valid_ids(spec_path):
    """Loads metadata and populates the global VALID_IDS set."""
    global VALID_IDS
    print(f"Loading Metadata from: {spec_path}")
    try:
        if spec_path.endswith('.pkl'):
            df_spec = pd.read_pickle(spec_path)
        else:
            df_spec = pd.read_feather(spec_path)
            
        # Detect ID column
        if 'spec_id' in df_spec.columns:
            id_col = 'spec_id'
        elif 'spectrum_id' in df_spec.columns:
            id_col = 'spectrum_id'
        else:
            raise ValueError("Metadata file must contain 'spec_id' or 'spectrum_id'.")
            
        VALID_IDS = set(df_spec[id_col].unique())
        print(f"Valid Spec IDs loaded: {len(VALID_IDS):,}")
        
    except Exception as e:
        print(f"CRITICAL ERROR loading metadata: {e}")
        sys.exit(1)

def process_file(args):
    """
    Worker function: Loads one file, filters it against VALID_IDS, and saves it.
    """
    file_path, output_dir = args
    filename = os.path.basename(file_path)
    save_path = os.path.join(output_dir, filename)
    
    try:
        # Load
        df = pd.read_feather(file_path)
        initial_count = len(df)
        
        if initial_count == 0:
            return 0, 0

        # Filter using Global Set
        mask = (df['name_main'].isin(VALID_IDS)) & (df['name_sub'].isin(VALID_IDS))
        df_clean = df[mask].reset_index(drop=True)
        final_count = len(df_clean)
        
        # Save only if we have data
        if final_count > 0:
            df_clean.to_feather(save_path)
            
        return initial_count, final_count

    except Exception as e:
        return f"Error in {filename}: {e}", 0

def clean_directory_parallel(args):
    print(f"--- Parallel Cleaning of Chunks ({args.n_cores} Cores) ---")
    
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Global Data
    load_valid_ids(args.spec_path)

    # 3. Scan Files
    search_pattern = os.path.join(args.input_dir, "*.feather")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No feather files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files. Starting workers...")

    # 4. Generate Tasks
    # We pass tuples of (file_path, output_dir)
    tasks = [(f, args.output_dir) for f in files]

    # 5. Parallel Execution
    total_initial = 0
    total_final = 0
    files_processed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.n_cores) as executor:
        # Submit all tasks
        futures = [executor.submit(process_file, t) for t in tasks]
        
        # Track progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning"):
            res = future.result()
            
            # Check if result is an error message (string) or stats (tuple)
            if isinstance(res[0], str):
                errors.append(res[0])
            else:
                initial, final = res
                total_initial += initial
                total_final += final
                files_processed += 1

    # 6. Final Report
    removed_count = total_initial - total_final
    percent_removed = (removed_count / total_initial * 100) if total_initial > 0 else 0
    
    print(f"\n=== PARALLEL CLEANING REPORT ===")
    print(f"Input Directory:     {args.input_dir}")
    print(f"Total Initial Pairs: {total_initial:,}")
    print(f"Total Clean Pairs:   {total_final:,}")
    print(f"Total Removed:       {removed_count:,} ({percent_removed:.2f}%)")
    print(f"Files Processed:     {files_processed}/{len(files)}")
    
    if errors:
        print(f"\n[WARNING] {len(errors)} files failed. First 5 errors:")
        for e in errors[:5]:
            print(f" - {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing raw chunk feather files")
    parser.add_argument("--output_dir", required=True, help="Directory to save cleaned chunks")
    parser.add_argument("--spec_path", required=True, help="Path to spec_df.pkl")
    parser.add_argument("--n_cores", type=int, default=8, help="Number of processor cores to use")
    
    args = parser.parse_args()
    clean_directory_parallel(args)