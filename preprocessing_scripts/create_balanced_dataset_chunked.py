import pandas as pd
import argparse
import os
import time
import glob
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- WORKER FUNCTIONS ---

def get_bin_indices(sims):
    """Vectorized binning helper."""
    bins_edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    indices = np.digitize(sims, bins_edges) - 1
    return np.clip(indices, 0, 9)

def worker_count_bins(file_path):
    """
    PASS 1 WORKER: Opens one file, counts bins, returns stats.
    Returns: (counts_array, total_rows)
    """
    try:
        # Load only similarity column
        df_chunk = pd.read_feather(file_path, columns=['cosine_similarity'])
        sims = df_chunk['cosine_similarity'].to_numpy()
        
        indices = get_bin_indices(sims)
        counts = np.bincount(indices, minlength=10)
        
        return counts, len(sims)
    except Exception as e:
        # Return zeros on failure so main process continues
        print(f"Error in Pass 1 (Count) on {os.path.basename(file_path)}: {e}")
        return np.zeros(10, dtype=int), 0

def worker_extract_data(args):
    """
    PASS 2 WORKER: Opens one file, extracts rows based on global ratios.
    Returns: DataFrame (subset) or None
    """
    file_path, ratios, seed = args
    
    try:
        # Load full file
        df_chunk = pd.read_feather(file_path)
        sims = df_chunk['cosine_similarity'].to_numpy()
        indices = get_bin_indices(sims)
        
        chunk_keepers = []
        
        # Iterate bins 0-9
        for bin_idx in range(10):
            ratio = ratios[bin_idx]
            if ratio <= 0: continue
                
            mask = (indices == bin_idx)
            if not np.any(mask): continue
            
            df_bin_subset = df_chunk[mask]
            n_available = len(df_bin_subset)
            
            # Sampling Logic
            if ratio >= 1.0:
                # OVERSAMPLING
                n_copies = int(ratio)
                prob_extra = ratio - n_copies
                
                keepers = [df_bin_subset] * n_copies
                
                if prob_extra > 0:
                    n_extra = int(np.round(n_available * prob_extra))
                    if n_extra > 0:
                        # Use deterministic seed + bin_idx to ensure reproducibility
                        local_seed = (seed + bin_idx) % (2**32)
                        keepers.append(df_bin_subset.sample(n=n_extra, replace=False, random_state=local_seed))
                
                chunk_keepers.append(pd.concat(keepers))
                
            else:
                # UNDERSAMPLING
                n_take = int(np.round(n_available * ratio))
                if n_take > 0:
                    local_seed = (seed + bin_idx) % (2**32)
                    chunk_keepers.append(df_bin_subset.sample(n=n_take, replace=False, random_state=local_seed))

        if chunk_keepers:
            return pd.concat(chunk_keepers)
            
    except Exception as e:
        print(f"Error in Pass 2 (Extract) on {os.path.basename(file_path)}: {e}")
    
    return None

# --- MAIN CONTROLLER ---

def main(args):
    print(f"--- Parallel 10-Bin Balancing ({args.n_cores} Cores) ---")
    
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "balanced_dataset_report.txt")
    output_file_path = os.path.join(args.output_dir, "balanced_10bin_spec_sim_dataset.feather")
    summary_file = open(report_path, "w")
    
    search_pattern = os.path.join(args.input_dir, "*.feather")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files.")
    start_time = time.time()

    # =========================================================================
    # PASS 1: PARALLEL COUNTING
    # =========================================================================
    print(f"\nPASS 1: Counting Bins in Parallel...")
    
    global_bin_counts = np.zeros(10, dtype=int)
    total_pairs_scanned = 0
    
    with ProcessPoolExecutor(max_workers=args.n_cores) as executor:
        # Submit all tasks
        futures = [executor.submit(worker_count_bins, f) for f in files]
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Counting"):
            counts, n_rows = future.result()
            global_bin_counts += counts
            total_pairs_scanned += n_rows

    print(f"\nGlobal Statistics (n={total_pairs_scanned:,}):")
    labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", 
              "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    
    for i, label in enumerate(labels):
        print(f"  {label}: {global_bin_counts[i]:,}")

    # =========================================================================
    # CALCULATE RATIOS
    # =========================================================================
    anchor_idx = 9 # 0.9-1.0
    target_size = global_bin_counts[anchor_idx]
    
    if target_size == 0:
        print("FATAL: Anchor bin empty.")
        return

    print(f"\nTarget Size (Anchor): {target_size:,}")
    
    ratios = np.zeros(10, dtype=float)
    for i in range(10):
        if global_bin_counts[i] > 0:
            ratios[i] = target_size / global_bin_counts[i]
            
    # =========================================================================
    # PASS 2: PARALLEL EXTRACTION
    # =========================================================================
    print(f"\nPASS 2: Extracting Data in Parallel...")
    
    accumulated_dfs = []
    
    # Prepare arguments: (file, ratios, seed_offset)
    # We change seed slightly per file to avoid identical random sampling patterns across files
    tasks = [(f, ratios, args.seed + i) for i, f in enumerate(files)]
    
    with ProcessPoolExecutor(max_workers=args.n_cores) as executor:
        futures = [executor.submit(worker_extract_data, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Extracting"):
            df_result = future.result()
            if df_result is not None:
                accumulated_dfs.append(df_result)

    # =========================================================================
    # MERGE & SAVE
    # =========================================================================
    print("\nMerging results...")
    if not accumulated_dfs:
        print("Error: No data extracted.")
        return

    df_final = pd.concat(accumulated_dfs, ignore_index=True)
    del accumulated_dfs
    gc.collect()
    
    print("Shuffling...")
    df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    print(f"Final Shape: {df_final.shape}")
    print(f"Saving to {output_file_path}...")
    df_final.to_feather(output_file_path)
    
    # Final Report
    report = f"Total Original: {total_pairs_scanned:,}\n"
    report += f"Final Balanced: {len(df_final):,}\n"
    report += "Final Distribution:\n"
    
    # Quick distribution check using cut
    df_final['temp_bin'] = pd.cut(df_final['cosine_similarity'], 
                                  bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], 
                                  labels=labels, right=False)
    report += f"{df_final['temp_bin'].value_counts().sort_index()}"
    
    summary_file.write(report)
    summary_file.close()
    
    print(f"Done in {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)