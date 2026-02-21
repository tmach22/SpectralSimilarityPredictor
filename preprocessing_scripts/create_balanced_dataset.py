import pandas as pd
import argparse
import os
import time
import numpy as np
import pyarrow.feather as feather
import pyarrow.ipc
import gc
from tqdm import tqdm

def main(args):
    print(f"--- Starting Memory-Efficient Balanced Dataset Creation ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "balanced_dataset_report.txt")
    output_file_path = os.path.join(args.output_dir, "balanced_10bin_spec_sim_dataset.feather")
    
    summary_file = open(report_path, "w")
    start_time = time.time()

    # =========================================================================
    # PASS 1: LIGHTWEIGHT SCAN (Load only Similarity Column)
    # =========================================================================
    print(f"PASS 1: Loading ONLY 'cosine_similarity' column to determine splits...")
    print(f"Input: {args.input_file}")

    # Open the file object to allow batch reading later
    source = pyarrow.memory_map(args.input_file, 'r')
    reader = pyarrow.ipc.RecordBatchFileReader(source)
    total_file_rows = reader.stats['num_rows'] if 'num_rows' in reader.stats else None
    
    # If using only a part of the file
    if args.max_rows and total_file_rows:
        limit_rows = min(int(args.max_rows), total_file_rows)
        print(f"   -> Limiting scan to first {limit_rows:,} rows (Part of file).")
    else:
        limit_rows = None

    # Load just the similarity column
    # PyArrow allows us to read specific columns without loading the whole table
    if limit_rows:
        # If limiting, we read the table partially (this might still be heavy if not careful, 
        # so we trust the OS paging, or we read the sim col fully and slice it)
        sim_table = feather.read_table(args.input_file, columns=['cosine_similarity'])
        df_sim = sim_table.to_pandas().iloc[:limit_rows]
    else:
        sim_table = feather.read_table(args.input_file, columns=['cosine_similarity'])
        df_sim = sim_table.to_pandas()
    
    # Free Arrow memory
    del sim_table
    gc.collect()

    print(f"   -> Analyzed {len(df_sim):,} similarity scores.")

    # =========================================================================
    # LOGIC: Define Bins & Sample Indices
    # =========================================================================
    print(f"\nCalculating Bins & Sample Indices...")
    
    # Define bins
    bins_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", 
              "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    
    # Assign bins
    df_sim['bin'] = pd.cut(df_sim['cosine_similarity'], bins=bins_edges, labels=labels, right=False)
    
    # Count
    counts = df_sim['bin'].value_counts()
    anchor_bin_label = "0.9-1.0"
    n_sample_size = counts.get(anchor_bin_label, 0)

    if n_sample_size == 0:
        print(f"FATAL: Anchor bin {anchor_bin_label} is empty. Cannot balance.")
        return

    print(f"   -> Anchor Bin '{anchor_bin_label}' count: {n_sample_size:,}")
    print(f"   -> Target Dataset Size: ~{n_sample_size * 10:,} pairs")

    # Select INDICES to keep
    indices_to_keep = []
    
    for label in labels:
        # Get indices for this bin
        bin_indices = df_sim[df_sim['bin'] == label].index.values
        n_bin = len(bin_indices)
        
        if n_bin == 0:
            print(f"   Warning: Bin {label} is empty.")
            continue
            
        if label == anchor_bin_label:
            # Keep all anchor
            selected = bin_indices
        elif n_bin < n_sample_size:
            # Oversample (Repeat indices)
            selected = np.random.choice(bin_indices, size=n_sample_size, replace=True)
        else:
            # Undersample
            selected = np.random.choice(bin_indices, size=n_sample_size, replace=False)
            
        indices_to_keep.append(selected)

    # Flatten array of indices
    all_target_indices = np.concatenate(indices_to_keep)
    
    # To use efficient batch filtering, we need a way to check membership fast.
    # However, since we might have duplicates (oversampling), we need to handle that.
    # Strategy: 
    # 1. We identify UNIQUE rows we need to load from disk.
    # 2. We load them.
    # 3. We reconstruct the balanced dataset (including duplicates) in memory at the end.
    
    unique_indices_to_load = np.unique(all_target_indices)
    unique_indices_set = set(unique_indices_to_load) # For O(1) lookup
    
    print(f"   -> Identified {len(unique_indices_to_load):,} unique rows to load from disk.")
    
    # Cleanup memory
    del df_sim
    del indices_to_keep
    gc.collect()

    # =========================================================================
    # PASS 2: BATCH EXTRACTION (Heavy Lifting)
    # =========================================================================
    print(f"\nPASS 2: iterating batches to extract data...")
    
    accumulated_dfs = []
    current_row_idx = 0
    num_batches = reader.num_record_batches
    
    for i in tqdm(range(num_batches), desc="Reading Batches"):
        # Read one batch (low memory)
        batch = reader.get_batch(i)
        batch_len = batch.num_rows
        
        # Calculate global row indices for this batch
        batch_indices = range(current_row_idx, current_row_idx + batch_len)
        
        # Check if ANY index in this batch is in our target set
        # (Intersection check)
        batch_indices_set = set(batch_indices)
        if batch_indices_set.isdisjoint(unique_indices_set):
            # Optimization: Skip converting to pandas if no rows needed
            current_row_idx += batch_len
            del batch
            continue
            
        # Convert to pandas
        df_batch = batch.to_pandas()
        
        # Create a mapping column to filter
        df_batch['global_idx'] = np.arange(current_row_idx, current_row_idx + batch_len)
        
        # Filter: Keep only rows present in our unique target list
        df_filtered = df_batch[df_batch['global_idx'].isin(unique_indices_set)].copy()
        
        if not df_filtered.empty:
            # Drop the helper column to save RAM
            df_filtered.set_index('global_idx', inplace=True)
            accumulated_dfs.append(df_filtered)
        
        current_row_idx += batch_len
        
        # Explicit Memory Cleanup
        del batch
        del df_batch
        del df_filtered
        
        # If we passed the limit, stop reading
        if limit_rows and current_row_idx >= limit_rows:
            break

    # =========================================================================
    # RECONSTRUCTION & SAVING
    # =========================================================================
    print(f"\nReconstructing balanced dataset...")
    
    # 1. Concat unique loaded rows
    df_unique_pool = pd.concat(accumulated_dfs)
    del accumulated_dfs
    gc.collect()
    
    # 2. Re-assemble the balanced set (handling the Oversampling duplicates)
    # We map the global indices back to the rows we loaded
    print("Applying oversampling and shuffling...")
    
    # df_unique_pool is indexed by 'global_idx'. 
    # We can perform a loc lookup using the list of target indices.
    # This automatically handles the duplication for oversampling.
    df_final = df_unique_pool.loc[all_target_indices].reset_index(drop=True)
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    print(f"Final Dataset Shape: {df_final.shape}")
    
    # Save
    print(f"Saving to {output_file_path}...")
    df_final.to_feather(output_file_path)
    
    # Report
    print(f"\nDone! Time taken: {time.time() - start_time:.2f}s")
    summary_file.write(f"Source: {args.input_file}\n")
    summary_file.write(f"Rows Scanned: {current_row_idx}\n")
    summary_file.write(f"Final Count: {len(df_final)}\n")
    summary_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to large feather file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional: Only use the first N rows of the file to save memory/time.")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)