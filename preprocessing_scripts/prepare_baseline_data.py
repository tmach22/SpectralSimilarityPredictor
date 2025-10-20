import argparse
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Global variable for the fingerprint map ---
# This is defined globally so that worker processes in the pool can access it
# without needing to pickle and transfer it repeatedly.
fp_map = {}

def process_chunk(chunk_df):
    """
    Worker function to process a chunk of the pairs DataFrame.
    This function will be executed in parallel by multiple processes.
    """
    global fp_map
    feature_vectors = []
    
    for _, row in chunk_df.iterrows():
        fp_a = fp_map.get(row['name_main'])
        fp_b = fp_map.get(row['name_sub'])
        
        if fp_a is not None and fp_b is not None:
            # Combine fingerprints: absolute difference and concatenation
            combined_fp = np.concatenate([
                np.abs(fp_a.astype(np.int8) - fp_b.astype(np.int8)).astype(np.uint8),
                fp_a,
                fp_b
            ])
            feature_vectors.append(combined_fp)
            
    return feature_vectors

def create_feature_file_hdf5(pairs_path, data_path, output_path, group_name, subset_size=None, num_workers=4, fp_bits=2048):
    """
    Loads paired data, merges with fingerprints, and saves the final
    feature matrix (X) and target vector (y) to a compressed HDF5 file using parallel processing.
    """
    global fp_map
    
    print(f"--- Preparing data for group '{group_name}' in {output_path} ---")
    
    print(f"Loading pairs from {pairs_path}...")
    pairs_df = pd.read_feather(pairs_path)
    
    # --- NEW: Handle subsetting ---
    if subset_size:
        if 0 < subset_size < 1:
            num_samples = int(len(pairs_df) * subset_size)
            print(f"Using a random subset of {subset_size:.0%} ({num_samples} pairs)...")
            pairs_df = pairs_df.sample(n=num_samples, random_state=42)
        elif subset_size >= 1:
            num_samples = int(subset_size)
            print(f"Using a random subset of {num_samples} pairs...")
            pairs_df = pairs_df.sample(n=num_samples, random_state=42)

    print(f"Loading data with fingerprints from {data_path}...")
    main_df = pd.read_feather(data_path)
    
    if 'morgan_fingerprint' not in main_df.columns:
        raise ValueError("morgan_fingerprint column not found in the data file.")

    # Create a dictionary for fast fingerprint lookups and set it as a global variable
    fp_map = main_df.set_index('spectrum_id')['morgan_fingerprint'].to_dict()

    # Filter pairs to ensure both molecules have a valid fingerprint
    initial_pair_count = len(pairs_df)
    pairs_df = pairs_df[pairs_df['name_main'].isin(fp_map) & pairs_df['name_sub'].isin(fp_map)].copy()
    print(f"Filtered pairs: {len(pairs_df)} of {initial_pair_count} remain after checking for valid fingerprints.")

    num_pairs = len(pairs_df)
    feature_dim = fp_bits * 3

    with h5py.File(output_path, 'a') as hf:
        if group_name in hf:
            del hf[group_name]
        group = hf.create_group(group_name)
        
        print(f"Creating HDF5 dataset 'X' with shape ({num_pairs}, {feature_dim})")
        dset_x = group.create_dataset('X', shape=(num_pairs, feature_dim), dtype=np.uint8, chunks=(1024, feature_dim), compression='gzip')
        dset_y = group.create_dataset('y', shape=(num_pairs,), dtype=np.float32)
        
        y = pairs_df['cosine_similarity'].values
        dset_y[:] = y
        
        # --- NEW: Parallel Processing ---
        print(f"Constructing feature vectors with {num_workers} workers...")
        
        # Split the DataFrame into chunks for the workers
        chunk_size = int(np.ceil(len(pairs_df) / num_workers))
        chunks = [pairs_df.iloc[i:i + chunk_size] for i in range(0, len(pairs_df), chunk_size)]
        
        write_index = 0
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for progress bar and efficient processing
            with tqdm(total=num_pairs, desc=f"Processing {group_name} pairs") as pbar:
                for result_chunk in pool.imap_unordered(process_chunk, chunks):
                    if result_chunk:
                        dset_x[write_index : write_index + len(result_chunk)] = np.array(result_chunk, dtype=np.uint8)
                        write_index += len(result_chunk)
                    pbar.update(len(result_chunk))
    
    print(f"--- Data preparation for '{group_name}' complete ---")

def main():
    parser = argparse.ArgumentParser(description="Prepare fingerprint data for baseline model training using HDF5.")
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the feather file with fingerprints.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the single HDF5 output file.")
    
    # --- NEW Arguments ---
    parser.add_argument("--subset_size", type=float, default=None, help="Use a subset of the data. E.g., 0.1 for 10% or 10000 for 10k pairs.")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of CPU cores to use for parallel processing.")

    args = parser.parse_args()

    # Process training data
    create_feature_file_hdf5(
        args.train_pairs_path, args.data_path, args.output_path, 'train',
        subset_size=args.subset_size, num_workers=args.num_workers
    )

    # Process validation data
    create_feature_file_hdf5(
        args.val_pairs_path, args.data_path, args.output_path, 'validation',
        subset_size=args.subset_size, num_workers=args.num_workers
    )

    # Process test data
    create_feature_file_hdf5(
        args.test_pairs_path, args.data_path, args.output_path, 'test',
        subset_size=args.subset_size, num_workers=args.num_workers
    )

if __name__ == '__main__':
    main()