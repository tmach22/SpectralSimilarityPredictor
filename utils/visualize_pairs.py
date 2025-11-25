import pandas as pd
import argparse
import os
import time
from rdkit import Chem
import random
import numpy as np # Added for np.random.seed
from tqdm import tqdm
import multiprocessing as mp # Added for parallelization
from functools import partial # Added for passing arguments to worker

def check_substructure_worker(pair_data):
    """
    Worker function for parallel substructure checking.
    Accepts a tuple: (id_main, smiles_main, id_sub, smiles_sub)
    Returns a dictionary with results or None on error.
    """
    id_main, smiles_main, id_sub, smiles_sub = pair_data
    try:
        mol_main = Chem.MolFromSmiles(smiles_main)
        mol_sub = Chem.MolFromSmiles(smiles_sub)

        if mol_main is None or mol_sub is None:
            return None # Indicate error/skip

        # Perform substructure check
        match_found = mol_main.HasSubstructMatch(mol_sub)

        return {
            'id_main': id_main,
            'smiles_main': smiles_main,
            'id_sub': id_sub,
            'smiles_sub': smiles_sub,
            'is_substructure': match_found
        }

    except Exception as e:
        # print(f"  Error processing pair ({id_main}, {id_sub}): {e}") # Optional: Reduce console noise
        return None # Indicate error/skip


def main():
    parser = argparse.ArgumentParser(description="Check for substructure relationships in a paired dataset (parallel) and save to CSV.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the paired feather file (e.g., taxanomy_file_v2.feather).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save the output CSV file.")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of pairs to sample and process. Set to 0 to process all pairs.")
    parser.add_argument("--sample_mode", type=str, default="random",
                        choices=["random", "high_cosine", "low_cosine"],
                        help="How to sample pairs if num_samples > 0: random, high_cosine (>=0.9), or low_cosine (<0.1).")
    parser.add_argument("--num_cores", type=int, default=mp.cpu_count(),
                        help="Number of CPU cores to use for parallel processing.")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Number of pairs to process per chunk in parallel.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed) # Seed numpy for pandas sampling

    # --- 1. Setup ---
    output_dir = os.path.dirname(args.output_csv)
    if output_dir: # Ensure output directory exists if specified
        os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Substructure Check (Parallel) ---")
    print(f"Input file: {args.input_file}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of samples: {'All' if args.num_samples <= 0 else args.num_samples}")
    if args.num_samples > 0:
        print(f"Sampling mode: {args.sample_mode}")
    print(f"Using {args.num_cores} CPU cores.")
    print(f"Processing chunk size: {args.chunk_size}")


    # --- 2. Load Data ---
    print(f"\nLoading data...")
    start_time = time.time()
    try:
        df = pd.read_feather(args.input_file)
        # Ensure required columns exist
        required_cols = ['name_main', 'name_sub', 'smiles_main', 'smiles_sub', 'cosine_similarity']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             print(f"Error: Input file missing required columns: {missing}")
             return
        print(f"Loaded {len(df):,} pairs in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        return

    # --- 3. Select Pairs to Process ---
    if args.num_samples <= 0 or args.num_samples >= len(df):
        print(f"\nProcessing all {len(df):,} pairs...")
        pairs_to_process_df = df
    else:
        print(f"\nSampling {args.num_samples} pairs...")
        sampled_df = pd.DataFrame() # Initialize empty DataFrame
        if args.sample_mode == "random":
            sampled_df = df.sample(n=args.num_samples, random_state=args.seed)
        elif args.sample_mode == "high_cosine":
            high_cosine_df = df[df['cosine_similarity'] >= 0.9]
            if len(high_cosine_df) >= args.num_samples:
                sampled_df = high_cosine_df.sample(n=args.num_samples, random_state=args.seed)
            else:
                print(f"Warning: Found only {len(high_cosine_df)} high-cosine pairs. Using all of them.")
                sampled_df = high_cosine_df
        elif args.sample_mode == "low_cosine":
            low_cosine_df = df[df['cosine_similarity'] < 0.1]
            if len(low_cosine_df) >= args.num_samples:
                sampled_df = low_cosine_df.sample(n=args.num_samples, random_state=args.seed)
            else:
                print(f"Warning: Found only {len(low_cosine_df)} low-cosine pairs. Using all of them.")
                sampled_df = low_cosine_df

        if sampled_df.empty:
            print("No pairs found for the selected sampling mode. Exiting.")
            return
        pairs_to_process_df = sampled_df
        print(f"Selected {len(pairs_to_process_df)} pairs for processing.")


    # --- 4. Process Pairs in Parallel ---
    print(f"\nChecking substructure relationships using {args.num_cores} cores...")
    start_process_time = time.time()
    results = []
    processed_count = 0
    match_count = 0
    error_count = 0

    # Prepare data iterable for the pool
    # Convert relevant columns to tuples for efficient passing
    data_iterable = pairs_to_process_df[['name_main', 'smiles_main', 'name_sub', 'smiles_sub']].itertuples(index=False, name=None)
    total_pairs = len(pairs_to_process_df)

    # Create a pool of workers
    with mp.Pool(processes=args.num_cores) as pool:
        # Use imap_unordered for potentially better memory usage and progress bar
        # chunksize helps manage overhead
        with tqdm(total=total_pairs, desc="Checking pairs") as pbar:
            # Use partial if check_substructure_worker needed extra constant args
            # process_func = partial(check_substructure_worker, extra_arg=value)
            process_func = check_substructure_worker

            for result in pool.imap_unordered(process_func, data_iterable, chunksize=args.chunk_size):
                if result is not None: # If processing was successful
                    results.append(result)
                    processed_count += 1
                    if result['is_substructure']:
                        match_count += 1
                else: # If check_substructure_worker returned None (error)
                    error_count += 1
                pbar.update(1) # Update progress bar for each result returned


    process_duration = time.time() - start_process_time
    print(f"Parallel processing finished in {process_duration:.2f}s")

    # --- 5. Save to CSV and Report Summary ---
    print(f"\nSaving results to {args.output_csv}...")
    results_df = pd.DataFrame(results)

    try:
        # Sort results if needed, e.g., by id_main (optional)
        # results_df = results_df.sort_values(by='id_main')
        results_df.to_csv(args.output_csv, index=False)
        print("CSV file saved successfully.")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    print(f"\n--- Substructure Check Complete ---")
    print(f"Total pairs processed: {processed_count}")
    print(f"Pairs skipped due to errors: {error_count}")
    if processed_count > 0:
        match_percentage = (match_count / processed_count) * 100
        print(f"Substructure matches found: {match_count} ({match_percentage:.1f}%)")
    else:
        print("No pairs were successfully processed.")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    # Set start method for multiprocessing (important for some environments)
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
             start_method = 'fork' if os.name == 'posix' else 'spawn'
             mp.set_start_method(start_method)
             print(f"Set multiprocessing start method to '{start_method}'.")
        else:
             print(f"Multiprocessing start method already set to '{current_method}'.")
    except RuntimeError as e:
        print(f"Could not set multiprocessing start method (may already be set): {e}")

    main()