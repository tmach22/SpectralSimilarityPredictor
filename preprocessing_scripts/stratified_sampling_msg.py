import pandas as pd
import argparse
import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
# Updated import for Morgan fingerprints
from rdkit.Chem import AllChem, rdMolDescriptors
# from scipy.sparse import csr_matrix # Not strictly needed anymore
# from scipy.spatial.distance import cdist # Not strictly needed anymore
import random
import multiprocessing as mp
from functools import partial
import os # Added for os.urandom

# --- Helper Functions ---

def process_spectrum(mzs_str, intensities_str, precision=10):
    """ Processes spectrum strings into a normalized sparse dict. """
    if pd.isna(mzs_str) or pd.isna(intensities_str): return {}
    try:
        mzs = [float(m) for m in mzs_str.split(',')]
        intensities = [float(i) for i in intensities_str.split(',')]
        intensities_sqrt = np.sqrt(intensities)
        binned_spectrum = {}
        for mz, intensity in zip(mzs, intensities_sqrt):
            if intensity > 0:
                bin_index = int(mz * precision)
                binned_spectrum[bin_index] = binned_spectrum.get(bin_index, 0) + intensity
        norm = np.sqrt(sum(v**2 for v in binned_spectrum.values()))
        if norm > 0:
            for bin_index in binned_spectrum: binned_spectrum[bin_index] /= norm
        return binned_spectrum
    except ValueError: # Handle potential malformed strings
        # print(f"Warning: Malformed spectrum string encountered. mzs='{mzs_str}', ints='{intensities_str}'")
        return {}


def fast_cosine(spec_a, spec_b):
    """ Calculates cosine between two sparse dict spectra. """
    # Ensure inputs are valid dictionaries
    if not isinstance(spec_a, dict) or not isinstance(spec_b, dict):
        return 0.0
    dot_product = 0.0
    # Iterate over the smaller dictionary for potential speedup
    if len(spec_a) < len(spec_b):
        for bin_index in spec_a:
            if bin_index in spec_b: dot_product += spec_a[bin_index] * spec_b[bin_index]
    else:
        for bin_index in spec_b:
            if bin_index in spec_a: dot_product += spec_a[bin_index] * spec_b[bin_index]
    return dot_product

# --- UPDATED get_fingerprint function ---
def get_fingerprint(smiles, radius=2, nBits=2048):
    """ Generates Morgan fingerprint using the recommended MorganGenerator. """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # Use the rdMolDescriptors module
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        # Ensure fingerprint is picklable for multiprocessing
        return fp # Convert to ExplicitBitVect
    except Exception as e:
        # print(f"Warning: RDKit error processing SMILES '{smiles}': {e}")
        return None
# --- END of UPDATED function ---


# --- Multiprocessing Worker Function ---
# --- *** CORRECTED SIGNATURE *** ---
def sample_and_calc_tanimoto_batch(batch_indices):
    """
    Worker function: Samples pairs and calculates Tanimoto for a batch.
    Returns list of (idx_a, idx_b, tanimoto).
    Relies on global variables valid_fps_list_global and total_valid_global
    set by the init_worker function.
    """
    # Access global variables set by init_worker
    global valid_fps_list_global, total_valid_global
    if valid_fps_list_global is None or total_valid_global is None:
         # Safety check in case initializer failed
         print("Error: Worker globals not initialized!")
         return []

    results = []
    batch_size = len(batch_indices) # This uses the dummy range passed by pool.map
    # Seed random number generator within each worker for better randomness
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    indices_a = np.random.randint(0, total_valid_global, batch_size)
    indices_b = np.random.randint(0, total_valid_global, batch_size)

    for i in range(batch_size):
        idx_a = indices_a[i]
        idx_b = indices_b[i]

        if idx_a == idx_b: continue

        fp_a = valid_fps_list_global[idx_a]
        fp_b = valid_fps_list_global[idx_b]
        # BulkTanimoto expects list of FPs for second arg
        # Ensure fp_b is in a list, even if it's just one
        if fp_a is None or fp_b is None: continue # Skip if fingerprint generation failed

        try:
             tanimoto = BulkTanimotoSimilarity(fp_a, [fp_b])[0]
             results.append((idx_a, idx_b, tanimoto))
        except Exception as e:
            # Handle potential errors during Tanimoto calculation if FPs are invalid
            # print(f"Warning: Error calculating Tanimoto for indices {idx_a}, {idx_b}: {e}")
            pass # Skip this pair

    return results
# --- *** END OF CORRECTION *** ---


# --- Global Variables for Workers ---
# Define these outside main so they are accessible to workers after fork
valid_fps_list_global = None
total_valid_global = None

def init_worker(valid_fps_list, total_valid):
    """Initializer for worker processes to set global variables."""
    global valid_fps_list_global, total_valid_global
    valid_fps_list_global = valid_fps_list
    total_valid_global = total_valid


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate a Tanimoto-stratified paired dataset from MassSpecGym (Parallel).")
    parser.add_argument("--input", type=str, required=True, help="Path to MassSpecGym.feather")
    parser.add_argument("--output", type=str, required=True, help="Path to save the new paired .feather file")
    parser.add_argument("--pairs_per_bin", type=int, default=1000000, help="Target number of pairs per Tanimoto bin")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of Tanimoto bins (0.0-1.0)")
    parser.add_argument("--sampling_batch_size", type=int, default=10000, help="Number of pairs to sample per core per iteration")
    parser.add_argument("--num_cores", type=int, default=mp.cpu_count(), help="Number of CPU cores to use")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")


    args = parser.parse_args()

    # --- 0. Set Seed (Optional) ---
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed) # Also seed Python's random


    # --- 1. Loading & Pre-processing Data ---
    print("--- 1. Loading & Pre-processing Data ---")
    start_time = time.time()
    try:
        df = pd.read_feather(args.input)
        print(f"Loaded {len(df)} records from {args.input}")
    except Exception as e:
        print(f"Error loading input file {args.input}: {e}")
        return


    print(f"Using {args.num_cores} CPU cores.")

    # --- Pre-process all spectra and fingerprints (Serially first) ---
    print("Pre-processing all spectra...")
    all_spectra = [process_spectrum(mzs, ints) for mzs, ints in tqdm(zip(df['mzs'], df['intensities']), total=len(df), desc="Processing Spectra")]

    print("Pre-processing all SMILES to fingerprints...")
    all_fps = [get_fingerprint(smiles) for smiles in tqdm(df['smiles'], total=len(df), desc="Processing SMILES")]

    all_ids = df['identifier'].values

    # Filter out failed fingerprints AND spectra
    valid_indices = [i for i, (fp, spec) in enumerate(zip(all_fps, all_spectra)) if fp is not None and isinstance(spec, dict) and spec]
    print(f"Filtered out {len(df) - len(valid_indices)} records with invalid SMILES or empty/malformed spectra.")

    total_valid = len(valid_indices)
    if total_valid < 2:
        print("Error: Need at least two valid records to create pairs.")
        return

    # Create lists containing only valid data, indexed 0 to total_valid-1
    # These will be passed to worker processes
    valid_fps_list = [all_fps[i] for i in valid_indices]
    valid_spectra_list = [all_spectra[i] for i in valid_indices]
    valid_ids_list = [all_ids[i] for i in valid_indices]

    print(f"Pre-processing complete. Time: {time.time() - start_time:.2f}s")

    # --- 2. Stratified Sampling by Tanimoto (Parallel) ---
    print(f"\n--- 2. Stratified Sampling ({args.pairs_per_bin:,} pairs per {args.num_bins} bins) ---")
    start_time_sampling = time.time()

    bins = np.linspace(0, 1, args.num_bins + 1)
    # Ensure bins cover the full range [0, 1] including edges
    bins[0] = 0.0
    bins[-1] = 1.0

    # Use Manager for shared dictionary if needed, but simple dict is usually fine if only main updates
    bin_counts = {i: 0 for i in range(args.num_bins)}
    target_per_bin = args.pairs_per_bin
    total_target = target_per_bin * args.num_bins

    pair_results = []
    sampled_pairs_set = set() # To avoid duplicate pairs (managed by main process)

    print(f"Starting sampling. Target: {total_target:,} pairs. This can take a very long time...")

    pbar = tqdm(total=total_target, desc="Sampling pairs")

    consecutive_misses = 0
    max_consecutive_misses_check_interval = 100 # Check every N batches
    miss_check_counter = 0
    # Increased safety break limit, adjust if needed
    max_total_misses_allowed = args.sampling_batch_size * args.num_cores * 5000

    # Initialize worker pool
    # Pass read-only data via initializer
    pool = mp.Pool(processes=args.num_cores, initializer=init_worker, initargs=(valid_fps_list, total_valid))

    try:
        while sum(bin_counts.values()) < total_target:

            # Create chunks of work for each core
            # We generate dummy indices; the actual sampling happens in the worker
            # Pass a unique seed to each chunk based on time/pid if args.seed is None?
            work_chunks = [range(args.sampling_batch_size) for _ in range(args.num_cores)]

            # Run sampling and Tanimoto calculation in parallel
            try:
                # pool.map only passes elements from work_chunks to the worker
                batch_results_list = pool.map(sample_and_calc_tanimoto_batch, work_chunks)
            except Exception as e:
                print(f"\nError during pool.map: {e}")
                print("Attempting to continue...") # Or break/re-init pool
                continue # Skip this iteration

            found_in_iteration = 0
            # Process results serially in the main process
            for batch_results in batch_results_list:
                for idx_a, idx_b, tanimoto in batch_results:
                    # Avoid duplicate pairs (order doesn't matter)
                    pair_key = tuple(sorted((idx_a, idx_b)))
                    if pair_key in sampled_pairs_set: continue

                    # Determine bin - use slightly safer binning
                    # np.digitize returns index of bin *edge* to the right
                    bin_index = np.searchsorted(bins, tanimoto, side='right') - 1
                    # Clamp index to be within [0, num_bins-1]
                    bin_index = max(0, min(bin_index, args.num_bins - 1))

                    # If bin is not full, add the pair
                    if bin_counts[bin_index] < target_per_bin:
                        spec_a = valid_spectra_list[idx_a]
                        spec_b = valid_spectra_list[idx_b]
                        cosine = fast_cosine(spec_a, spec_b)

                        pair_results.append((valid_ids_list[idx_a], valid_ids_list[idx_b], tanimoto, cosine, bin_index))
                        bin_counts[bin_index] += 1
                        sampled_pairs_set.add(pair_key)
                        pbar.update(1)
                        found_in_iteration += 1

                    # Check if overall target reached after adding this pair
                    if sum(bin_counts.values()) >= total_target: break
                if sum(bin_counts.values()) >= total_target: break

            if found_in_iteration == 0:
                consecutive_misses += args.sampling_batch_size * args.num_cores
            else:
                consecutive_misses = 0 # Reset misses counter
                # Update progress bar description with current counts
                bins_filled_count = sum(c>=target_per_bin for c in bin_counts.values())
                pbar.set_description(f"Sampling pairs (Bins filled: {bins_filled_count}/{args.num_bins})")

            # Check for excessive misses periodically
            miss_check_counter += 1
            if miss_check_counter >= max_consecutive_misses_check_interval:
                 if consecutive_misses >= max_total_misses_allowed:
                     print(f"\nWarning: Stopped early after {consecutive_misses:,} total consecutive misses without finding needed pairs.")
                     print("Bins might be underfilled. Check final bin counts.")
                     break
                 miss_check_counter = 0 # Reset check counter

            if sum(bin_counts.values()) >= total_target: break

    finally: # Ensure pool is closed even if errors occur
        print("Closing worker pool...")
        pool.close()
        pool.join()
        pbar.close()

    print(f"\nGenerated {len(pair_results):,} pairs across {args.num_bins} bins.")
    print("Final bin counts:")
    for i in range(args.num_bins):
        print(f"  Bin {i} [{bins[i]:.1f}-{bins[i+1]:.1f}): {bin_counts[i]:,}")
    print(f"Sampling Time: {time.time() - start_time_sampling:.2f}s")

    # --- 3. Combine and Save ---
    print("\n--- 3. Saving ---")
    if not pair_results:
        print("No pairs were generated. Saving an empty file.")
        final_df = pd.DataFrame(columns=['id_a', 'id_b', 'tanimoto', 'cosine_similarity', 'tanimoto_bin'])
    else:
        final_df = pd.DataFrame(pair_results, columns=['id_a', 'id_b', 'tanimoto', 'cosine_similarity', 'tanimoto_bin'])

        # Shuffle the final dataset
        final_df = final_df.sample(frac=1).reset_index(drop=True)

    print(final_df.head())
    print(f"Total pairs saved: {len(final_df):,}")

    # Analyze the resulting cosine distribution (optional)
    if not final_df.empty:
        print("\nResulting Cosine Similarity Distribution:")
        print(final_df['cosine_similarity'].describe())
    else:
        print("\nResulting Cosine Similarity Distribution: N/A (empty dataset)")


    try:
        final_df.to_feather(args.output)
        print(f"Successfully saved stratified dataset to {args.output}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Set start method for multiprocessing (important for some environments)
    # 'fork' is default on Unix, 'spawn' might be needed on Windows/macOS sometimes
    # It's generally safer to set it only if __name__ == '__main__'
    import os # Needed for os.urandom
    try:
        # Check if start method is already set to avoid error
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
             # Try 'fork' first, fallback to 'spawn' if needed or on non-Unix
             start_method = 'fork' if os.name == 'posix' else 'spawn'
             mp.set_start_method(start_method)
             print(f"Set multiprocessing start method to '{start_method}'.")
        elif current_method != 'fork': # Or whichever you prefer
             # If already set but not to 'fork', maybe warn or just accept it
             print(f"Multiprocessing start method already set to '{current_method}'.")
        else:
             print(f"Multiprocessing start method already set to '{current_method}'.")

    except RuntimeError as e:
        print(f"Could not set multiprocessing start method (may already be set): {e}")

    main()