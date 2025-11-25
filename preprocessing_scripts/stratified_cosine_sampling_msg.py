import pandas as pd
import argparse
import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from rdkit.Chem import AllChem, rdMolDescriptors
import random
import multiprocessing as mp
from functools import partial
import os
import collections

# --- SpectrumTuple Definition ---
SpectrumTuple = collections.namedtuple("SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"])

# --- Helper Functions ---

def parse_mzs_intensities(mzs_str, intensities_str):
    try:
        mzs = np.array([float(m) for m in mzs_str.split(',')], dtype=np.float32)
        intensities = np.array([float(i) for i in intensities_str.split(',')], dtype=np.float32)
        if mzs.size == 0 or intensities.size == 0: return None, None
        sort_idx = np.argsort(mzs)
        return mzs[sort_idx], intensities[sort_idx]
    except Exception:
        return None, None

def preprocess_spectrum_for_fast_cosine(row):
    mzs, intensities = parse_mzs_intensities(row.get('mzs'), row.get('intensities'))
    if mzs is None or intensities is None: return None
    precursor_mz = row.get('precursor_mz', 0.0)
    precursor_charge_val = row.get('precursor_charge', 1)
    try:
        precursor_charge = int(precursor_charge_val); precursor_charge = max(1, precursor_charge)
    except (ValueError, TypeError):
        precursor_charge = 1
    return SpectrumTuple(precursor_mz, precursor_charge, mzs, intensities) # Store RAW intensities

def normalize_spectrum(spec_tuple):
    if spec_tuple is None or spec_tuple.intensity is None or len(spec_tuple.intensity) == 0:
        return None
    try:
        sqrt_intensities = np.sqrt(spec_tuple.intensity)
        norm = np.linalg.norm(sqrt_intensities)
        if norm < 1e-9:
             normalized_intensities = np.zeros_like(sqrt_intensities)
        else:
             normalized_intensities = sqrt_intensities / norm
        return SpectrumTuple(spec_tuple.precursor_mz, spec_tuple.precursor_charge, spec_tuple.mz, normalized_intensities)
    except Exception:
        return None

def get_fingerprint(smiles, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return fp
    except Exception:
        return None

def _cosine_fast_original_logic(spec, spec_other, fragment_mz_tolerance, fragment_ppm_tolerance, allow_shift=False):
    # This is the complex, accurate peak-matching function
    precursor_charge = max(spec.precursor_charge, 1)
    precursor_mass_diff = (spec.precursor_mz - spec_other.precursor_mz) * precursor_charge
    num_shifts = 1
    if allow_shift and abs(precursor_mass_diff) >= fragment_mz_tolerance:
        num_shifts += precursor_charge
    other_peak_index = np.zeros(num_shifts, dtype=np.uint32)
    mass_diff = np.zeros(num_shifts, dtype=np.float32)
    for charge in range(1, num_shifts):
        mass_diff[charge] = precursor_mass_diff / charge
    
    peak_match_scores, peak_match_idx = [], []
    other_mz_len = len(spec_other.mz)
    
    for peak_index, (peak_mz, peak_intensity) in enumerate(zip(spec.mz, spec.intensity)):
        if peak_intensity == 0: continue
        for cpi in range(num_shifts):
            current_other_idx = other_peak_index[cpi]
            target_mz_shifted = peak_mz - mass_diff[cpi]
            while current_other_idx < other_mz_len - 1 and \
                  (spec_other.mz[current_other_idx] < target_mz_shifted - fragment_mz_tolerance):
                   current_other_idx += 1
            other_peak_index[cpi] = current_other_idx
        
        for cpi in range(num_shifts):
            index = 0
            other_peak_i = other_peak_index[cpi] + index
            target_mz_shifted = peak_mz - mass_diff[cpi]
            
            while other_peak_i < other_mz_len:
                current_other_mz_shifted = spec_other.mz[other_peak_i] + mass_diff[cpi]
                delta_mz = abs(peak_mz - current_other_mz_shifted)
                
                if delta_mz > fragment_mz_tolerance:
                    if spec_other.mz[other_peak_i] > target_mz_shifted + fragment_mz_tolerance: break
                    index += 1; other_peak_i = other_peak_index[cpi] + index
                    continue
                    
                ppm_limit = fragment_ppm_tolerance * peak_mz / 1e6
                if delta_mz <= ppm_limit:
                    if spec_other.intensity[other_peak_i] > 0: # Only match non-zero peaks
                        peak_match_scores.append(peak_intensity * spec_other.intensity[other_peak_i])
                        peak_match_idx.append((peak_index, other_peak_i))
                index += 1; other_peak_i = other_peak_index[cpi] + index

    score = 0.0
    if peak_match_scores:
        peak_match_scores_arr = np.asarray(peak_match_scores, dtype=np.float32)
        peak_match_order = np.argsort(peak_match_scores_arr)[::-1]
        peaks_used, other_peaks_used = set(), set()
        for i in peak_match_order:
             current_score = peak_match_scores_arr[i]
             peak_i, other_peak_i = peak_match_idx[i]
             if peak_i not in peaks_used and other_peak_i not in other_peaks_used:
                 score += current_score
                 peaks_used.add(peak_i); other_peaks_used.add(other_peak_i)
    
    return max(0.0, min(1.0, score)) # Return final cosine score

# --- Global Variables for Workers ---
# These will hold the *full* pre-processed dataset
g_all_data = None

def init_worker(all_data):
    """Initializer for worker processes."""
    global g_all_data
    g_all_data = all_data

# --- Multiprocessing Worker Function ---
def process_block(job_indices, output_dir, chunk_size, mz_tol, ppm_tol):
    """
    Processes a single block (chunk_i vs chunk_j) of the all-pairs matrix.
    """
    global g_all_data
    chunk_i_idx, chunk_j_idx = job_indices
    
    # Define the slice for chunk i and chunk j
    start_i = chunk_i_idx * chunk_size
    end_i = min(start_i + chunk_size, len(g_all_data))
    
    start_j = chunk_j_idx * chunk_size
    end_j = min(start_j + chunk_size, len(g_all_data))

    results = []
    
    # Iterate through every pair in this block
    for i in range(start_i, end_i):
        # Handle diagonal blocks (i==j) vs. off-diagonal (i!=j)
        # If i==j, only compute the upper triangle (j starts from i+1)
        # If i!=j, compute the full rectangle (j starts from start_j)
        j_start_offset = (i + 1) if (chunk_i_idx == chunk_j_idx) else start_j
        
        if j_start_offset >= end_j: # Skip if no work
            continue

        # Get data for spectrum 'i'
        data_i = g_all_data[i]
        # Pre-normalize spectrum 'i' once for the inner loop
        spec_i_norm = normalize_spectrum(data_i['spec_raw'])
        if spec_i_norm is None: continue

        for j in range(j_start_offset, end_j):
            data_j = g_all_data[j]

            # --- APPLY FILTERS ---
            if data_i['inchikey'] == data_j['inchikey'] and data_i['inchikey'] != 'Unknown': continue
            if data_i['instrument'] != data_j['instrument']: continue
            if data_i['adduct'] != data_j['adduct']: continue
            if not np.isclose(data_i['energy'], data_j['energy'], atol=1e-6): continue
            
            # --- FILTERS PASSED ---
            
            # 1. Calculate Tanimoto
            fp_i = data_i['fp']
            fp_j = data_j['fp']
            if fp_i is None or fp_j is None: continue
            try:
                tanimoto = BulkTanimotoSimilarity(fp_i, [fp_j])[0]
            except Exception:
                continue

            # 2. Calculate Cosine
            spec_j_norm = normalize_spectrum(data_j['spec_raw'])
            if spec_j_norm is None: continue
            
            cosine_score = _cosine_fast_original_logic(
                spec_i_norm, spec_j_norm,
                fragment_mz_tolerance=mz_tol,
                fragment_ppm_tolerance=ppm_tol,
                allow_shift=False
            )
            
            results.append((data_i['id'], data_j['id'], tanimoto, cosine_score))
    
    # --- Save results for this block ---
    if results:
        try:
            block_filename = f"block_{chunk_i_idx:04d}_{chunk_j_idx:04d}.parquet"
            results_df = pd.DataFrame(results, columns=['id_a', 'id_b', 'tanimoto', 'cosine_similarity'])
            results_df.to_parquet(os.path.join(output_dir, block_filename), index=False, engine='pyarrow')
            return len(results)
        except Exception as e:
            print(f"Error saving block {chunk_i_idx}-{chunk_j_idx}: {e}")
            return 0
    return 0

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Generate ALL filtered pairs in a chunked, parallel way.")
    parser.add_argument("--input", type=str, required=True, help="Path to MassSpecGym.feather")
    parser.add_argument("--output_dir", type=str, required=True, help="*Directory* to save the chunked result files (e.g., .parquet)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of spectra per chunk (e.g., 1000). Total jobs = (N/chunk_size)^2 / 2")
    parser.add_argument("--num_cores", type=int, default=mp.cpu_count(), help="Number of CPU cores to use")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--mz_tolerance", type=float, default=0.02)
    parser.add_argument("--ppm_tolerance", type=float, default=10.0)
    args = parser.parse_args()

    if args.seed is not None: np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- 1. Loading & Pre-processing All Data (This may take a lot of RAM) ---")
    start_time = time.time()
    try:
        df = pd.read_feather(args.input)
    except Exception as e:
        print(f"Error loading input file {args.input}: {e}"); return
    print(f"Loaded {len(df)} records from {args.input}")

    # Pre-process ALL data into a list of dicts (to be held in main process RAM)
    all_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        spec_raw = preprocess_spectrum_for_fast_cosine(row)
        fp = get_fingerprint(row.get('smiles'))
        
        # Only include valid, filterable data
        if spec_raw is not None and fp is not None:
            all_data.append({
                'id': row.get('identifier'),
                'spec_raw': spec_raw,
                'fp': fp,
                'inchikey': row.get('inchikey', 'Unknown'),
                'instrument': row.get('instrument_type', 'Unknown'),
                'adduct': row.get('adduct', 'Unknown'),
                'energy': row.get('collision_energy', -1.0)
            })
    
    total_valid = len(all_data)
    print(f"Kept {total_valid} valid spectra for processing.")
    print(f"Pre-processing complete. Time: {time.time() - start_time:.2f}s")
    
    # --- 2. Create Job List ---
    num_chunks = int(np.ceil(total_valid / args.chunk_size))
    jobs = []
    for i in range(num_chunks):
        for j in range(i, num_chunks): # Only compute the upper triangle (i, j) where i <= j
            jobs.append((i, j))
    
    print(f"Divided {total_valid} spectra into {num_chunks} chunks of size {args.chunk_size}.")
    print(f"Total jobs to compute: {len(jobs):,}")
    print(f"Output files will be saved in: {args.output_dir}")
    print(f"Using {args.num_cores} CPU cores. This will take a very long time...")
    
    # --- 3. Run Jobs in Parallel ---
    start_compute_time = time.time()
    total_pairs_found = 0
    
    # Create the partial function to pass fixed arguments to the worker
    worker_func = partial(process_block, 
                          output_dir=args.output_dir, 
                          chunk_size=args.chunk_size, 
                          mz_tol=args.mz_tolerance, 
                          ppm_tol=args.ppm_tolerance)
    
    # Initialize worker pool
    pool = mp.Pool(processes=args.num_cores, initializer=init_worker, initargs=(all_data,))
    
    try:
        with tqdm(total=len(jobs), desc="Processing Chunks") as pbar:
            for pairs_in_chunk in pool.imap_unordered(worker_func, jobs):
                total_pairs_found += pairs_in_chunk
                pbar.update(1)
                pbar.set_postfix_str(f"Total Pairs Found: {total_pairs_found:,}")
    
    finally:
        print("Closing worker pool...")
        pool.close()
        pool.join()

    print(f"\n--- Brute Force Computation Complete ---")
    print(f"Total processing time: {(time.time() - start_compute_time) / 3600:.2f} hours")
    print(f"Total filtered pairs found: {total_pairs_found:,}")
    print(f"Result files saved in: {args.output_dir}")
    print("\nYou must now write a separate script to combine all .parquet files in this directory.")

if __name__ == "__main__":
    import os
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