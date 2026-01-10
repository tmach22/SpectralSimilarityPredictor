import pandas as pd
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Import MatchMS ---
try:
    from matchms import Spectrum
    from matchms.similarity import ModifiedCosine
except ImportError:
    raise ImportError("matchms is not installed. Please install it with 'pip install matchms'")

# --- CONFIGURATION ---
CHUNK_SIZE = 5000      # Spectra per block
NUM_WORKERS = 4        # Number of CPU cores
TOLERANCE = 0.05       # Modified Cosine Tolerance (Da)

# --- NORMALIZATION HELPERS ---
def normalize_instrument(val):
    s = str(val).upper().strip()
    if any(x in s for x in ['ORBITRAP', 'EXACTIVE', 'EXPLORIS', 'HYBRID FT', 'ID-X']): return 'Orbitrap'
    if any(x in s for x in ['QTOF', 'Q-TOF', 'Q TOF', 'TRIPLETOF', 'SYNAPT', 'XEVO', 'MAXIS', 'MICROTOF', 'OTOF', 'TOF']): return 'QTOF'
    if any(x in s for x in ['QQQ', 'TRIPLE QUAD', 'API', 'QUATTRO', 'TSQ', 'QTRAP', 'SCIENTIFIC Q-EXACTIVE FOCUS']): return 'Triple Quad'
    if any(x in s for x in ['ION TRAP', 'LTQ', 'LCQ', 'ESQUIRE', 'IT-TOF']): return 'Ion Trap'
    return 'Unknown'

def normalize_ce(val):
    """
    Snaps to nearest 5 eV.
    Examples: 18, 19, 20, 21, 22 -> 20.
    Missing/Negative -> -1.
    """
    try:
        if pd.isna(val) or val < 0: return -1
        return int(round(val / 5.0) * 5)
    except: return -1

def normalize_adduct(val):
    """Simple cleanup for strict string matching"""
    return str(val).strip().upper()

def prepare_data_for_worker(df_subset):
    """
    Helper: Converts DF -> Spectrum Objects AND Metadata Arrays
    Includes the 'Nuclear Option' memory fix.
    """
    spectra_objs = []
    ids = []
    
    # Metadata vectors for filtering
    meta_inst = []
    meta_ce = []
    meta_adduct = []

    for _, row in df_subset.iterrows():
        peak_list = row['peaks']
        if peak_list is None: continue
        if hasattr(peak_list, '__len__') and len(peak_list) == 0: continue

        try:
            # 1. Nuclear Memory Fix (Force Contiguous Float64)
            clean_list = list(peak_list)
            peaks_arr = np.array(clean_list, dtype=np.float64)

            if peaks_arr.ndim != 2 or peaks_arr.shape[1] < 2: continue

            mz = np.ascontiguousarray(peaks_arr[:, 0].astype(np.float64))
            intensities = np.ascontiguousarray(peaks_arr[:, 1].astype(np.float64))
            intensities = np.sqrt(intensities)
            
            pmz_val = row.get('precursor_mz')
            if pd.isna(pmz_val): continue
            
            # 2. Extract Metadata (Snap logic applied here)
            inst = normalize_instrument(row.get('instrument_type', ''))
            ce = normalize_ce(row.get('collision_energy', -1))
            adduct = normalize_adduct(row.get('adduct', ''))
            
            meta = {
                'spectrum_id': str(row['spectrum_id']),
                'precursor_mz': float(pmz_val) 
            }
            
            spec = Spectrum(mz=mz, intensities=intensities, metadata=meta)
            spectra_objs.append(spec)
            ids.append(str(row['spectrum_id']))
            
            # Append to vectors
            meta_inst.append(inst)
            meta_ce.append(ce)
            meta_adduct.append(adduct)
            
        except Exception:
            continue
            
    return spectra_objs, ids, np.array(meta_inst), np.array(meta_ce), np.array(meta_adduct)

def process_block_pair(args):
    """
    Worker function: FILTER FIRST -> COMPUTE LATER.
    """
    (i, df_i, j, df_j, output_dir) = args
    
    # 1. Prepare Objects & Vectors
    specs_i, ids_i, inst_i, ce_i, add_i = prepare_data_for_worker(df_i)
    specs_j, ids_j, inst_j, ce_j, add_j = prepare_data_for_worker(df_j)
    
    if not specs_i or not specs_j:
        return 0

    # 2. VECTORIZED FILTERING (Broadcasting)
    # Shape: (N, M) boolean matrices
    
    # A. Instrument (Exact Match)
    mask_inst = (inst_i[:, None] == inst_j[None, :])
    
    # B. Adduct (Exact Match)
    mask_add = (add_i[:, None] == add_j[None, :])
    
    # C. Collision Energy (Exact Match on Snapped Values)
    # Since we already snapped them to [20, 25, 30...], exact match works perfectly.
    # Note: -1 matches -1 (Unknowns match each other).
    mask_ce = (ce_i[:, None] == ce_j[None, :])

    # D. Combined Mask
    final_mask = mask_inst & mask_add & mask_ce
    
    # Handle Symmetric Blocks (Exclude duplicates/diagonal)
    if i == j:
        final_mask = np.triu(final_mask, k=1).astype(bool)

    # 3. GET MATCH INDICES
    rows, cols = np.where(final_mask)
    
    if len(rows) == 0:
        return 0 

    # 4. COMPUTE SIMILARITY (Sparse Loop)
    reference_measure = ModifiedCosine(tolerance=TOLERANCE)
    results = []
    
    for idx in range(len(rows)):
        r = rows[idx]
        c = cols[idx]
        
        spec_a = specs_i[r]
        spec_b = specs_j[c]
        
        # Only calculating for valid experimental pairs
        score_obj = reference_measure.pair(spec_a, spec_b)
        
        if score_obj is not None:
            results.append({
                "name_main": ids_i[r],
                "name_sub": ids_j[c],
                "cosine_similarity": float(score_obj['score'])
            })

    # 5. Save Results
    if results:
        df_res = pd.DataFrame(results)
        filename = f"block_{i}_{j}.feather"
        save_path = output_dir / filename
        df_res.to_feather(save_path)
        return len(results)
        
    return 0

def create_oracle_pairs_parallel(input_path: Path, temp_dir: Path):
    
    input_path = Path(input_path)
    temp_dir = Path(temp_dir)
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input data from: {input_path}")
    df = pd.read_feather(input_path)
    print(f"Loaded {len(df)} records.")
    
    # 1. Chunking
    num_chunks = int(np.ceil(len(df) / CHUNK_SIZE))
    chunks = [df.iloc[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE].copy() for i in range(num_chunks)]
    print(f"Split data into {num_chunks} blocks of size ~{CHUNK_SIZE}")

    # 2. Generate Task List
    tasks = []
    print("Generating task list...")
    for i in range(num_chunks):
        for j in range(i, num_chunks):
            tasks.append((i, chunks[i], j, chunks[j], temp_dir))
    
    print(f"Total block comparisons to process: {len(tasks)}")

    # 3. Parallel Execution
    print(f"Starting execution with {NUM_WORKERS} cores...")
    
    total_pairs = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_block_pair, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Blocks"):
            try:
                count = future.result()
                if count:
                    total_pairs += count
            except Exception as e:
                print(f"Task failed: {e}")

    print(f"\nProcessing complete.")
    print(f"Total VALID pairs generated: {total_pairs}")
    print(f"All chunk files saved to: {temp_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--temp_dir", required=True, type=str, help="Directory to save the chunk files")
    parser.add_argument("--n_cores", type=int, default=4)
    args = parser.parse_args()
    
    NUM_WORKERS = args.n_cores 
    
    create_oracle_pairs_parallel(args.input_file, args.temp_dir)