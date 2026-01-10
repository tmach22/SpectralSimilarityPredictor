import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import re
import ast

# --- Reuse Robust Parsers ---
def parse_peaks(peaks_input):
    if peaks_input is None: return np.zeros((0, 2), dtype=np.float32)
    if isinstance(peaks_input, str):
        try:
            matches = re.findall(r'\[([\d\.\-eE]+)[,\s]+([\d\.\-eE]+)\]', peaks_input)
            if matches: return np.array(matches, dtype=np.float32)
        except: pass
        return np.zeros((0, 2), dtype=np.float32)
    try:
        # If it's a list of arrays (ragged), stack them
        if isinstance(peaks_input, (list, np.ndarray)) and len(peaks_input) > 0:
             # Check if it's already a 2D array
            if isinstance(peaks_input, np.ndarray) and peaks_input.ndim == 2:
                return peaks_input.astype(np.float32)
            # Otherwise stack
            peaks = np.vstack(peaks_input)
            if peaks.ndim == 2 and peaks.shape[1] == 2: return peaks.astype(np.float32)
    except: pass
    return np.zeros((0, 2), dtype=np.float32)

def get_binned_vector(peaks, min_mz=0, max_mz=1000, bin_size=1.0):
    peaks = parse_peaks(peaks)
    n_bins = int((max_mz - min_mz) / bin_size) + 1
    if len(peaks) == 0: return None
    
    mzs, ints = peaks[:, 0], peaks[:, 1]
    mask = (mzs >= min_mz) & (mzs < max_mz)
    mzs, ints = mzs[mask], ints[mask]
    
    if len(mzs) == 0: return None
    
    indices = np.floor((mzs - min_mz) / bin_size).astype(int)
    vector = np.bincount(indices, weights=ints, minlength=n_bins)
    
    # Normalize
    vector = np.sqrt(vector)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None

def diagnose(args):
    print("--- Diagnosing 'Dirty' Data (Full Dataset) ---")
    print(f"Loading {args.input_path}...")
    df = pd.read_feather(args.input_path)
    
    # Group by identical conditions
    group_cols = ['smiles', 'adduct', 'instrument_type', 'collision_energy']
    for col in group_cols:
        if col in df.columns: df[col] = df[col].fillna("Unknown")
            
    print(f"Grouping by {group_cols}...")
    grouped = df.groupby(group_cols)
    
    # Counters
    stats = {
        'total_pairs_checked': 0,
        'clean_pairs': 0,      # Sim >= 0.8
        'dirty_pairs': 0,      # Sim < 0.8
        'type_ghost': 0,       # < 5 peaks
        'type_imposter': 0,    # Peaks exist but don't match (Mass mismatch)
        'type_shapeshifter': 0 # Peaks match but intensities wrong
    }
    
    print("Filtering for groups with replicates...")
    # We iterate over the groupby object directly to avoid creating a massive list in memory
    # However, to use tqdm, we need the total count first, or we filter beforehand.
    # Filtering beforehand is safer for progress tracking.
    multi_spec_groups = [g for n, g in grouped if len(g) > 1]
    
    total_groups = len(multi_spec_groups)
    print(f"Found {total_groups:,} groups with replicates.")

    # Optional Limit
    if args.limit and args.limit > 0:
        if total_groups > args.limit:
            print(f"Limit set: Analyzing random {args.limit} groups...")
            import random
            multi_spec_groups = random.sample(multi_spec_groups, args.limit)
        else:
            print(f"Limit set ({args.limit}), but total groups ({total_groups}) is smaller. Analyzing all.")
    else:
        print(f"Analyzing ALL {total_groups:,} groups...")
    
    for group in tqdm(multi_spec_groups):
        peaks_list = group['peaks'].tolist()
        
        # Compare every pair in the group
        for i in range(len(peaks_list)):
            for j in range(i + 1, len(peaks_list)):
                stats['total_pairs_checked'] += 1
                
                p1_raw = parse_peaks(peaks_list[i])
                p2_raw = parse_peaks(peaks_list[j])
                
                # Check 1: The "Ghost" (Empty/Sparse)
                if len(p1_raw) < 5 or len(p2_raw) < 5:
                    stats['dirty_pairs'] += 1
                    stats['type_ghost'] += 1
                    continue
                
                # Calculate Similarity
                v1 = get_binned_vector(p1_raw)
                v2 = get_binned_vector(p2_raw)
                
                if v1 is None or v2 is None: 
                    stats['dirty_pairs'] += 1
                    stats['type_ghost'] += 1
                    continue
                    
                sim = np.dot(v1, v2)
                
                if sim >= 0.8:
                    stats['clean_pairs'] += 1
                    continue
                
                # If we are here, it's a "Dirty" pair (Sim < 0.8)
                stats['dirty_pairs'] += 1
                
                # Diagnose: Imposter vs. Shapeshifter
                b1_bool = v1 > 0
                b2_bool = v2 > 0
                intersection = np.sum(b1_bool & b2_bool)
                union = np.sum(b1_bool | b2_bool)
                jaccard = intersection / union if union > 0 else 0
                
                if jaccard < 0.2:
                    # They share very few masses -> Different Molecules?
                    stats['type_imposter'] += 1
                else:
                    # They share masses but intensities are wrong -> Instrument Variance
                    stats['type_shapeshifter'] += 1

    # Report
    print("\n" + "="*40)
    print("FULL DATASET DIAGNOSTIC REPORT")
    print("="*40)
    total = stats['total_pairs_checked']
    if total == 0: 
        print("No pairs analyzed.")
        return

    print(f"Total Pairs Analyzed: {total:,}")
    print(f"Clean Pairs (Sim >= 0.8): {stats['clean_pairs']:,} ({stats['clean_pairs']/total*100:.1f}%)")
    print(f"Dirty Pairs (Sim < 0.8):  {stats['dirty_pairs']:,} ({stats['dirty_pairs']/total*100:.1f}%)")
    
    print("\n--- Breakdown of Dirty Data ---")
    dirty = stats['dirty_pairs']
    if dirty > 0:
        print(f"1. The 'Ghost' (Sparse/Empty):     {stats['type_ghost']:,} ({stats['type_ghost']/dirty*100:.1f}%)")
        print(f"   -> Cause: Instrument Failure / Peak Picking Error")
        
        print(f"2. The 'Imposter' (Mass Mismatch): {stats['type_imposter']:,} ({stats['type_imposter']/dirty*100:.1f}%)")
        print(f"   -> Cause: Metadata Mislabeling / Contamination")
        
        print(f"3. The 'Shapeshifter' (Intensity): {stats['type_shapeshifter']:,} ({stats['type_shapeshifter']/dirty*100:.1f}%)")
        print(f"   -> Cause: Instrument Calibration / Energy Variance")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to augmented_msg_df.feather")
    # Optional limit for testing; default is None (run all)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of groups to analyze (default: All)")
    
    args = parser.parse_args()
    diagnose(args)