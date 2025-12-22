import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import re

# --- 1. Helper Functions ---
def parse_peaks(peaks_input):
    """Robustly parses peaks into a (N, 2) array."""
    if peaks_input is None: return np.zeros((0, 2), dtype=np.float32)
    if isinstance(peaks_input, (list, np.ndarray)):
        try:
            peaks = np.array(peaks_input) if not isinstance(peaks_input, np.ndarray) else peaks_input
            if peaks.ndim == 2 and peaks.shape[1] == 2: return peaks.astype(np.float32)
            try: peaks = np.vstack(peaks_input)
            except: pass
            if peaks.ndim == 2 and peaks.shape[1] == 2: return peaks.astype(np.float32)
        except: pass
    if isinstance(peaks_input, str):
        try:
            matches = re.findall(r'\[([\d\.\-eE]+)[,\s]+([\d\.\-eE]+)\]', peaks_input)
            if matches: return np.array(matches, dtype=np.float32)
        except: pass
    return np.zeros((0, 2), dtype=np.float32)

def get_binned_set(peaks):
    if len(peaks) == 0: return set()
    return set(np.floor(peaks[:, 0]).astype(int))

def calculate_jaccard(set_a, set_b):
    if len(set_a) == 0 or len(set_b) == 0: return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

# --- 2. Main Cleaning Logic ---
def strict_clean(args):
    print(f"--- Strict Cleaning (Ghosts + Imposters): {args.input_path} ---")
    df = pd.read_feather(args.input_path)
    initial_count = len(df)
    print(f"Total Records: {initial_count:,}")
    
    # Track removal stats
    stats = {'ghost': 0, 'imposter': 0, 'kept': 0}
    
    # --- PHASE 1: GHOST FILTERING & PARSING ---
    print("\nPhase 1: Identifying Ghosts (< 5 peaks)...")
    
    all_sets = []     # Store parsed sets for Phase 2
    ghost_indices = set() # Track ghosts to exclude them from grouping
    
    for idx, peaks_raw in tqdm(enumerate(df['peaks']), total=initial_count):
        # Parse once
        p_arr = parse_peaks(peaks_raw)
        
        # Check Ghost
        if len(p_arr) < 5:
            ghost_indices.add(idx)
            stats['ghost'] += 1
            all_sets.append(None) # Placeholder
        else:
            # Store valid set
            all_sets.append(get_binned_set(p_arr))
            
    print(f"  Ghosts Found: {stats['ghost']:,}")
    
    # --- PHASE 2: GROUPING (Safe Type Handling) ---
    print("Grouping by metadata...")
    group_cols = ['smiles', 'adduct', 'instrument_type', 'collision_energy']
    
    # Create temporary dataframe for grouping to avoid Feather type errors
    df_grouping = df[group_cols].copy()
    for col in group_cols:
        if col in df_grouping.columns:
            if col == 'collision_energy':
                df_grouping[col] = df_grouping[col].fillna(-1.0)
            else:
                df_grouping[col] = df_grouping[col].fillna("Unknown")
            
    grouped = df_grouping.groupby(group_cols)
    
    # --- PHASE 3: IMPOSTER FILTERING (Medoid Strategy) ---
    print("Phase 2: Identifying Imposters (Group Consistency)...")
    
    keep_indices = []
    
    for _, group in tqdm(grouped):
        # Get indices for this group
        raw_indices = group.index.tolist()
        
        # Filter out Ghosts immediately
        valid_indices = [i for i in raw_indices if i not in ghost_indices]
        
        # If group is empty after removing ghosts, continue
        if not valid_indices:
            continue
            
        # If singleton (only 1 valid spectrum), we keep it (benefit of the doubt)
        if len(valid_indices) == 1:
            keep_indices.append(valid_indices[0])
            stats['kept'] += 1
            continue
            
        # --- Medoid Logic for Replicates ---
        n = len(valid_indices)
        sets = [all_sets[i] for i in valid_indices]
        
        # Calculate N x N Jaccard Matrix
        scores = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j: 
                    scores[i, j] = 1.0
                else:
                    s = calculate_jaccard(sets[i], sets[j])
                    scores[i, j] = s
                    scores[j, i] = s
        
        # Find Medoid (The spectrum most similar to all others)
        sum_scores = scores.sum(axis=1)
        medoid_idx_local = np.argmax(sum_scores)
        
        # Filter members against Medoid
        for i in range(n):
            sim_to_medoid = scores[i, medoid_idx_local]
            
            # STRICT THRESHOLD: Must share 30% of peaks with the center
            if sim_to_medoid >= 0.3:
                keep_indices.append(valid_indices[i])
                stats['kept'] += 1
            else:
                stats['imposter'] += 1

    # --- PHASE 4: SAVE ---
    print("\n" + "="*40)
    print("FINAL CLEANING REPORT")
    print("="*40)
    print(f"Original:  {initial_count:,}")
    print(f"Ghosts:    {stats['ghost']:,} (Removed)")
    print(f"Imposters: {stats['imposter']:,} (Removed)")
    print(f"Kept:      {len(keep_indices):,} ({len(keep_indices)/initial_count*100:.1f}%)")
    
    # Select from ORIGINAL df to preserve types
    df_clean = df.loc[sorted(keep_indices)].reset_index(drop=True)
    
    base, ext = os.path.splitext(args.input_path)
    output_path = f"{base}_STRICT{ext}"
    
    df_clean.to_feather(output_path)
    print(f"\nSaved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to augmented_msg_df.feather")
    args = parser.parse_args()
    strict_clean(args)