import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import gc
import re

# --- 1. Helper Functions ---
def parse_peaks(peaks_input):
    """Robustly parses peaks into a (N, 2) array."""
    if peaks_input is None: return np.zeros((0, 2), dtype=np.float32)
    if isinstance(peaks_input, str):
        try:
            matches = re.findall(r'\[([\d\.\-eE]+)[,\s]+([\d\.\-eE]+)\]', peaks_input)
            if matches: return np.array(matches, dtype=np.float32)
        except: pass
        return np.zeros((0, 2), dtype=np.float32)
    try:
        if isinstance(peaks_input, (list, np.ndarray)):
            peaks = np.array(peaks_input) if not isinstance(peaks_input, np.ndarray) else peaks_input
            if peaks.ndim == 2 and peaks.shape[1] == 2: return peaks.astype(np.float32)
            try: peaks = np.vstack(peaks_input)
            except: pass
            if peaks.ndim == 2 and peaks.shape[1] == 2: return peaks.astype(np.float32)
    except: pass
    return np.zeros((0, 2), dtype=np.float32)

def get_binned_vector(peaks, min_mz=0, max_mz=1000, bin_size=1.0):
    n_bins = int((max_mz - min_mz) / bin_size) + 1
    if len(peaks) == 0: return None
    
    mzs, ints = peaks[:, 0], peaks[:, 1]
    mask = (mzs >= min_mz) & (mzs < max_mz)
    mzs, ints = mzs[mask], ints[mask]
    
    if len(mzs) == 0: return None
    
    indices = np.floor((mzs - min_mz) / bin_size).astype(int)
    vector = np.bincount(indices, weights=ints, minlength=n_bins).astype(np.float32)
    
    vector = np.sqrt(vector)
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return None

def update_ground_truth(args):
    print(f"--- Updating Ground Truth with Consensus Similarity ---")
    
    # 1. Load Metadata
    print(f"Loading master spectra from {args.spec_data_path}...")
    spec_df = pd.read_feather(args.spec_data_path)
    
    # --- Explicit Column Mapping based on your file structure ---
    id_col = 'spectrum_id'   # As per your file report
    mol_col = 'smiles'     # Using InChIKey for stable grouping
    
    print(f"Using Spectrum ID Column: '{id_col}'")
    print(f"Using Molecule Grouping Column: '{mol_col}'")
    
    # 2. Build Caches
    print("Pre-calculating spectrum vectors and grouping...")
    spec_vectors = {}
    mol_to_specs = {}
    spec_to_mol = {} 
    
    for idx, row in tqdm(spec_df.iterrows(), total=len(spec_df)):
        sid = row[id_col]
        mid = row[mol_col]
        
        # Parse peaks
        peaks = parse_peaks(row['peaks'])
        vec = get_binned_vector(peaks)
        
        if vec is not None:
            spec_vectors[sid] = vec
            spec_to_mol[sid] = mid
            
            if mid not in mol_to_specs:
                mol_to_specs[mid] = []
            mol_to_specs[mid].append(sid)
            
    print(f"Cached vectors for {len(spec_vectors)} spectra.")
    print(f"Mapped {len(mol_to_specs)} unique molecules (by {mol_col}).")
    
    # Free memory
    del spec_df
    gc.collect()

    # 3. Process Pairs File
    print(f"Loading pairs from {args.pairs_path}...")
    pairs_df = pd.read_feather(args.pairs_path)
    
    print("Calculating Consensus Similarity...")
    new_similarities = []
    old_similarities = []
    diffs = []
    
    # Iterate pairs
    # Using values directly for speed loop
    name_mains = pairs_df['name_main'].values
    name_subs = pairs_df['name_sub'].values
    old_sims = pairs_df['cosine_similarity'].values
    
    for sid_a, sid_b, old_sim in tqdm(zip(name_mains, name_subs, old_sims), total=len(pairs_df)):
        
        # Resolve Molecule ID (InChIKey) from our local map
        mid_a = spec_to_mol.get(sid_a)
        mid_b = spec_to_mol.get(sid_b)
        
        # Validity Check
        if mid_a is None or mid_b is None:
            new_similarities.append(old_sim)
            old_similarities.append(old_sim)
            diffs.append(0.0)
            continue
            
        # Get all spectra for these molecules
        specs_a = mol_to_specs.get(mid_a, [])
        specs_b = mol_to_specs.get(mid_b, [])
        
        # Get vectors
        vecs_a = [spec_vectors[s] for s in specs_a if s in spec_vectors]
        vecs_b = [spec_vectors[s] for s in specs_b if s in spec_vectors]
        
        if not vecs_a or not vecs_b:
             new_similarities.append(old_sim)
             old_similarities.append(old_sim)
             diffs.append(0.0)
             continue
             
        # Stack for matrix multiplication
        mat_a = np.stack(vecs_a)
        mat_b = np.stack(vecs_b)
        
        # Calculate pairwise cosine similarity matrix
        sim_matrix = cosine_similarity(mat_a, mat_b)
        
        # Consensus = Mean of all pairwise similarities
        consensus_sim = np.mean(sim_matrix)
        
        new_similarities.append(consensus_sim)
        old_similarities.append(old_sim)
        diffs.append(abs(consensus_sim - old_sim))
        
    # 4. Update and Save
    pairs_df['original_similarity'] = old_similarities
    pairs_df['cosine_similarity'] = np.array(new_similarities, dtype=np.float32)
    
    mean_diff = np.mean(diffs)
    print(f"\n--- Update Summary ---")
    print(f"Mean Absolute Change: {mean_diff:.4f}")
    print(f"Max Change: {np.max(diffs):.4f}")
    
    base, ext = os.path.splitext(args.pairs_path)
    output_path = f"{base}_CONSENSUS{ext}"
    
    print(f"Saving to: {output_path}")
    pairs_df.to_feather(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to binary_07_dataset_train_STRICT.feather")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to augmented_msg_df_STRICT.feather")
    
    args = parser.parse_args()
    update_ground_truth(args)