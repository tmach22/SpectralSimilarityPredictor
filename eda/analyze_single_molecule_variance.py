import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import re

def parse_peaks(peaks_input):
    """
    Robustly converts various peak formats into a standard (N, 2) Float32 array.
    Handles:
      - Standard lists/arrays: [[mz, i], [mz, i]]
      - Ragged object arrays: [array([mz, i]), array([mz, i])]
      - String representations (fallback)
    """
    # 1. Handle None/Empty
    if peaks_input is None:
        return np.zeros((0, 2), dtype=np.float32)

    # 2. Handle Strings (Rescue Mode)
    if isinstance(peaks_input, str):
        # Regex to find pairs of numbers inside brackets, e.g., [100.0, 1.0]
        # This handles standard string lists or numpy string dumps
        try:
            matches = re.findall(r'\[([\d\.\-eE]+)[,\s]+([\d\.\-eE]+)\]', peaks_input)
            if matches:
                return np.array(matches, dtype=np.float32)
        except:
            pass
        return np.zeros((0, 2), dtype=np.float32)

    # 3. Handle Objects (Lists, Arrays, Series)
    try:
        # Attempt to force stacking.
        # If peaks_input is a list of arrays, np.vstack converts it to (N, 2)
        # If peaks_input is already (N, 2), np.vstack preserves it.
        peaks = np.vstack(peaks_input)
        
        # Ensure correct shape and type
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            return peaks.astype(np.float32)
            
    except Exception:
        # Fallback for very messy ragged lists
        pass

    return np.zeros((0, 2), dtype=np.float32)

def get_binned_spectrum(peaks, min_mz=0, max_mz=1000, bin_size=1.0):
    # Parse using the new robust function
    peaks = parse_peaks(peaks)
    n_bins = int((max_mz - min_mz) / bin_size) + 1
    
    if len(peaks) == 0:
        return np.zeros(n_bins, dtype=np.float32)
        
    mzs = peaks[:, 0]
    intensities = peaks[:, 1]
    
    # Filter range
    mask = (mzs >= min_mz) & (mzs < max_mz)
    mzs = mzs[mask]
    intensities = intensities[mask]
    
    if len(mzs) == 0:
        return np.zeros(n_bins, dtype=np.float32)
    
    # Binning
    indices = np.floor((mzs - min_mz) / bin_size).astype(int)
    vector = np.bincount(indices, weights=intensities, minlength=n_bins)
    
    # Normalize (Square Root + L2)
    vector = np.sqrt(vector)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
        
    return vector

def analyze_variance(args):
    print(f"--- Analyzing Conditional Spectral Variance (Fixed Parser) ---")
    print(f"Input File: {args.input_path}")
    
    # 1. Load Data
    df = pd.read_feather(args.input_path)
    print(f"Total Spectra Records: {len(df):,}")
    
    # 2. Grouping
    group_cols = ['smiles', 'adduct', 'instrument_type', 'collision_energy']
    for col in group_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            
    grouped = df.groupby(group_cols)
    n_groups = len(grouped)
    
    # 3. Filter for Replicates
    counts = grouped.size()
    multi_spec_groups = counts[counts > 1].index.tolist()
    print(f"Conditions with >1 Spectrum: {len(multi_spec_groups):,} ({len(multi_spec_groups)/n_groups*100:.1f}%)")
    
    if len(multi_spec_groups) == 0:
        print("No replicates found.")
        return

    # Sample for speed
    sample_size = min(3000, len(multi_spec_groups))
    # Get indices for sampling
    indices = np.random.choice(len(multi_spec_groups), size=sample_size, replace=False)
    sampled_groups = [multi_spec_groups[i] for i in indices]
    
    print(f"Analyzing variance for {sample_size} sampled conditions...")
    
    self_similarities = []
    
    for key in tqdm(sampled_groups):
        group_df = grouped.get_group(key)
        
        # Bin spectra
        vectors = []
        for peaks in group_df['peaks']:
            vec = get_binned_spectrum(peaks)
            # Sanity check: Ensure vector is not zero-sum (unless spectrum was truly empty)
            if np.sum(vec) > 0:
                vectors.append(vec)
        
        if len(vectors) < 2: continue
            
        vectors_np = np.stack(vectors)
        
        # Calculate pairwise similarity
        sim_matrix = cosine_similarity(vectors_np)
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        pairwise_sims = sim_matrix[triu_indices]
        
        self_similarities.extend(pairwise_sims)

    # 4. Report
    if len(self_similarities) == 0:
        print("No valid comparisons could be made (check data parsing).")
        return

    avg_self_sim = np.mean(self_similarities)
    median_self_sim = np.median(self_similarities)
    
    print("\n" + "="*50)
    print("CONDITIONAL VARIANCE REPORT (CORRECTED)")
    print("="*50)
    print(f"\nSpectral Consistency:")
    print(f"  Mean Similarity:   {avg_self_sim:.4f}")
    print(f"  Median Similarity: {median_self_sim:.4f}")
    print(f"  Min Similarity:    {np.min(self_similarities):.4f}")
    
    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(self_similarities, bins=50, kde=True, color='blue')
    plt.axvline(avg_self_sim, color='red', linestyle='--', label=f'Mean: {avg_self_sim:.3f}')
    plt.title('Conditional Spectral Variance\n(Similarity of Replicates: Same Structure + Same Physics)', fontsize=14)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Count of Pairs', fontsize=12)
    plt.legend()
    
    out_path = os.path.join(args.output_dir, "conditional_variance_plot_fixed.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to: {out_path}")
    
    if avg_self_sim > 0.85:
        print("\n[CONCLUSION]: LOW NOISE. (Good)")
        print("Fixing the collision energy makes the spectra consistent.")
        print("Your 'Physics-Aware' model strategy is sound.")
    else:
        print("\n[CONCLUSION]: HIGH INTRINSIC NOISE. (Bad)")
        print(f"Similarity is only {avg_self_sim:.2f} even with identical conditions.")
        print("Instrument noise or uncaptured metadata is limiting performance.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to augmented_msg_df.feather")
    parser.add_argument("--output_dir", type=str, default="./analysis_plots")
    args = parser.parse_args()
    analyze_variance(args)