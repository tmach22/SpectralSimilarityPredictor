import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_hallucinations(args):
    print(f"--- Deep Dive: Analyzing 'Hallucination' False Positives ---")
    
    # 1. Load Diagnosed FPs
    print(f"Loading Diagnostic File: {args.diagnostic_csv}")
    df_fp = pd.read_csv(args.diagnostic_csv)
    
    # Filter for the "Hallucination" group (Low Tanimoto)
    # Adjust logic based on your exact previous output labels
    df_hallucinations = df_fp[df_fp['failure_reason'] == "Unknown/Hallucination"].copy()
    
    print(f"Total FPs: {len(df_fp)}")
    print(f"Hallucinations to Analyze: {len(df_hallucinations)}")
    
    if len(df_hallucinations) == 0:
        print("No hallucinations found to analyze.")
        return

    # 2. Load Metadata (We need Mass and Peak Counts)
    print("Loading Spec Metadata...")
    df_spec = pd.read_pickle(args.spec_meta_path)
    
    # Create Lookup for Mass (prec_mz) and Peak Count
    # We calculate peak count from the 'peaks' column if it exists, or just check 'prec_mz'
    meta_lookup = {}
    
    # Pre-scan dataframe to extract needed fields
    # We assume 'peaks' is a list of tuples. len(row['peaks']) = num_peaks
    for idx, row in tqdm(df_spec.iterrows(), total=len(df_spec), desc="Indexing Metadata"):
        spec_id = row['spec_id']
        
        # Safe extraction of peak count
        n_peaks = 0
        if isinstance(row['peaks'], list) or isinstance(row['peaks'], np.ndarray):
            n_peaks = len(row['peaks'])
        
        meta_lookup[spec_id] = {
            'mz': float(row['prec_mz']),
            'n_peaks': n_peaks,
            'instrument': row['inst_type']
        }

    # 3. Calculate Physics Metrics
    print("Calculating Physics Metrics for Hallucinations...")
    
    delta_mzs = []
    peak_counts = [] # Store tuple (min_peaks, max_peaks) for the pair
    instruments = []
    
    for idx, row in tqdm(df_hallucinations.iterrows(), total=len(df_hallucinations)):
        id_a = row['name_main']
        id_b = row['name_sub']
        
        info_a = meta_lookup.get(id_a)
        info_b = meta_lookup.get(id_b)
        
        if info_a and info_b:
            # A. Mass Difference
            dmz = abs(info_a['mz'] - info_b['mz'])
            delta_mzs.append(dmz)
            
            # B. Peak Counts (We care if BOTH are low)
            p_a = info_a['n_peaks']
            p_b = info_b['n_peaks']
            peak_counts.append((p_a, p_b))
            
            # C. Instrument (Just tracking A for distribution)
            instruments.append(info_a['instrument'])
        else:
            delta_mzs.append(-1)
            peak_counts.append((-1, -1))
            instruments.append("Unknown")

    df_hallucinations['delta_mz'] = delta_mzs
    df_hallucinations['peaks_A'] = [x[0] for x in peak_counts]
    df_hallucinations['peaks_B'] = [x[1] for x in peak_counts]
    df_hallucinations['min_peaks'] = df_hallucinations[['peaks_A', 'peaks_B']].min(axis=1)

    # 4. Analysis & plotting
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Analysis 1: The Isobar Check (Delta m/z) ---
    # Ideally, FPs should have delta_mz near 0 (Isobars).
    # If delta_mz is huge, the model is ignoring mass.
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_hallucinations['delta_mz'], bins=50, kde=False, color='purple')
    plt.title("Are these Isobars?\nPrecursor m/z Difference of Hallucinations")
    plt.xlabel("Delta m/z (Daltons)")
    plt.ylabel("Count")
    plt.xlim(0, 50) # Zoom in on small differences first
    plt.savefig(os.path.join(args.output_dir, "hallucination_delta_mz.png"))
    plt.close()
    
    # Calculate Isobar Stats
    n_isobars = len(df_hallucinations[df_hallucinations['delta_mz'] < 0.1])
    print(f"\n--- HYPOTHESIS 1: ISOBARS ---")
    print(f"Pairs with Delta m/z < 0.1 Da: {n_isobars} ({n_isobars/len(df_hallucinations)*100:.1f}%)")
    print(f"Pairs with Delta m/z > 10 Da:  {len(df_hallucinations[df_hallucinations['delta_mz'] > 10])}")

    # --- Analysis 2: The Sparse Data Check ---
    # Are these spectra empty?
    plt.figure(figsize=(10, 6))
    sns.histplot(df_hallucinations['min_peaks'], bins=30, color='orange')
    plt.title("Are these Empty Spectra?\nMinimum Peak Count in Pair")
    plt.xlabel("Minimum Number of Peaks in Pair")
    plt.ylabel("Count")
    plt.savefig(os.path.join(args.output_dir, "hallucination_peak_counts.png"))
    plt.close()
    
    n_sparse = len(df_hallucinations[df_hallucinations['min_peaks'] < 5])
    print(f"\n--- HYPOTHESIS 2: SPARSE DATA ---")
    print(f"Pairs with < 5 Peaks: {n_sparse} ({n_sparse/len(df_hallucinations)*100:.1f}%)")

    # --- Analysis 3: Instrument Bias ---
    print(f"\n--- HYPOTHESIS 3: INSTRUMENT BIAS ---")
    print(pd.Series(instruments).value_counts())

    # Save
    save_path = os.path.join(args.output_dir, "hallucinations_deep_dive.csv")
    df_hallucinations.to_csv(save_path, index=False)
    print(f"\nDeep dive data saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic_csv", required=True, help="Path to false_positives_diagnosed.csv")
    parser.add_argument("--spec_meta_path", required=True, help="Path to spec_df.pkl")
    parser.add_argument("--output_dir", default="./analysis_results")
    args = parser.parse_args()
    analyze_hallucinations(args)