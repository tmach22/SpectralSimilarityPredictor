import pandas as pd
import numpy as np
import ms_entropy
import os
from tqdm import tqdm

# Enable progress bars for pandas apply operations
tqdm.pandas()

def _get_processed_spectrum_for_entropy(raw_peaks, prec_mz, max_peaks=60, parent_intensity=1.1):
    """
    Replicates the EXACT preprocessing logic from generate_phase2_data.py
    so the ground-truth target perfectly matches the network's input features.
    """
    if not isinstance(raw_peaks, list) or len(raw_peaks) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # 1. Sort by intensity and take top (max_peaks - 1)
    peaks_list = sorted(raw_peaks, key=lambda x: x[1], reverse=True)[:max_peaks - 1]

    # 2. Extract m/z and intensity (Keep raw Da for ms_entropy tolerance calculations)
    processed_peaks = [[mz, float(intensity)] for mz, intensity in peaks_list]

    # 3. Inject the precursor ion (exactly as MassFormer sees it)
    if prec_mz > 0:
        processed_peaks.append([prec_mz, parent_intensity])

    # 4. Convert to numpy array and sort strictly by m/z (required by ms_entropy)
    p_array = np.array(processed_peaks, dtype=np.float32)
    p_array = p_array[p_array[:, 0].argsort()]

    return p_array

def calculate_entropy_sim(row, spec_dict, mass_tolerance=0.01):
    """
    Fetches processed peaks and calculates Spectral Entropy Similarity.
    """
    # Extract raw data for both spectra
    main_info = spec_dict.get(row['name_main'], {})
    sub_info = spec_dict.get(row['name_sub'], {})
    
    # Process both spectra using the MassFormer dataset logic
    p_main = _get_processed_spectrum_for_entropy(
        raw_peaks=main_info.get('peaks', []), 
        prec_mz=float(main_info.get('prec_mz', 0.0))
    )
    
    p_sub = _get_processed_spectrum_for_entropy(
        raw_peaks=sub_info.get('peaks', []), 
        prec_mz=float(sub_info.get('prec_mz', 0.0))
    )
    
    # If either spectrum is completely empty, similarity is 0.0
    if len(p_main) == 0 or len(p_sub) == 0:
        return 0.0
    
    # Calculate the exact Spectral Entropy Similarity (Stein Dot Product)
    sim = ms_entropy.calculate_entropy_similarity(p_main, p_sub, ms2_tolerance_in_da=mass_tolerance)
    
    return float(sim)

def process_datasets():
    # --- SETUP PATHS ---
    base_dir = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor"
    spec_df_path = os.path.join(base_dir, "mass_spec_gym_data", "spec_df_COMBINED.pkl")
    
    # List the splits you need to update
    feather_files = [
        os.path.join(base_dir, "data_splits", "stratified_binary_07_dataset_train.feather"),
        os.path.join(base_dir, "data_splits", "stratified_binary_07_dataset_val.feather"),
        os.path.join(base_dir, "data_splits", "stratified_binary_07_dataset_test.feather"),
        os.path.join(base_dir, "data_splits", "stratified_binary_07_unseen_nist.feather")
    ]
    
    # 1. Load the raw peaks and precursor m/z into memory
    print(f"[*] Loading raw spectra database from: {spec_df_path}")
    spec_df = pd.read_pickle(spec_df_path)
    
    # Create a highly efficient dictionary mapping spec_id -> {'peaks': [...], 'prec_mz': 123.4}
    spec_dict = spec_df.set_index('spec_id')[['peaks', 'prec_mz']].to_dict('index')
    print(f"[+] Loaded {len(spec_dict)} unique spectra into memory.\n")

    # ==========================================================
    # BINARY CLASSIFICATION THRESHOLD
    # Adjust this value based on your discussion with the expert.
    # ==========================================================
    THRESHOLD = 0.75 

    # 2. Process each dataset split
    for feather_path in feather_files:
        if not os.path.exists(feather_path):
            print(f"[-] Skipping {os.path.basename(feather_path)} (File not found).")
            continue
            
        print(f"[*] Processing Dataset: {os.path.basename(feather_path)}")
        df = pd.read_feather(feather_path)
        
        # Calculate the metric using the blazing-fast ms_entropy C++ backend
        print("    -> Calculating aligned Spectral Entropy Similarities...")
        df['entropy_similarity'] = df.progress_apply(
            calculate_entropy_sim, 
            axis=1, 
            args=(spec_dict, 0.01) # 0.01 Da tolerance exactly matches your cleavage logic
        )
        
        # Generate the binary label based on the threshold
        print(f"    -> Applying Binary Classification Threshold (T >= {THRESHOLD})...")
        df['entropy_label'] = (df['entropy_similarity'] >= THRESHOLD).astype(int)
        
        # Save the updated DataFrame back to the original file
        df.to_feather(feather_path)
        print(f"[+] Successfully appended 'entropy_similarity' and 'entropy_label'. Saved to {feather_path}\n")

if __name__ == '__main__':
    process_datasets()
    print("[*] All datasets updated. The ground truth targets now perfectly align with the network inputs.")