import pandas as pd
import numpy as np
import os

def check_data_integrity(pairs_path, spec_path, mol_path):
    print("--- 1. Loading Datafiles ---")
    
    # Load Pairs
    print(f"Loading Pairs: {pairs_path}")
    df_pairs = pd.read_feather(pairs_path)
    print(f"Pairs loaded: {len(df_pairs)} records")
    
    # Load Spectrum Data
    print(f"Loading Specs: {spec_path}")
    df_spec = pd.read_pickle(spec_path)
    # Create a set of valid Spec IDs for O(1) lookup
    valid_spec_ids = set(df_spec['spec_id'].unique())
    print(f"Unique Valid Spec IDs: {len(valid_spec_ids)}")
    
    # Load Molecule Data
    print(f"Loading Mols: {mol_path}")
    df_mol = pd.read_pickle(mol_path)
    # Create a set of valid Mol IDs
    valid_mol_ids = set(df_mol['mol_id'].unique())
    print(f"Unique Valid Mol IDs: {len(valid_mol_ids)}")

    print("\n--- 2. Checking Referential Integrity ---")
    
    # Check 1: Do all Spec IDs in pairs exist in spec_df?
    # We check both 'name_main' and 'name_sub' against 'spec_id'
    
    missing_A = df_pairs[~df_pairs['name_main'].isin(valid_spec_ids)]
    missing_B = df_pairs[~df_pairs['name_sub'].isin(valid_spec_ids)]
    
    if len(missing_A) > 0 or len(missing_B) > 0:
        print(f"CRITICAL ERROR: Found pairs pointing to missing spectra!")
        print(f" - Missing Spec IDs in 'name_main': {len(missing_A)}")
        print(f" - Missing Spec IDs in 'name_sub':  {len(missing_B)}")
        
        # Print a few examples
        if len(missing_A) > 0:
            print(f"Example missing ID (Main): {missing_A['name_main'].iloc[0]}")
        if len(missing_B) > 0:
            print(f"Example missing ID (Sub):  {missing_B['name_sub'].iloc[0]}")
    else:
        print("PASS: All Spec IDs in pairs exist in spec_df.")

    # Check 2: Do all Spectra point to valid Molecules?
    # Your spec_df has a 'mol_id' column. Does that column line up with mol_df?
    
    missing_mols = df_spec[~df_spec['mol_id'].isin(valid_mol_ids)]
    
    if len(missing_mols) > 0:
        print(f"\nCRITICAL ERROR: Found spectra pointing to missing molecules!")
        print(f" - Count: {len(missing_mols)}")
        print(f" - Example Spec ID: {missing_mols['spec_id'].iloc[0]} points to Mol ID: {missing_mols['mol_id'].iloc[0]}")
    else:
        print("PASS: All Spectra point to valid Molecules.")

    # Check 3: Check for Empty Peaks
    # Even if the ID exists, the 'peaks' column might be empty or None
    
    print("\n--- 3. Checking for Empty/Null Peaks ---")
    
    # Check for actual None or empty list []
    # Note: Depending on how it's saved, empty might be [] or NaN.
    
    empty_peaks = df_spec[df_spec['peaks'].apply(lambda x: x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0))]
    
    if len(empty_peaks) > 0:
        print(f"WARNING: Found {len(empty_peaks)} spectra with EMPTY peaks!")
        print(empty_peaks[['spec_id', 'peaks']].head())
        
        # Check if any of these empty spectra are actually used in your pairs
        bad_ids = set(empty_peaks['spec_id'])
        affected_pairs = df_pairs[df_pairs['name_main'].isin(bad_ids) | df_pairs['name_sub'].isin(bad_ids)]
        print(f" -> These empty spectra affect {len(affected_pairs)} rows in your test pairs.")
    else:
        print("PASS: No empty peak lists found.")

if __name__ == "__main__":
    # Update paths here
    pairs = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/binary_07_dataset_casmi.feather"
    mol = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df_casmi.pkl"   # Update this
    spec = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df_casmi.pkl" # Update this
    
    check_data_integrity(pairs, spec, mol)