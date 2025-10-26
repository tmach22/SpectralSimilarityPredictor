import pandas as pd
import numpy as np
import argparse
import time

def build_taxonomy_columns_v2(merged_pairs_path, output_path):
    """
    Adds all hierarchical taxonomy columns to the merged pairs file,
    using robust InChIKey comparison for molecular identity.
    """
    
    print(f"--- 1. Loading Merged Pairs Data ---")
    try:
        start_time = time.time()
        df = pd.read_feather(merged_pairs_path)
        print(f"Successfully loaded {len(df):,} records from {merged_pairs_path} in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error loading {merged_pairs_path}: {e}")
        return

    print("\n--- 2. Handling NaNs (CRITICAL for InChIKey) ---")
    # This is the most important step for robust comparison
    fill_map = {
        'inchikey_main': 'NO_INCHI_1',
        'inchikey_sub': 'NO_INCHI_2',
        'instrument_main': 'unknown_inst_1',
        'instrument_sub': 'unknown_inst_2',
        'adduct_main': 'unknown_adduct_1',
        'adduct_sub': 'unknown_adduct_2',
        'collision_energy_main': -1.0,
        'collision_energy_sub': -2.0
    }
    
    cols_to_fill = {col: val for col, val in fill_map.items() if col in df.columns}
    df = df.fillna(value=cols_to_fill)
    print(f"Filled NaNs in {len(cols_to_fill)} key metadata columns.")
    
    # Ensure InChIKeys are strings so we can slice them
    df['inchikey_main'] = df['inchikey_main'].astype(str)
    df['inchikey_sub'] = df['inchikey_sub'].astype(str)

    print("\n--- 3. Building All Taxonomy Columns (InChIKey-based) ---")
    start_time = time.time()

    # --- L1: Molecular Identity (NEW ROBUST LOGIC) ---
    print("Building L1: Molecular Identity (taxonomy_L1)...")
    
    conditions_L1 = [
        # Condition 1: Full InChIKeys match
        (df['inchikey_main'] == df['inchikey_sub']) & (df['inchikey_main'] != 'NO_INCHI_1'),
        
        # Condition 2: Only Connectivity (first 14 chars) matches
        (df['inchikey_main'].str[0:14] == df['inchikey_sub'].str[0:14]) &
        (df['inchikey_main'] != df['inchikey_sub']) &
        (df['inchikey_main'] != 'NO_INCHI_1'),
        
        # Condition 3: Connectivity does not match
        (df['inchikey_main'].str[0:14] != df['inchikey_sub'].str[0:14])
    ]
    
    choices_L1 = [
        'Identical-Molecule',  # Our new "Intra-Molecule"
        'Stereoisomer',        # New powerful category!
        'Different-Structure'  # Our new "Inter-Molecule"
    ]
    
    df['taxonomy_L1'] = np.select(conditions_L1, choices_L1, default='Unknown/NoInChI')

    # --- L2, L3, L4 (These are unchanged) ---
    print("Building L2, L3, L4...")
    df['taxonomy_L2'] = np.where(df['instrument_main'] == df['instrument_sub'], 'Same-Instrument', 'Cross-Instrument')
    df['taxonomy_L3'] = np.where(df['adduct_main'] == df['adduct_sub'], 'Same-Adduct', 'Cross-Adduct')
    df['taxonomy_L4'] = np.where(df['collision_energy_main'] == df['collision_energy_sub'], 'Same-Energy', 'Cross-Energy')

    print(f"Taxonomy column build complete in {time.time() - start_time:.2f}s")

    print("\n--- 4. Final Data Overview (New L1) ---")
    print("\nNew L1 Distribution (based on InChIKey):")
    print(df['taxonomy_L1'].value_counts(dropna=False))

    print("\n--- 5. Saving Final File ---")
    try:
        start_time = time.time()
        df.to_feather(output_path)
        print(f"\nSuccessfully saved final data with *robust* taxonomy to: {output_path} in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\nError saving file to {output_path}: {e}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add all taxonomy columns (v2, InChIKey-based) to the merged pairs file.")
    parser.add_argument("--merged_pairs_path", type=str, required=True,
                        help="Path to your merged pairs file (e.g., train_pairs_with_metadata.feather).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the new file with all taxonomy columns (e.g., taxanomy_file_v2.feather).")
    
    args = parser.parse_args()
    print("--- Starting Taxonomy Column Building Script (v2, InChIKey-based) ---")
    build_taxonomy_columns_v2(args.merged_pairs_path, args.output_path)
    print("--- Script Finished ---")