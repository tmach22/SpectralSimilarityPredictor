import pandas as pd
import numpy as np
import argparse

def analyze_pairs(pairs_file, metadata_file):
    print("Loading Metadata...")
    # Load metadata (only need ID and experimental columns)
    meta_df = pd.read_feather(metadata_file)
    
    # Ensure ID is string for merging
    meta_df['spectrum_id'] = meta_df['spectrum_id'].astype(str)
    
    # Create a clean lookup dataframe
    # We strip whitespace and lowercase strings to ensure fair comparison
    cols_to_keep = ['spectrum_id', 'collision_energy', 'adduct', 'instrument_type']
    # If instrument_type is missing, check for 'instrument'
    if 'instrument_type' not in meta_df.columns and 'instrument' in meta_df.columns:
        meta_df.rename(columns={'instrument': 'instrument_type'}, inplace=True)
        
    lookup = meta_df[cols_to_keep].copy()
    
    # normalize strings for comparison
    lookup['adduct'] = lookup['adduct'].astype(str).str.strip()
    lookup['instrument_type'] = lookup['instrument_type'].astype(str).str.lower().str.strip()
    lookup['collision_energy'] = pd.to_numeric(lookup['collision_energy'], errors='coerce')

    print(f"Metadata loaded: {len(lookup)} spectra.")

    print("\nLoading Pairs...")
    # Load pairs file (assuming columns: name_main, name_sub, ...)
    pairs_df = pd.read_feather(pairs_file)
    print(f"Pairs loaded: {len(pairs_df)} pairs.")
    
    # Ensure ID columns match metadata type
    pairs_df['name_main'] = pairs_df['name_main'].astype(str)
    pairs_df['name_sub'] = pairs_df['name_sub'].astype(str)

    # --- MERGE METADATA ---
    # Merge metadata for Spectrum 1 (Main)
    merged = pairs_df.merge(
        lookup.add_suffix('_1'), 
        left_on='name_main', 
        right_on='spectrum_id_1', 
        how='left'
    )
    
    # Merge metadata for Spectrum 2 (Sub)
    merged = merged.merge(
        lookup.add_suffix('_2'), 
        left_on='name_sub', 
        right_on='spectrum_id_2', 
        how='left'
    )

    # --- ANALYZE CONSISTENCY ---
    print("\n==============================================")
    print("      EXPERIMENTAL CONSISTENCY REPORT")
    print("==============================================")

    # 1. Adduct Consistency
    # We compare string equality
    merged['same_adduct'] = (merged['adduct_1'] == merged['adduct_2'])
    adduct_match_pct = merged['same_adduct'].mean() * 100
    print(f"Adduct Match:           {adduct_match_pct:.2f}%")
    
    # 2. Instrument Consistency
    merged['same_inst'] = (merged['instrument_type_1'] == merged['instrument_type_2'])
    inst_match_pct = merged['same_inst'].mean() * 100
    print(f"Instrument Match:       {inst_match_pct:.2f}%")

    # 3. Collision Energy Consistency
    # Exact match
    merged['same_ce_exact'] = (merged['collision_energy_1'] == merged['collision_energy_2'])
    
    # Relaxed match (within +/- 1.0 eV) to account for float drift (35.0 vs 35.0001)
    # We also handle NaNs: If both are NaN, they 'match' in relaxed mode if you choose, 
    # but here we check numeric proximity.
    ce_diff = (merged['collision_energy_1'] - merged['collision_energy_2']).abs()
    merged['same_ce_relaxed'] = ce_diff <= 1.0
    
    ce_match_pct = merged['same_ce_exact'].mean() * 100
    ce_relaxed_pct = merged['same_ce_relaxed'].mean() * 100
    
    print(f"CE Match (Exact):       {ce_match_pct:.2f}%")
    print(f"CE Match (±1.0 eV):     {ce_relaxed_pct:.2f}%")
    
    # 4. Strict "All Match" Count
    # How many pairs would have survived your grouping script?
    merged['all_match'] = merged['same_adduct'] & merged['same_inst'] & merged['same_ce_relaxed']
    total_strict_pairs = merged['all_match'].sum()
    print(f"----------------------------------------------")
    print(f"Pairs passing STRICT grouping: {total_strict_pairs} / {len(pairs_df)} ({merged['all_match'].mean()*100:.2f}%)")
    print("==============================================")

    # --- FAILURE ANALYSIS ---
    # If count is 0, show examples of why they failed
    if total_strict_pairs == 0 or total_strict_pairs < 10:
        print("\n[DEBUG] Top Mismatches (Why are they failing?):")
        mismatches = merged[~merged['all_match']].head(5)
        cols_show = ['name_main', 'adduct_1', 'instrument_type_1', 'collision_energy_1',
                     'name_sub', 'adduct_2', 'instrument_type_2', 'collision_energy_2']
        print(mismatches[cols_show].to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_file", required=True, help="Path to existing pairs feather file")
    parser.add_argument("--metadata_file", required=True, help="Path to augmented_msg_df.feather")
    args = parser.parse_args()

    analyze_pairs(args.pairs_file, args.metadata_file)