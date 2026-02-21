import pandas as pd
import argparse
import os
import numpy as np
from pathlib import Path

def normalize_metadata(df):
    """
    Standardizes text columns to ensure compatibility with Fixed Vocabulary.
    """
    # 1. Adducts: Uppercase and strip
    if 'adduct' in df.columns:
        df['adduct'] = df['adduct'].astype(str).str.strip().str.upper()
    
    # 2. Instruments: Uppercase 
    if 'instrument_type' in df.columns:
        df['instrument_type'] = df['instrument_type'].astype(str).str.strip().str.upper()
        
    # 3. Handle Missing CE
    if 'collision_energy' in df.columns:
        df['collision_energy'] = df['collision_energy'].fillna(-1.0)
        
    # 4. CRITICAL FIX: Drop Pre-calculated Fingerprints
    # Mixing fingerprints from different sources is dangerous (mismatched radii/bits).
    # We will regenerate them from SMILES in the pairing step.
    if 'morgan_fingerprint' in df.columns:
        print("   -> Dropping 'morgan_fingerprint' to prevent Type Mismatch & Scientific Inconsistency.")
        df.drop(columns=['morgan_fingerprint'], inplace=True)
        
    # 5. CRITICAL FIX: Ensure Peaks are Lists (not numpy arrays)
    # Arrow hates mixed Numpy/List types. Force conversion to pure Python lists.
    if 'peaks' in df.columns:
        df['peaks'] = df['peaks'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
    return df

def merge_and_analyze(msg_path, mona_path, output_path):
    print(f"--- Loading Datasets ---")
    print(f"1. MassSpecGym: {msg_path}")
    df_msg = pd.read_feather(msg_path)
    print(f"   Rows: {len(df_msg)}")
    
    print(f"2. MoNA Cleaned: {mona_path}")
    df_mona = pd.read_feather(mona_path)
    print(f"   Rows: {len(df_mona)}")

    # --- Step 1: Schema Alignment ---
    common_cols = list(set(df_msg.columns).intersection(set(df_mona.columns)))
    print(f"\n[Schema Check] Keeping {len(common_cols)} common columns.")
    
    df_msg = df_msg[common_cols].copy()
    df_mona = df_mona[common_cols].copy()
    
    # Tag sources
    df_msg['source_dataset'] = 'mass_spec_gym'
    df_mona['source_dataset'] = 'mona_external'

    # --- Step 2: Normalization (Fixes the ArrowInvalid Error) ---
    print("\n[Normalization] Standardizing data types...")
    print("Processing MassSpecGym...")
    df_msg = normalize_metadata(df_msg)
    print("Processing MoNA...")
    df_mona = normalize_metadata(df_mona)

    # --- Step 3: Overlap Analysis ---
    msg_inchis = set(df_msg['inchikey'].unique())
    mona_inchis = set(df_mona['inchikey'].unique())
    
    overlap = msg_inchis.intersection(mona_inchis)
    print(f"\n[Leakage Analysis]")
    print(f"   Unique Molecules (MSG):  {len(msg_inchis)}")
    print(f"   Unique Molecules (MoNA): {len(mona_inchis)}")
    print(f"   OVERLAP (Shared InChIKeys): {len(overlap)}")
    print(f"   New Molecules from MoNA: {len(mona_inchis - overlap)}")

    # --- Step 4: Merge ---
    print(f"\n[Merging] Concatenating datasets...")
    df_merged = pd.concat([df_msg, df_mona], ignore_index=True)
    
    # --- Step 5: ID Handling ---
    if df_merged['spectrum_id'].duplicated().any():
        print(">> WARNING: Duplicate spectrum_ids detected! Renaming...")
        df_merged['spectrum_id'] = "MergedID_" + df_merged.index.astype(str)
    
    # --- Step 6: Save ---
    print(f"\n[Saving] Writing to Feather...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_feather(output_path)
    
    print(f"\n--- Success! ---")
    print(f"Merged Dataset Saved: {output_path}")
    print(f"Total Records: {len(df_merged)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--msg_path", type=str, required=True)
    parser.add_argument("--mona_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()
    merge_and_analyze(args.msg_path, args.mona_path, args.output_path)