import pandas as pd
import numpy as np
import argparse
import os

def analyze_isomers(args):
    print("--- Starting Isomer Analysis ---")
    
    # 1. Load Data
    print(f"Loading pairs from: {args.pairs_path}")
    pairs_df = pd.read_feather(args.pairs_path)
    
    print(f"Loading metadata...")
    spec_df = pd.read_pickle(args.spec_data_path)
    mol_df = pd.read_pickle(args.mol_data_path)
    
    # 2. Create Lookups for Speed
    # Map spec_id -> mol_id
    spec_to_mol = spec_df.set_index('spec_id')['mol_id'].to_dict()
    
    # Map mol_id -> InChIKey and Exact Mass
    # We use 'inchikey_s' (standard InChIKey) if available
    mol_to_inchikey = mol_df.set_index('mol_id')['inchikey_s'].to_dict()
    mol_to_mw = mol_df.set_index('mol_id')['exact_mw'].to_dict()
    
    # 3. Define Helper Function
    def categorize_pair(row):
        id_a = row['name_main']
        id_b = row['name_sub']
        
        mol_a = spec_to_mol.get(id_a)
        mol_b = spec_to_mol.get(id_b)
        
        if mol_a is None or mol_b is None:
            return "Unknown"
            
        ik_a = mol_to_inchikey.get(mol_a, "")
        ik_b = mol_to_inchikey.get(mol_b, "")
        
        # Check InChIKey (Identical Structure)
        if ik_a == ik_b:
            return "Identical"
            
        # Check Stereoisomers (Same connectivity/skeleton, different stereochemistry)
        # The first block of InChIKey (14 chars) encodes the skeleton.
        if ik_a[:14] == ik_b[:14]:
            return "Stereoisomer"
            
        # Check Constitutional Isomers (Same Mass/Formula, different skeleton)
        mw_a = mol_to_mw.get(mol_a, 0.0)
        mw_b = mol_to_mw.get(mol_b, 0.0)
        
        # Use a small tolerance for float comparison of mass
        if abs(mw_a - mw_b) < 0.001:
            return "Constitutional Isomer"
            
        return "Different Molecule"

    # 4. Analyze High vs. Medium Bins
    print("\nCategorizing pairs... (This may take a moment)")
    
    # Filter for the bins we care about
    high_sim = pairs_df[pairs_df['cosine_similarity'] >= 0.85].copy()
    med_sim = pairs_df[(pairs_df['cosine_similarity'] >= 0.65) & (pairs_df['cosine_similarity'] < 0.85)].copy()
    
    # Apply categorization
    high_sim['category'] = high_sim.apply(categorize_pair, axis=1)
    med_sim['category'] = med_sim.apply(categorize_pair, axis=1)
    
    # 5. Print Report
    def print_stats(df, name):
        total = len(df)
        counts = df['category'].value_counts()
        print(f"\n--- {name} Similarity Pairs (Total: {total}) ---")
        for cat in ["Identical", "Stereoisomer", "Constitutional Isomer", "Different Molecule"]:
            count = counts.get(cat, 0)
            pct = (count / total) * 100
            print(f"  {cat:<25}: {count:<6} ({pct:.1f}%)")

    print_stats(high_sim, "HIGH (>= 0.85)")
    print_stats(med_sim, "MEDIUM (0.65 - 0.85)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to binary_07_dataset_test.feather")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to spec_df_2.pkl")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to mol_df_2.pkl")
    
    args = parser.parse_args()
    analyze_isomers(args)