import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("Error: RDKit not found. Please install it to calculate fingerprints.")
    exit(1)

def calculate_tanimoto_batch(mol_list_a, mol_list_b):
    """
    Computes Tanimoto similarity for lists of RDKit molecules.
    Returns a list of scores.
    """
    scores = []
    for m1, m2 in zip(mol_list_a, mol_list_b):
        if m1 is None or m2 is None:
            scores.append(0.0)
            continue
            
        # Generate Morgan Fingerprints (Radius 2, 2048 bits)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        scores.append(sim)
    return scores

def analyze_structure(args):
    print("--- Starting Structural Ambiguity Analysis ---")
    
    # 1. Load Data
    print(f"Loading pairs from: {args.pairs_path}")
    pairs_df = pd.read_feather(args.pairs_path)
    
    print(f"Loading metadata...")
    spec_df = pd.read_pickle(args.spec_data_path)
    mol_df = pd.read_pickle(args.mol_data_path)
    
    # Create Lookups
    spec_to_mol = spec_df.set_index('spec_id')['mol_id'].to_dict()
    mol_to_obj = mol_df.set_index('mol_id')['mol'].to_dict()
    
    # 2. Define Bins
    # Bin 1: Medium Spectral Similarity (0.65 - 0.85)
    # Bin 2: High Spectral Similarity (>= 0.85)
    
    med_mask = (pairs_df['cosine_similarity'] >= 0.65) & (pairs_df['cosine_similarity'] < 0.85)
    high_mask = (pairs_df['cosine_similarity'] >= 0.85)
    
    df_med = pairs_df[med_mask].copy()
    df_high = pairs_df[high_mask].copy()
    
    print(f"\nFound {len(df_med):,} Medium pairs and {len(df_high):,} High pairs.")
    
    # 3. Calculate Structural Similarity (Tanimoto)
    def get_mols(df):
        mols_a = []
        mols_b = []
        for _, row in df.iterrows():
            id_a, id_b = row['name_main'], row['name_sub']
            mol_id_a = spec_to_mol.get(id_a)
            mol_id_b = spec_to_mol.get(id_b)
            mols_a.append(mol_to_obj.get(mol_id_a))
            mols_b.append(mol_to_obj.get(mol_id_b))
        return mols_a, mols_b

    print("Calculating Tanimoto Scores for Medium Bin...")
    ma_med, mb_med = get_mols(df_med)
    med_tanimoto = calculate_tanimoto_batch(ma_med, mb_med)
    
    print("Calculating Tanimoto Scores for High Bin...")
    ma_high, mb_high = get_mols(df_high)
    high_tanimoto = calculate_tanimoto_batch(ma_high, mb_high)
    
    # 4. Plotting
    print("\nGenerating Distribution Plot...")
    plt.figure(figsize=(12, 6))
    
    sns.histplot(med_tanimoto, color='orange', label='Medium Spectral Sim (0.65-0.85)', stat="density", element="step", fill=False, linewidth=2)
    sns.histplot(high_tanimoto, color='blue', label='High Spectral Sim (>=0.85)', stat="density", element="step", fill=False, linewidth=2)
    
    plt.title('Structural Similarity (Tanimoto) Distribution', fontsize=16)
    plt.xlabel('Tanimoto Similarity (Structure)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(args.output_dir, "structural_ambiguity_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    
    # 5. Statistical Summary
    print("\n" + "="*50)
    print("STRUCTURAL AMBIGUITY REPORT")
    print("="*50)
    
    def print_stats(name, scores):
        scores = np.array(scores)
        print(f"\n--- {name} Spectral Bin ---")
        print(f"  Mean Tanimoto:      {np.mean(scores):.4f}")
        print(f"  Median Tanimoto:    {np.median(scores):.4f}")
        print(f"  Pairs with Tanimoto < 0.8:  {np.mean(scores < 0.8)*100:.1f}% (Structurally Different)")
        print(f"  Pairs with Tanimoto > 0.99: {np.mean(scores > 0.99)*100:.1f}% (Identical)")

    print_stats("Medium (0.65-0.85)", med_tanimoto)
    print_stats("High (>= 0.85)", high_tanimoto)
    print("\n" + "="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to test feather file")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    
    args = parser.parse_args()
    analyze_structure(args)