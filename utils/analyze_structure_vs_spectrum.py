import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
except ImportError:
    print("Error: RDKit not found. Please install it.")
    exit(1)

def analyze_correlation(args):
    print("--- Starting Structure-Spectrum Correlation Analysis ---")
    
    # 1. Load Data
    print(f"Loading pairs from: {args.pairs_path}")
    pairs_df = pd.read_feather(args.pairs_path)
    
    # Subsample for speed/plotting clarity if needed
    if len(pairs_df) > 10000:
        print(f"Subsampling 10,000 pairs from {len(pairs_df)} for plotting clarity...")
        pairs_df = pairs_df.sample(n=10000, random_state=42)
    
    print(f"Loading metadata...")
    spec_df = pd.read_pickle(args.spec_data_path)
    mol_df = pd.read_pickle(args.mol_data_path)
    
    spec_to_mol = spec_df.set_index('spec_id')['mol_id'].to_dict()
    mol_to_obj = mol_df.set_index('mol_id')['mol'].to_dict()
    
    # 2. Calculate Tanimoto Scores
    print("Calculating Tanimoto Similarity for all pairs...")
    tanimoto_scores = []
    spectral_scores = []
    
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        id_a, id_b = row['name_main'], row['name_sub']
        mol_id_a = spec_to_mol.get(id_a)
        mol_id_b = spec_to_mol.get(id_b)
        
        m1 = mol_to_obj.get(mol_id_a)
        m2 = mol_to_obj.get(mol_id_b)
        
        if m1 and m2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            tanimoto_scores.append(sim)
            spectral_scores.append(row['cosine_similarity'])
            
    # 3. Create DataFrame for Analysis
    df_analysis = pd.DataFrame({
        'Tanimoto': tanimoto_scores,
        'Spectral': spectral_scores
    })
    
    # 4. Generate Scatter Plot
    print("\nGenerating Correlation Plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_analysis, x='Tanimoto', y='Spectral', alpha=0.3, s=10, color='purple')
    
    # Add trend line
    sns.regplot(data=df_analysis, x='Tanimoto', y='Spectral', scatter=False, color='black', line_kws={'linestyle':'--'})
    
    plt.title('Structure (Tanimoto) vs. Spectrum (Cosine) Similarity', fontsize=16)
    plt.xlabel('Structural Similarity (Tanimoto)', fontsize=14)
    plt.ylabel('Spectral Similarity (Cosine)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "structure_spectrum_correlation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    
    # 5. Variance Analysis (The "Vertical Slice" Check)
    print("\n" + "="*60)
    print("VARIANCE ANALYSIS (Does same structure mean same spectrum?)")
    print("="*60)
    
    # Bin Tanimoto scores to analyze variance within fixed structural ranges
    bins = [0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    df_analysis['Tanimoto_Bin'] = pd.cut(df_analysis['Tanimoto'], bins=bins, labels=labels)
    
    for label in labels:
        subset = df_analysis[df_analysis['Tanimoto_Bin'] == label]
        if len(subset) == 0: continue
        
        spec_mean = subset['Spectral'].mean()
        spec_std = subset['Spectral'].std()
        spec_min = subset['Spectral'].min()
        spec_max = subset['Spectral'].max()
        
        print(f"\n--- Pairs with Tanimoto {label} (n={len(subset)}) ---")
        print(f"  Spectral Sim Range:  {spec_min:.2f} to {spec_max:.2f}")
        print(f"  Spectral Sim Mean:   {spec_mean:.2f}")
        print(f"  Spectral Sim StdDev: {spec_std:.4f}  <-- High Value = High Ambiguity")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to any pairs feather file")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./analysis_plots")
    
    args = parser.parse_args()
    analyze_correlation(args)