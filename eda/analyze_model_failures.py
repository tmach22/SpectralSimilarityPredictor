import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

# --- CONFIGURATION ---
STRUCTURAL_SIMILARITY_THRESHOLD = 0.85 

def calculate_tanimoto(smi1, smi2):
    """Calculates Tanimoto similarity between two SMILES strings."""
    try:
        if pd.isna(smi1) or pd.isna(smi2): return 0.0
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None: return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def analyze_failures(args):
    print(f"--- Starting Deep Dive Analysis ---")
    
    # 1. Load Data
    print(f"Loading Results: {args.results_csv}")
    df_res = pd.read_csv(args.results_csv)
    
    print(f"Loading Metadata Files...")
    # Using read_pickle as per your file format description
    df_spec = pd.read_pickle(args.spec_meta_path)
    df_mol = pd.read_pickle(args.mol_meta_path)

    # 2. Build Efficient Lookups (Relational Join)
    print("Building Relational Lookups...")
    
    # A. Mol ID -> SMILES Mapping
    # df_mol has ['mol_id', 'smiles']
    mol_to_smiles = df_mol.set_index('mol_id')['smiles'].to_dict()
    
    # B. Spec ID -> Mol ID Mapping
    # df_spec has ['spec_id', 'mol_id']
    spec_to_mol = df_spec.set_index('spec_id')['mol_id'].to_dict()
    
    # C. Spec ID -> Metadata Mapping
    # df_spec has ['spec_id', 'inst_type', 'prec_type', 'ace']
    # We rename keys to match our logic: inst_type->instrument, prec_type->adduct, ace->collision_energy
    meta_lookup = {}
    
    # Iterate to build dictionary (safer than to_dict('index') for huge dfs if filtering needed)
    # But for 200k rows, to_dict is fine.
    temp_meta = df_spec.set_index('spec_id')[['inst_type', 'prec_type', 'ace']].to_dict('index')
    
    # Clean up keys for easier access later
    for spec_id, data in temp_meta.items():
        meta_lookup[spec_id] = {
            'instrument': data.get('inst_type'),
            'adduct': data.get('prec_type'),
            'collision_energy': data.get('ace')
        }

    # Helper to get SMILES from Spec ID
    def get_smiles(spec_id):
        mol_id = spec_to_mol.get(spec_id)
        if mol_id is not None:
            return mol_to_smiles.get(mol_id)
        return None

    # 3. Filter for False Positives (FP)
    # FP: True Label = 0, Predicted Label = 1
    # Ensure we use the exact column names from your results csv
    if 'true_label' in df_res.columns and 'predicted_label' in df_res.columns:
        df_fp = df_res[(df_res['true_label'] == 0.0) & (df_res['predicted_label'] == 1.0)].copy()
    else:
        # Fallback for old CSVs
        print("Warning: Standard columns not found. Checking for 'label'/'pred'...")
        df_fp = df_res[(df_res['label'] == 0) & (df_res['pred'] == 1)].copy()
    
    print(f"\n--- Analysis Scope ---")
    print(f"Total Test Pairs: {len(df_res):,}")
    print(f"Total False Positives (FPs): {len(df_fp):,} ({len(df_fp)/len(df_res)*100:.2f}%)")
    
    if len(df_fp) == 0:
        print("No False Positives found! (Perfect model? Or check thresholds?)")
        return

    # 4. Enrich FPs with Diagnostics
    print(f"Diagnosing {len(df_fp)} False Positives...")
    
    tanimotos = []
    reasons = []
    
    # Counters for missing data
    missing_smiles = 0
    
    for idx, row in tqdm(df_fp.iterrows(), total=len(df_fp)):
        id_a = row['name_main']
        id_b = row['name_sub']
        
        # A. Structure Check (SMILES)
        smi_a = get_smiles(id_a)
        smi_b = get_smiles(id_b)
        
        if smi_a is None or smi_b is None:
            missing_smiles += 1
            tanimoto = 0.0
        else:
            tanimoto = calculate_tanimoto(smi_a, smi_b)
            
        tanimotos.append(tanimoto)
        
        # B. Metadata Check
        meta_a = meta_lookup.get(id_a, {})
        meta_b = meta_lookup.get(id_b, {})
        
        # Normalize strings for comparison
        inst_a = str(meta_a.get('instrument', 'UNK')).upper()
        inst_b = str(meta_b.get('instrument', 'UNK')).upper()
        
        adduct_a = str(meta_a.get('adduct', 'UNK')).upper()
        adduct_b = str(meta_b.get('adduct', 'UNK')).upper()
        
        # C. Categorize Failure
        fail_type = "Unknown/Hallucination"
        
        # Priority 1: Structural Mimic (Chemist would also be confused)
        if tanimoto >= STRUCTURAL_SIMILARITY_THRESHOLD:
            fail_type = "Structural Mimic"
            
            # Sub-case: Physics Mismatch (Same molecule, different settings)
            # Threshold 0.99 allows for minor stereo/tautomer diffs that are effectively same mol
            if tanimoto > 0.99:
                if adduct_a != adduct_b and adduct_a != 'UNK' and adduct_b != 'UNK':
                    fail_type = "Adduct Mismatch (Same Mol)"
                elif inst_a != inst_b and inst_a != 'UNK' and inst_b != 'UNK':
                    fail_type = "Instrument Mismatch (Same Mol)"
        
        reasons.append(fail_type)

    df_fp['tanimoto_sim'] = tanimotos
    df_fp['failure_reason'] = reasons
    
    if missing_smiles > 0:
        print(f"\nWarning: Could not find SMILES for {missing_smiles} pairs. Treated as 0.0 similarity.")

    # 5. Generate Report
    print("\n--- FALSE POSITIVE BREAKDOWN ---")
    counts = df_fp['failure_reason'].value_counts()
    percentages = df_fp['failure_reason'].value_counts(normalize=True) * 100
    
    print(f"{'Failure Category':<30} | {'Count':<10} | {'Percent':<10}")
    print("-" * 55)
    for cat, count in counts.items():
        print(f"{cat:<30} | {count:<10} | {percentages[cat]:.1f}%")
        
    # Additional Stats
    avg_sim = np.mean(tanimotos)
    print(f"\nAverage Tanimoto of FPs: {avg_sim:.4f}")

    # 6. Visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot 1: Tanimoto Distribution
    plt.figure(figsize=(10, 6))
    # Filter out 0.0s that might be due to missing SMILES if they dominate
    valid_tanimotos = [t for t in tanimotos if t > 0.0]
    if valid_tanimotos:
        sns.histplot(valid_tanimotos, bins=20, kde=True, color='red')
        plt.axvline(STRUCTURAL_SIMILARITY_THRESHOLD, color='black', linestyle='--', label='Structural Mimic Threshold')
        plt.title(f"Why is the model confused?\nStructural Similarity of False Positive Pairs")
        plt.xlabel("Tanimoto Similarity (Chemical Structure)")
        plt.ylabel("Count of False Positives")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "fp_structural_distribution_mona.png"))
        plt.close()
    
    # Plot 2: Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title("Dominant Failure Modes")
    plt.xlabel("Number of Errors")
    plt.savefig(os.path.join(args.output_dir, "fp_failure_categories_mona.png"))
    plt.close()

    # 7. Save Enriched Data
    save_path = os.path.join(args.output_dir, "false_positives_diagnosed_mona.csv")
    df_fp.to_csv(save_path, index=False)
    print(f"\nDetailed forensic file saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True, help="Path to your results_X.csv")
    parser.add_argument("--spec_meta_path", required=True, help="Path to spec_df.pkl")
    parser.add_argument("--mol_meta_path", required=True, help="Path to mol_df.pkl")
    parser.add_argument("--output_dir", default="./analysis_results")
    
    args = parser.parse_args()
    analyze_failures(args)