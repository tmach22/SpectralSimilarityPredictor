import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import os

def get_unique_smiles(pairs_df_paths, spectra_df):
    """
    Loads one or more pair files, gets all unique spectrum IDs,
    and maps them to a set of unique, valid SMILES strings.
    """
    print(f"Processing {len(pairs_df_paths)} pair file(s)...")
    
    # Create the ID-to-SMILES lookup dictionary from the main spectra file
    id_to_smiles = spectra_df.set_index('spectrum_id')['smiles'].to_dict()
    
    all_spec_ids = set()
    for fpath in pairs_df_paths:
        try:
            pairs_df = pd.read_feather(fpath)
            # Add all unique IDs from both columns to the set
            all_spec_ids.update(pairs_df['name_main'])
            all_spec_ids.update(pairs_df['name_sub'])
        except Exception as e:
            print(f"Warning: Could not read {fpath}. Error: {e}")
            
    print(f"Found {len(all_spec_ids)} unique spectrum IDs.")
    
    # Map IDs to SMILES strings, filtering out any missing/invalid ones
    unique_smiles_set = set()
    for spec_id in all_spec_ids:
        smi = id_to_smiles.get(spec_id)
        if smi and pd.notna(smi):
            unique_smiles_set.add(smi)
            
    print(f"Mapped to {len(unique_smiles_set)} unique SMILES strings.")
    return unique_smiles_set

def generate_fingerprints(smiles_set, fpgen):
    """
    Converts a set of SMILES strings to a list of RDKit Fingerprint objects.
    Includes error handling for invalid SMILES.
    """
    fingerprints = []
    invalid_smiles = 0
    for smi in tqdm(smiles_set, desc="Generating Fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fingerprints.append(fpgen.GetFingerprint(mol))
        else:
            invalid_smiles += 1
            
    if invalid_smiles > 0:
        print(f"Warning: Skipped {invalid_smiles} invalid SMILES strings.")
    
    return fingerprints

def main(args):
    """
    Main function to run the Nearest-Neighbor Tanimoto Similarity (NNTS) analysis.
    """
    print("--- Starting Nearest-Neighbor Tanimoto Similarity (NNTS) Analysis ---")
    
    # --- 1. Load Main Spectra Lookup File ---
    print(f"Loading spectra lookup: {args.spectra_lookup_path}")
    try:
        spectra_df = pd.read_feather(args.spectra_lookup_path)
    except Exception as e:
        print(f"Fatal Error: Could not load spectra lookup file. Error: {e}")
        return

    # --- 2. Get Unique SMILES and Fingerprints for Train/Test Sets ---
    
    # Set up the fingerprint generator (Morgan FP, radius 2, 2048 bits) [2, 3]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    print("\n--- Processing Training Set ---")
    train_smiles_set = get_unique_smiles([args.train_pairs_path], spectra_df)
    train_fps = generate_fingerprints(train_smiles_set, fpgen)
    
    print("\n--- Processing Test Set ---")
    test_smiles_set = get_unique_smiles([args.test_pairs_path], spectra_df)
    test_fps = generate_fingerprints(test_smiles_set, fpgen)

    if not train_fps or not test_fps:
        print("Fatal Error: Could not generate fingerprints for train or test set.")
        return

    print(f"\nComparing {len(test_fps)} test molecules against {len(train_fps)} train molecules.")

    # --- 3. Calculate NNTS Distribution ---
    print("Calculating NNTS distribution (this may take a moment)...")
    nnts_scores = []
    
    # Loop over each test set fingerprint
    for test_fp in tqdm(test_fps, desc="Calculating NNTS"):
        # Compare one test FP against ALL training FPs efficiently [4, 5]
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        
        # Find the single highest similarity score (the "nearest neighbor")
        if similarities:
            nnts_scores.append(max(similarities))
        else:
            nnts_scores.append(0.0)

    # --- 4. Generate and Save Report ---
    print("\n--- NNTS Analysis Report ---")
    
    mean_nnts = np.mean(nnts_scores)
    median_nnts = np.median(nnts_scores)
    p90_nnts = np.percentile(nnts_scores, 90)
    p95_nnts = np.percentile(nnts_scores, 95)
    p99_nnts = np.percentile(nnts_scores, 99)
    
    # Find the percentage of the test set that is "highly similar" (Tanimoto > 0.85)
    # to *any* molecule in the training set.
    high_similarity_count = sum(1 for s in nnts_scores if s > 0.85)
    high_similarity_percent = (high_similarity_count / len(nnts_scores)) * 100

    report = (
        f"Total Test Molecules Analyzed: {len(nnts_scores)}\n"
        f"Total Train Molecules Reference: {len(train_fps)}\n"
        "--------------------------------------------------\n"
        f"Mean NNTS:   {mean_nnts:.4f}\n"
        f"Median NNTS: {median_nnts:.4f} (50th Percentile)\n"
        f"90th Percentile: {p90_nnts:.4f}\n"
        f"95th Percentile: {p95_nnts:.4f}\n"
        f"99th Percentile: {p99_nnts:.4f}\n"
        "--------------------------------------------------\n"
        f"Test molecules with high similarity (Tanimoto > 0.85) to training set:\n"
        f"  Count: {high_similarity_count}\n"
        f"  Percent: {high_similarity_percent:.2f}%\n"
    )
    
    print(report)
    
    # Save the text report
    try:
        os.makedirs(os.path.dirname(args.output_report_path), exist_ok=True)
        with open(args.output_report_path, 'w') as f:
            f.write(report)
        print(f"Successfully saved report to: {args.output_report_path}")
    except Exception as e:
        print(f"Warning: Could not save text report. Error: {e}")

    # --- 5. Generate and Save Plot ---
    try:
        plt.figure(figsize=(12, 7))
        plt.hist(nnts_scores, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
        plt.title(f'Nearest-Neighbor Tanimoto Similarity (NNTS) Distribution\nTest Set ({len(test_fps)} mols) vs. Train Set ({len(train_fps)} mols)', fontsize=14)
        plt.xlabel('Tanimoto Similarity to Closest Molecule in Training Set', fontsize=12)
        plt.ylabel('Count of Test Set Molecules', fontsize=12)
        plt.axvline(median_nnts, color='red', linestyle='--', linewidth=2, label=f'Median = {median_nnts:.3f}')
        plt.axvline(p95_nnts, color='orange', linestyle=':', linewidth=2, label=f'95th Percentile = {p95_nnts:.3f}')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        os.makedirs(os.path.dirname(args.output_plot_path), exist_ok=True)
        plt.savefig(args.output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved plot to: {args.output_plot_path}")
        
    except Exception as e:
        print(f"Warning: Could not save plot. Error: {e}")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Nearest-Neighbor Tanimoto Similarity (NNTS) analysis on train/test splits.")
    
    parser.add_argument("--train_pairs_path", type=str, required=True,
                        help="Path to the *training* pairs.feather file (e.g., balanced_111_dataset_train.feather).")
    parser.add_argument("--test_pairs_path", type=str, required=True,
                        help="Path to the *test* pairs.feather file (e.g., balanced_111_dataset_test.feather).")
    parser.add_argument("--spectra_lookup_path", type=str, required=True,
                        help="Path to the main spectra lookup file (e.g., augmented_msg_df.feather).")
    parser.add_argument("--output_report_path", type=str, required=True,
                        help="Path to save the output.txt report file.")
    parser.add_argument("--output_plot_path", type=str, required=True,
                        help="Path to save the output.png histogram plot.")

    args = parser.parse_args()
    main(args)