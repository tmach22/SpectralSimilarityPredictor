import pandas as pd
import argparse
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

def get_unique_smiles(pairs_df_paths, spectra_df):
    """
    Loads one or more pair files, gets all unique spectrum IDs,
    and maps them to a set of unique, valid SMILES strings.
    
    (This function is from the previous NNTS script)
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

def get_generic_murcko_scaffold(smiles):
    """
    Calculates the Generic Murcko Scaffold for a given SMILES string.
    Returns None if the molecule is invalid or has no scaffold (linear).
    """
    if not smiles or pd.isna(smiles):
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    try:
        # 1. Get the standard Murcko scaffold (retains atom types/bonds) [2]
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        
        # 2. Check if it's a linear molecule (no scaffold) [3]
        if scaffold_mol.GetNumAtoms() == 0:
            return None
            
        # 3. Convert to a generic scaffold (all atoms -> C, all bonds -> single) 
        generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        
        # 4. Return the canonical SMILES string for the generic scaffold
        return Chem.MolToSmiles(generic_mol)
    except Exception as e:
        # Catch any other RDKit errors
        print(f"Warning: Scaffold calculation failed for SMILES: {smiles}. Error: {e}")
        return None

def main(args):
    """
    Main function to run the Generic Murcko Scaffold overlap analysis.
    """
    print("--- Starting Generic Murcko Scaffold Overlap Analysis ---")
    
    # --- 1. Load Main Spectra Lookup File ---
    print(f"Loading spectra lookup: {args.spectra_lookup_path}")
    try:
        spectra_df = pd.read_feather(args.spectra_lookup_path)
    except Exception as e:
        print(f"Fatal Error: Could not load spectra lookup file. Error: {e}")
        return

    # --- 2. Get Unique SMILES for Train/Test Sets ---
    print("\n--- Processing Training Set ---")
    train_smiles_set = get_unique_smiles([args.train_pairs_path], spectra_df)
    
    print("\n--- Processing Test Set ---")
    test_smiles_set = get_unique_smiles([args.test_pairs_path], spectra_df)

    # --- 3. Calculate Unique Generic Scaffolds for Each Set ---
    
    print("\nCalculating generic scaffolds for TRAINING set...")
    train_scaffolds = set()
    for smi in tqdm(train_smiles_set, desc="Train Scaffolds"):
        scaffold = get_generic_murcko_scaffold(smi)
        if scaffold:
            train_scaffolds.add(scaffold)
            
    print("\nCalculating generic scaffolds for TEST set...")
    test_scaffolds = set()
    for smi in tqdm(test_smiles_set, desc="Test Scaffolds"):
        scaffold = get_generic_murcko_scaffold(smi)
        if scaffold:
            test_scaffolds.add(scaffold)

    # --- 4. Calculate Overlap and Generate Report ---
    print("\n--- Generic Murcko Scaffold Overlap Report ---")

    num_train_scaffolds = len(train_scaffolds)
    num_test_scaffolds = len(test_scaffolds)
    
    # Find the intersection
    overlap_set = train_scaffolds.intersection(test_scaffolds)
    num_overlap = len(overlap_set)
    
    if num_test_scaffolds > 0:
        percent_leakage = (num_overlap / num_test_scaffolds) * 100
    else:
        percent_leakage = 0.0

    report = (
        f"Unique Generic Scaffolds in Training Set: {num_train_scaffolds}\n"
        f"Unique Generic Scaffolds in Test Set:   {num_test_scaffolds}\n"
        "--------------------------------------------------\n"
        f"Scaffolds in BOTH Train and Test Sets:   {num_overlap}\n"
        f"Overlap as % of Test Set Scaffolds:       {percent_leakage:.4f}%\n"
    )
    
    if num_overlap == 0:
        report += "\nResult: PASSED. No scaffold leakage was detected."
    else:
        report += f"\nResult: FAILED. {num_overlap} overlapping scaffolds detected."

    print(report)
    
    # Save the text report
    try:
        os.makedirs(os.path.dirname(args.output_report_path), exist_ok=True)
        with open(args.output_report_path, 'w') as f:
            f.write(report)
        print(f"\nSuccessfully saved report to: {args.output_report_path}")
    except Exception as e:
        print(f"Warning: Could not save text report. Error: {e}")
        
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generic Murcko Scaffold overlap analysis on train/test splits.")
    
    parser.add_argument("--train_pairs_path", type=str, required=True,
                        help="Path to the *training* pairs.feather file (e.g., balanced_111_dataset_train.feather).")
    parser.add_argument("--test_pairs_path", type=str, required=True,
                        help="Path to the *test* pairs.feather file (e.g., balanced_111_dataset_test.feather).")
    parser.add_argument("--spectra_lookup_path", type=str, required=True,
                        help="Path to the main spectra lookup file (e.g., augmented_msg_df.feather).")
    parser.add_argument("--output_report_path", type=str, required=True,
                        help="Path to save the output.txt report file.")

    args = parser.parse_args()
    main(args)