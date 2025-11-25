import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import os
import argparse # Added to accept num_cores
import multiprocessing as mp
import numpy as np
from functools import partial

# --- Configuration ---
# Moved to argparse for flexibility

# --- Worker Function for Step 2: Scaffold Generation ---
def generate_scaffold_worker(smiles):
    """
    Worker function to calculate Murcko scaffold for a single SMILES.
    Returns a tuple (scaffold_smiles, original_smiles).
    """
    if not smiles or pd.isna(smiles):
        return ("", None) # Group with no scaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)

            # Check if it's a linear molecule (no scaffold)
            if scaffold.GetNumAtoms() == 0:
                return ("", smiles)
            
            # 3. Convert to a generic scaffold (all atoms -> C, all bonds -> single) 
            generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold)

            # 4. Get the canonical SMILES string for the generic scaffold
            scaffold_smi = Chem.MolToSmiles(generic_mol)
            return (scaffold_smi, smiles)
        except ValueError:
            # Handle cases where scaffold generation fails
            return ("", smiles) # Group molecules without scaffolds
    return ("", smiles) # Group unparseable SMILES

# --- Worker Function for Step 5: Pair Assignment ---
def assign_pairs_worker(df_chunk, id_to_smiles_map, train_smiles_set, val_smiles_set, test_smiles_set):
    """
    Worker function to assign pairs from a DataFrame chunk to splits.
    Uses .itertuples() for speed.
    """
    train_indices, val_indices, test_indices = [], [], []
    
    # itertuples is much faster than iterrows
    for row in df_chunk.itertuples():
        # Access by attribute (e.g., row.name_main) or index (row[1])
        # Make sure 'name_main' and 'name_sub' are in the chunk
        smiles_main = id_to_smiles_map.get(row.name_main)
        smiles_sub = id_to_smiles_map.get(row.name_sub)

        if smiles_main and smiles_sub:
            # Assign to a split only if BOTH molecules belong to that split
            if smiles_main in train_smiles_set and smiles_sub in train_smiles_set:
                train_indices.append(row.Index) # row.Index is the original index
            elif smiles_main in val_smiles_set and smiles_sub in val_smiles_set:
                val_indices.append(row.Index)
            elif smiles_main in test_smiles_set and smiles_sub in test_smiles_set:
                test_indices.append(row.Index)
                
    return (train_indices, val_indices, test_indices)

def main(args):
    # --- Main Script ---
    print("--- Starting Rigorous Scaffold-Based Data Splitting (Parallel) ---")
    print(f"Using {args.num_cores} CPU cores.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Load and Merge Initial Data ---
    print("Loading and merging data files...")
    try:
        spectra_df = pd.read_feather(args.spectra_file)
        pairs_df = pd.read_feather(args.pairs_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Details: {e}")
        exit()

    # Create a mapping from spectrum_id to smiles for efficient lookup
    id_to_smiles = spectra_df.set_index('spectrum_id')['smiles'].to_dict()
    
    # --- Step 2: Get All Unique Molecules and Calculate Scaffolds (Parallel) ---
    print("Extracting unique molecules and calculating Murcko scaffolds (in parallel)...")
    unique_smiles = spectra_df['smiles'].dropna().unique()
    
    scaffold_to_smiles = defaultdict(list)
    
    # Create a pool of workers
    with mp.Pool(processes=args.num_cores) as pool:
        # Use imap_unordered for efficiency, wrap with tqdm for progress
        results = list(tqdm(
            pool.imap_unordered(generate_scaffold_worker, unique_smiles, chunksize=1000),
            total=len(unique_smiles),
            desc="Generating Scaffolds"
        ))
    
    # Aggregate results from parallel workers
    for scaffold, smiles in results:
        if smiles: # Only add if SMILES was valid
            scaffold_to_smiles[scaffold].append(smiles)

    unique_scaffolds = list(scaffold_to_smiles.keys())
    print(f"Found {len(unique_smiles)} unique molecules with {len(unique_scaffolds)} unique scaffolds.")

    # --- Step 3: Split the Scaffolds into Train, Validation, and Test Sets ---
    print("Splitting scaffolds into train, validation, and test sets...")
    train_scaffolds, temp_scaffolds = train_test_split(
        unique_scaffolds, 
        test_size=(args.validation_size + args.test_size), 
        random_state=42
    )
    # Calculate correct relative test size
    relative_test_size = args.test_size / (args.validation_size + args.test_size)
    validation_scaffolds, test_scaffolds = train_test_split(
        temp_scaffolds,
        test_size=relative_test_size,
        random_state=42
    )

    print(f"Scaffold splits: {len(train_scaffolds)} train, {len(validation_scaffolds)} validation, {len(test_scaffolds)} test.")

    # --- Step 4: Assign All Molecules to Their Respective Splits ---
    print("Assigning molecules to splits based on their scaffolds...")
    # Using sets for fast 'in' checks
    train_smiles = set([s for scaffold in train_scaffolds for s in scaffold_to_smiles[scaffold]])
    validation_smiles = set([s for scaffold in validation_scaffolds for s in scaffold_to_smiles[scaffold]])
    test_smiles = set([s for scaffold in test_scaffolds for s in scaffold_to_smiles[scaffold]])

    # --- Step 5: Assign Pairs to Final Datasets (Parallel) ---
    print(f"Assigning {len(pairs_df):,} pairs to the final datasets (in parallel)...")
    
    # Split the massive pairs_df into chunks for the workers
    df_chunks = np.array_split(pairs_df, args.num_cores * 2) # Split into more chunks than cores
    
    # Create a partial function to "freeze" the large arguments
    # This avoids pickling them for every single task
    p_assign_worker = partial(
        assign_pairs_worker,
        id_to_smiles_map=id_to_smiles,
        train_smiles_set=train_smiles,
        val_smiles_set=validation_smiles,
        test_smiles_set=test_smiles
    )

    all_train_indices, all_val_indices, all_test_indices = [], [], []

    # Re-create the pool
    with mp.Pool(processes=args.num_cores) as pool:
        # Use imap_unordered, wrap with tqdm
        results = list(tqdm(
            pool.imap_unordered(p_assign_worker, df_chunks),
            total=len(df_chunks),
            desc="Assigning Pairs"
        ))

    # Aggregate results from parallel workers
    print("Aggregating parallel results...")
    for train_idx_chunk, val_idx_chunk, test_idx_chunk in results:
        all_train_indices.extend(train_idx_chunk)
        all_val_indices.extend(val_idx_chunk)
        all_test_indices.extend(test_idx_chunk)

    # --- Step 6: Create and Save the Final DataFrames ---
    print("Creating and saving the final data splits...")
    # Use .loc to select rows by their original index
    train_pairs_df = pairs_df.loc[all_train_indices].reset_index(drop=True)
    validation_pairs_df = pairs_df.loc[all_val_indices].reset_index(drop=True)
    test_pairs_df = pairs_df.loc[all_test_indices].reset_index(drop=True)

    # Define output paths
    train_path = os.path.join(args.output_dir, os.path.basename(args.pairs_file).replace('.feather', '_generic_train.feather'))
    val_path = os.path.join(args.output_dir, os.path.basename(args.pairs_file).replace('.feather', '_generic_val.feather'))
    test_path = os.path.join(args.output_dir, os.path.basename(args.pairs_file).replace('.feather', '_generic_test.feather'))

    train_pairs_df.to_feather(train_path)
    validation_pairs_df.to_feather(val_path)
    test_pairs_df.to_feather(test_path)

    print("\n--- Data Splitting Complete ---")
    print(f"Training pairs: {len(train_pairs_df)}")
    print(f"Validation pairs: {len(validation_pairs_df)}")
    print(f"Test pairs: {len(test_pairs_df)}")
    print(f"Data splits saved to: {args.output_dir}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    # Set start method for multiprocessing
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            start_method = 'fork' if os.name == 'posix' else 'spawn'
            mp.set_start_method(start_method)
            print(f"Set multiprocessing start method to '{start_method}'.")
        else:
            print(f"Multiprocessing start method already set to '{current_method}'.")
    except RuntimeError as e:
        print(f"Could not set multiprocessing start method (may already be set): {e}")

    parser = argparse.ArgumentParser(description="Parallel Murcko Scaffold data splitting.")
    
    # 1. Input file paths
    parser.add_argument("--spectra_file", type=str, required=True,
                        help="Path to the spectra file (e.g., msg_df_fingerprint.feather)")
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to the pairs file (e.g., brute_force_combined.feather)")
    
    # 2. Output directory
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the final data splits.")
    
    # 3. Split ratios
    parser.add_argument("--validation_size", type=float, default=0.1, help="Fraction for validation (e.g., 0.1)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction for testing (e.g., 0.1)")
    
    # 4. Parallelization
    parser.add_argument("--num_cores", type=int, default=mp.cpu_count(),
                        help="Number of CPU cores to use. Defaults to all available.")

    args = parser.parse_args()
    
    if (args.validation_size + args.test_size) >= 1.0:
        print("Error: validation_size and test_size must sum to less than 1.0")
    else:
        main(args)