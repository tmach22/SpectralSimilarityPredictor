import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def smiles_to_morgan_fp(smiles_string: str, radius: int = 2, n_bits: int = 2048):
    """
    Converts a SMILES string to a Morgan fingerprint.

    Args:
        smiles_string (str): The SMILES representation of the molecule.
        radius (int): The radius of the Morgan fingerprint.
        n_bits (int): The size of the bit vector.

    Returns:
        np.ndarray or None: A NumPy array of the fingerprint, or None if the SMILES is invalid.
    """
    if not isinstance(smiles_string, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None
        # RDKit's GetMorganFingerprintAsBitVect returns a bit vector object
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        # Convert the bit vector to a NumPy array for easier storage and use
        return np.array(fp, dtype=np.uint8)
    except Exception as e:
        print(f"Warning: Could not process SMILES '{smiles_string}'. Error: {e}")
        return None

def main():
    """
    Main function to read a feather file, generate Morgan fingerprints,
    and save the updated DataFrame to a new feather file.
    """
    parser = argparse.ArgumentParser(description="Generate Morgan fingerprints for molecules in a feather file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input feather file (e.g., msg_df.feather).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output feather file.")
    parser.add_argument("--smiles_column", type=str, default="smiles", help="Name of the column containing SMILES strings.")
    parser.add_argument("--radius", type=int, default=2, help="Radius for the Morgan fingerprint.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Number of bits for the Morgan fingerprint.")
    args = parser.parse_args()

    print(f"Input Path: {args}")

    print(f"--- 1. Loading data from {args.input_path} ---")
    try:
        df = pd.read_feather(args.input_path)
        print(f"Successfully loaded {len(df)} records.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if args.smiles_column not in df.columns:
        print(f"Error: SMILES column '{args.smiles_column}' not found in the DataFrame.")
        return

    print(f"\n--- 2. Generating Morgan fingerprints (radius={args.radius}, nBits={args.n_bits}) ---")
    # Initialize tqdm for pandas apply
    tqdm.pandas(desc="Calculating Fingerprints")
    
    df['morgan_fingerprint'] = df[args.smiles_column].progress_apply(
        lambda x: smiles_to_morgan_fp(x, radius=args.radius, n_bits=args.n_bits)
    )

    # Report on any molecules that failed
    failed_count = df['morgan_fingerprint'].isna().sum()
    if failed_count > 0:
        print(f"\nWarning: Failed to generate fingerprints for {failed_count} molecules (invalid SMILES). These will be stored as null.")

    print(f"\n--- 3. Saving updated DataFrame to {args.output_path} ---")
    try:
        df.to_feather(args.output_path)
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")

    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()