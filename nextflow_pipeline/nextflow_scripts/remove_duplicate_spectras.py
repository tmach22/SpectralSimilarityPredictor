import pandas as pd
import os
import argparse
from rdkit import Chem

# --- Helper Functions ---
def generate_inchikey(smiles):
    if not smiles or pd.isna(smiles): return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol) if mol else None
    except: return None

def create_dedup_hash(row, precursor_col="precursor_mz", adduct_col="adduct"):
    inchikey = row.get("_temp_inchikey", "UNKNOWN")
    try:
        mz = round(float(row[precursor_col]), 2)
    except:
        mz = "0.00"
    hash_key = f"{inchikey}_{mz}"
    if adduct_col and adduct_col in row:
        hash_key += f"_{row[adduct_col]}"
    return hash_key

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_files", nargs='+', required=True, help="List of new feather files")
    parser.add_argument("--existing_dataset", required=True, help="Path to augmented_msg_df")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    args = parser.parse_args()

    # 1. Load Existing Dataset & Build Hash Set
    print(f"Loading reference dataset: {args.existing_dataset}")
    existing_df = pd.read_feather(args.existing_dataset)
    
    # Generate hashes for existing data
    existing_df["_temp_inchikey"] = existing_df["smiles"].apply(generate_inchikey)
    existing_hashes = set(existing_df.apply(create_dedup_hash, axis=1))
    print(f"Loaded {len(existing_hashes)} unique reference spectra.")
    del existing_df

    # 2. Process New Files
    os.makedirs(args.output_dir, exist_ok=True)
    
    for file_path in args.new_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        new_df = pd.read_feather(file_path)
        
        # Calculate hashes
        new_df["_temp_inchikey"] = new_df["smiles"].apply(generate_inchikey)
        new_df["_temp_hash"] = new_df.apply(create_dedup_hash, axis=1)
        
        # Filter
        cleaned_df = new_df[~new_df["_temp_hash"].isin(existing_hashes)].copy()
        cleaned_df.drop(columns=["_temp_inchikey", "_temp_hash"], inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True) # FIX: Reset index for Feather
        
        # Save
        output_path = os.path.join(args.output_dir, f"cleaned_{filename}")
        cleaned_df.to_feather(output_path)
        print(f"Saved {output_path} (Records: {len(new_df)} -> {len(cleaned_df)})")