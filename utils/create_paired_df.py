import pandas as pd
import argparse
import time

def create_paired_metadata(pairs_path, lookup_path, output_path):
    """
    Merges a pairs file with an augmented metadata lookup file twice 
    to create a single DataFrame with metadata for both spectra in each pair.

    Args:
        pairs_path (str): Path to the train_pairs.feather file.
        lookup_path (str): Path to the augmented_msg_df.feather file.
        output_path (str): Path to save the final merged_pairs.feather file.
    """
    
    print("--- 1. Loading Data ---")
    try:
        start_time = time.time()
        pairs_df = pd.read_feather(pairs_path)
        print(f"Successfully loaded {len(pairs_df):,} records from {pairs_path} in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error loading {pairs_path}: {e}")
        return

    try:
        start_time = time.time()
        lookup_df = pd.read_feather(lookup_path)
        print(f"Successfully loaded {len(lookup_df):,} records from {lookup_path} in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error loading {lookup_path}: {e}")
        return

    print("\n--- 2. Preparing Lookup Data ---")
    
    # To save memory, we select only the columns we need for our taxonomy
    # and rename the 'instrument' column to avoid conflicts with your first file
    # (Though your first file didn't have 'instrument', this is good practice)
    columns_to_keep = [
        'spectrum_id',
        'smiles',
        'adduct',
        'instrument',
        'morgan_fingerprint',
        'inchikey',
        'formula',
        'precursor_mz',
        'parent_mass',
        'collision_energy',
        'fold'
    ]
    
    # Filter out any columns that might not be in the lookup file
    available_cols = [col for col in columns_to_keep if col in lookup_df.columns]
    
    if len(available_cols) < len(columns_to_keep):
        missing = set(columns_to_keep) - set(available_cols)
        print(f"Warning: Missing expected columns from lookup: {missing}")
        
    slim_lookup_df = lookup_df[available_cols]
    print(f"Created slim lookup table with {len(slim_lookup_df):,} records and {len(available_cols)} columns.")

    print(f"\nColumns: {slim_lookup_df.info(verbose=True)}")

    print("\n--- 3. Performing Merges ---")
    
    # --- First Merge (for name_main) ---
    start_time = time.time()
    print(f"Merging {len(pairs_df):,} pairs with lookup on 'name_main'...")
    merged_df = pd.merge(
        left=pairs_df,
        right=slim_lookup_df,
        left_on='name_main',
        right_on='spectrum_id',
        how='left'
    )
    print(f"First merge complete in {time.time() - start_time:.2f}s. New shape: {merged_df.shape}")

    # --- Second Merge (for name_sub) ---
    start_time = time.time()
    print(f"Merging {len(merged_df):,} pairs with lookup on 'name_sub'...")
    # We use suffixes to automatically distinguish the columns from the two merges
    final_merged_df = pd.merge(
        left=merged_df,
        right=slim_lookup_df,
        left_on='name_sub',
        right_on='spectrum_id',
        how='left',
        suffixes=('_main', '_sub') 
    )
    print(f"Second merge complete in {time.time() - start_time:.2f}s. Final shape: {final_merged_df.shape}")

    print("\n--- 4. Cleaning and Saving ---")
    
    # Drop the redundant spectrum_id columns from the merges
    if 'spectrum_id_main' in final_merged_df.columns:
        final_merged_df.drop(columns=['spectrum_id_main'], inplace=True)
    if 'spectrum_id_sub' in final_merged_df.columns:
        final_merged_df.drop(columns=['spectrum_id_sub'], inplace=True)

    # Display the structure of the new, final DataFrame
    print("\nFinal Paired DataFrame Info:")
    final_merged_df.info()

    print("\nFirst 5 records of the final paired DataFrame:")
    print(final_merged_df.head())

    # Save the new DataFrame to a feather file
    try:
        start_time = time.time()
        final_merged_df.to_feather(output_path)
        print(f"\nSuccessfully saved final paired data to: {output_path} in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"\nError saving file to {output_path}: {e}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge spectrum pairs with metadata from the augmented lookup file.")
    parser.add_argument("--pairs_path", type=str, required=True, 
                        help="Path to your train_pairs.feather (or similar) file.")
    parser.add_argument("--lookup_path", type=str, required=True, 
                        help="Path to your augmented_msg_df.feather file.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the new merged_pairs.feather file.")
    
    args = parser.parse_args()
    
    print("--- Starting Paired Metadata Creation Script ---")
    create_paired_metadata(args.pairs_path, args.lookup_path, args.output_path)
    print("--- Script Finished ---")