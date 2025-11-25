import pandas as pd
import argparse
import sys
from collections import Counter
import numpy as np

class Logger:
    """Simple logger class to print to console and write to a file."""
    def __init__(self, filepath=None):
        self.file_handle = None
        if filepath:
            try:
                self.file_handle = open(filepath, 'w')
            except Exception as e:
                print(f"Warning: Could not open log file {filepath}. Error: {e}")
                print("Logging to console only.")
    
    def log(self, message):
        """Prints to console and writes to log file if available."""
        print(message)
        if self.file_handle:
            self.file_handle.write(message + '\n')
            
    def close(self):
        """Closes the log file."""
        if self.file_handle:
            self.file_handle.close()

# --- NEW: Helper function to get the mode ---
def get_mode(series):
    """
    Calculates the mode of a pandas Series.
    Returns the first mode if multiple modes exist, or NaN if empty.
    """
    modes = series.mode()
    if modes.empty:
        return np.nan
    return modes.iloc[0]

def analyze_duplicate_variations(pairs_file, lookup_file, logger, csv_filepath):
    """
    Analyzes a pairs file to find duplicate molecule pairs (by InChIKey)
    and reports on variations in their metadata and cosine similarity.
    """
    
    METADATA_COLS = ['instrument', 'adduct', 'collision_energy']
    
    logger.log(f"Loading pairs file: {pairs_file}")
    try:
        df_pairs = pd.read_feather(pairs_file)
        logger.log(f"Loaded {len(df_pairs)} pairs.")
    except Exception as e:
        logger.log(f"Error loading {pairs_file}: {e}")
        return
        
    logger.log(f"Loading lookup file: {lookup_file}")
    try:
        df_lookup = pd.read_feather(lookup_file)
        lookup_cols_to_keep = ['spectrum_id', 'inchikey'] + METADATA_COLS
        df_lookup_subset = df_lookup[lookup_cols_to_keep].drop_duplicates(subset=['spectrum_id'])
        logger.log(f"Loaded {len(df_lookup_subset)} unique lookup entries.")
    except Exception as e:
        logger.log(f"Error loading {lookup_file}: {e}")
        return

    logger.log("Files loaded successfully.")
    logger.log("-" * 30)

    # 1. Join metadata for 'name_main' (Molecule A)
    logger.log("Joining metadata for 'name_main'...")
    df_merged = pd.merge(
        df_pairs, 
        df_lookup_subset, 
        left_on='name_main', 
        right_on='spectrum_id',
        how='left'
    )
    rename_dict_A = {'inchikey': 'inchikey_A'}
    for col in METADATA_COLS:
        rename_dict_A[col] = f"{col}_A"
    df_merged = df_merged.rename(columns=rename_dict_A).drop(columns=['spectrum_id'])

    # 2. Join metadata for 'name_sub' (Molecule B)
    logger.log("Joining metadata for 'name_sub'...")
    df_merged = pd.merge(
        df_merged,
        df_lookup_subset,
        left_on='name_sub',
        right_on='spectrum_id',
        how='left'
    )
    rename_dict_B = {'inchikey': 'inchikey_B'}
    for col in METADATA_COLS:
        rename_dict_B[col] = f"{col}_B"
    df_merged = df_merged.rename(columns=rename_dict_B).drop(columns=['spectrum_id'])

    logger.log(f"Total pairs after merging: {len(df_merged)}")

    # 3. Handle missing InChIKeys
    initial_count = len(df_merged)
    df_merged = df_merged.dropna(subset=['inchikey_A', 'inchikey_B'])
    if len(df_merged) < initial_count:
        logger.log(f"Warning: Dropped {initial_count - len(df_merged)} pairs due to missing InChIKeys.")
    
    logger.log("-" * 30)
    logger.log("Canonicalizing pairs (this may take a moment)...")

    # 4. Canonicalize Pairs & Metadata
    rows_to_swap = df_merged['inchikey_A'] > df_merged['inchikey_B']
    cols_A = ['inchikey_A'] + [f"{col}_A" for col in METADATA_COLS]
    cols_B = ['inchikey_B'] + [f"{col}_B" for col in METADATA_COLS]
    
    df_merged.loc[rows_to_swap, cols_A + cols_B] = \
        df_merged.loc[rows_to_swap, cols_B + cols_A].values

    # 5. Create the canonical pair string
    df_merged['canonical_pair'] = df_merged['inchikey_A'] + '|' + df_merged['inchikey_B']
    logger.log("Canonicalization complete.")
    logger.log("-" * 30)

    # 6. Group by the canonical pair and analyze variations
    logger.log("Grouping by canonical pair and analyzing variations...")
    
    # --- UPDATE: Added 'get_mode' to all aggregations ---
    agg_dict = {
        'cosine_similarity': ['size', 'nunique', 'mean', 'std', get_mode] 
    }
    for col in METADATA_COLS:
        agg_dict[f"{col}_A"] = ['nunique', get_mode]
        agg_dict[f"{col}_B"] = ['nunique', get_mode]

    df_analysis = df_merged.groupby('canonical_pair').agg(agg_dict)
    
    # Flatten the multi-index columns
    df_analysis.columns = ['_'.join(col).strip() for col in df_analysis.columns.values]
    
    # --- UPDATE: Add renames for all new 'get_mode' columns ---
    rename_map = {'cosine_similarity_size': 'total_count'}
    rename_map['cosine_similarity_get_mode'] = 'cosine_similarity_mode'
    for col in METADATA_COLS:
        rename_map[f'{col}_A_get_mode'] = f'{col}_A_mode'
        rename_map[f'{col}_B_get_mode'] = f'{col}_B_mode'
    df_analysis = df_analysis.rename(columns=rename_map)
    
    # Filter for only the duplicate pairs
    df_duplicates = df_analysis[df_analysis['total_count'] > 1].reset_index()
    df_duplicates['cosine_similarity_std'] = df_duplicates['cosine_similarity_std'].fillna(0)
    
    logger.log(f"Found {len(df_duplicates)} unique molecule pairs that appear more than once.")
    
    if len(df_duplicates) == 0:
        logger.log("SUCCESS: No duplicate molecule pairs (by InChIKey) found.")
        logger.log("Analysis complete.")
        return

    # 7. Create summary columns for reporting
    df_duplicates['has_similarity_variation'] = df_duplicates['cosine_similarity_nunique'] > 1
    
    for col in METADATA_COLS:
        df_duplicates[f'has_{col}_variation'] = \
            (df_duplicates[f'{col}_A_nunique'] > 1) | (df_duplicates[f'{col}_B_nunique'] > 1)

    # 8. Log the summary report (No changes here, std is still the best summary for variation)
    logger.log("-" * 30)
    logger.log("--- Summary of Variations in Duplicate Pairs ---")
    
    total_dupes = len(df_duplicates)
    logger.log(f"Total duplicate pairs: {total_dupes}")
    
    sim_vars = df_duplicates['has_similarity_variation'].sum()
    logger.log(f" - Pairs with varying 'cosine_similarity': {sim_vars} / {total_dupes}")
    
    if sim_vars > 0:
        varying_pairs = df_duplicates[df_duplicates['has_similarity_variation']]
        avg_std = varying_pairs['cosine_similarity_std'].mean()
        max_std = varying_pairs['cosine_similarity_std'].max()
        logger.log(f"    - Avg standard deviation for varying pairs: {avg_std:.4f}")
        logger.log(f"    - Max standard deviation for varying pairs: {max_std:.4f}")
    
    for col in METADATA_COLS:
        var_count = df_duplicates[f'has_{col}_variation'].sum()
        logger.log(f" - Pairs with varying '{col}': {var_count} / {total_dupes}")
    
    logger.log("--------------------------------------------------")

    # 9. Save detailed analysis to CSV
    if csv_filepath:
        logger.log(f"Saving detailed variation analysis to {csv_filepath}...")
        try:
            # Sort by std, then count, descending
            df_duplicates = df_duplicates.sort_values(
                by=['cosine_similarity_std', 'total_count'], 
                ascending=False
            )
            
            # --- UPDATE: Added mode columns to the output list ---
            key_cols = [
                'canonical_pair', 'total_count', 
                'cosine_similarity_nunique', 'has_similarity_variation',
                'cosine_similarity_mean', 'cosine_similarity_std', 
                'cosine_similarity_mode'
            ]
            meta_cols = []
            for col in METADATA_COLS:
                meta_cols.extend([
                    f'{col}_A_nunique', f'{col}_A_mode',
                    f'{col}_B_nunique', f'{col}_B_mode', 
                    f'has_{col}_variation'
                ])
            
            df_duplicates = df_duplicates[key_cols + meta_cols]
            
            df_duplicates.to_csv(csv_filepath, index=False)
            logger.log(f"SUCCESS: Saved detailed analysis to {csv_filepath}")
        except Exception as e:
            logger.log(f"Error saving CSV to {csv_filepath}: {e}")

    logger.log("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze variations in metadata for duplicate molecule pairs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pairs_file", 
        type=str, 
        required=True, 
        help="Path to the feather file containing molecule pairs (e.g., balanced_10bin_dataset_train.feather)"
    )
    parser.add_argument(
        "--lookup_file", 
        type=str, 
        required=True, 
        help="Path to the lookup feather file containing spectrum_id and inchikey (e.g., augmented_msg_df.feather)"
    )
    parser.add_argument(
        "--output_prefix", 
        type=str, 
        default=None, 
        help="Optional prefix for output files (e.g., 'analysis_results').\n"
             "This will create 'analysis_results.txt' (log) and 'analysis_results.csv' (data)."
    )
    
    args = parser.parse_args()
    
    logger = None
    try:
        txt_filepath = f"{args.output_prefix}.txt" if args.output_prefix else None
        csv_filepath = f"{args.output_prefix}.csv" if args.output_prefix else None
        
        logger = Logger(txt_filepath)
        
        if args.output_prefix:
            logger.log(f"Logging console output to: {txt_filepath}")
        
        analyze_duplicate_variations(
            args.pairs_file, 
            args.lookup_file, 
            logger,
            csv_filepath
        )
        
    except Exception as e:
        if logger:
            logger.log(f"\n--- An unexpected error occurred ---")
            import traceback
            logger.log(traceback.format_exc())
        else:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            
    finally:
        if logger:
            logger.log("Script finished.")
            logger.close()

if __name__ == "__main__":
    main()