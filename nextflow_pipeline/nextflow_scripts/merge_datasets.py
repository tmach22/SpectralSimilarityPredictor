import pandas as pd
import argparse
import os
import numpy as np

def sanitize_columns(df):
    """
    Feather does not support numpy arrays in object columns.
    This function detects array-like columns (fingerprints/embeddings) 
    and serializes them to bytes to match the existing dataset schema.
    """
    # List of columns that often contain numpy arrays in MassSpec/Chemoinformatics
    complex_cols = ['morgan_fingerprint', 'embeddings', 'spectrum', 'intensity']
    
    for col in complex_cols:
        if col in df.columns:
            # Check if the column actually contains numpy arrays
            # We check the first non-null value to guess the type
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            
            if isinstance(sample, np.ndarray):
                print(f"  - Sanitizing column '{col}': Converting NumPy arrays to bytes...")
                # Convert all numpy arrays in this column to bytes
                df[col] = df[col].apply(lambda x: x.tobytes() if isinstance(x, np.ndarray) else x)
            
            elif isinstance(sample, list):
                 # If they are lists, ensure they are distinct lists, not numpy-wrapped
                 pass 

    return df

def merge_feather_files(input_files, output_file):
    dfs = []
    
    print(f"Starting merge of {len(input_files)} files...")
    
    for file_path in input_files:
        try:
            print(f"  - Loading {os.path.basename(file_path)}...")
            df = pd.read_feather(file_path)
            
            # Sanitize immediately after loading to ensure compatibility during concat
            df = sanitize_columns(df)
            
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return

    if not dfs:
        print("No dataframes to merge!")
        return

    print("  - Concatenating dataframes...")
    # sort=False prevents column reordering issues
    merged_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    
    # Final safety check on the merged dataframe
    merged_df = sanitize_columns(merged_df)

    # Reset index for Feather compatibility
    merged_df.reset_index(drop=True, inplace=True)
    
    print(f"  - Final dataset shape: {merged_df.shape}")
    
    # Save output
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        merged_df.to_feather(output_file)
        print(f"Successfully saved merged dataset to {output_file}")
    except Exception as e:
        print(f"Error saving feather file: {e}")
        print("Tip: If this fails, try dropping the 'morgan_fingerprint' column if not strictly needed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple feather files into one.")
    
    parser.add_argument("--input_files", nargs='+', required=True, help="List of feather files to merge")
    parser.add_argument("--output_file", required=True, help="Path for the output merged feather file")
    
    args = parser.parse_args()
    
    merge_feather_files(args.input_files, args.output_file)