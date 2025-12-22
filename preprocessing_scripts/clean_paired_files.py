import pandas as pd
import argparse
import os
from tqdm import tqdm

def filter_pairs(args):
    print(f"--- Filtering Pairs against Clean Master File ---")
    
    # 1. Load the Clean Master File to get valid IDs
    print(f"Loading Clean Master File: {args.clean_spec_path}")
    spec_df = pd.read_feather(args.clean_spec_path)
    valid_spec_ids = set(spec_df['spectrum_id']) # Adjust column name if needed (e.g. 'spec_id')
    # Check if column is 'spectrum_id' or 'spec_id'
    if 'spec_id' in spec_df.columns:
        valid_spec_ids = set(spec_df['spec_id'])
    elif 'spectrum_id' in spec_df.columns:
        valid_spec_ids = set(spec_df['spectrum_id'])
        
    print(f"Valid Unique Spectra: {len(valid_spec_ids):,}")
    
    # 2. Process Each Pair File
    for pair_file in [args.train_path, args.val_path, args.test_path]:
        if not os.path.exists(pair_file):
            print(f"Skipping {pair_file} (Not found)")
            continue
            
        print(f"\nProcessing {os.path.basename(pair_file)}...")
        df = pd.read_feather(pair_file)
        initial_count = len(df)
        
        # Filter: Keep row only if BOTH spectra exist in the clean master file
        mask = df['name_main'].isin(valid_spec_ids) & df['name_sub'].isin(valid_spec_ids)
        df_clean = df[mask].reset_index(drop=True)
        
        final_count = len(df_clean)
        print(f"  Kept: {final_count:,} / {initial_count:,} ({final_count/initial_count*100:.1f}%)")
        
        # Save
        base, ext = os.path.splitext(pair_file)
        output_path = f"{base}_STRICT{ext}"
        df_clean.to_feather(output_path)
        print(f"  Saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_spec_path", type=str, required=True, help="Path to augmented_msg_df_STRICT.feather")
    parser.add_argument("--train_path", type=str, required=True, help="Path to original train feather")
    parser.add_argument("--val_path", type=str, required=True, help="Path to original val feather")
    parser.add_argument("--test_path", type=str, required=True, help="Path to original test feather")
    
    args = parser.parse_args()
    filter_pairs(args)