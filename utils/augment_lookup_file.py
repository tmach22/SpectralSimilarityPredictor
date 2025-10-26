import pandas as pd
import os
import argparse

def augment_lookup_file(msg_df_path, gym_df_path, output_path):
    """
    Augments the msg_df lookup file with additional metadata from the main
    Mass Spec Gym dataset.

    Args:
        msg_df_path (str): Path to the msg_df.feather file.
        gym_df_path (str): Path to the full Mass Spec Gym data feather file.
        output_path (str): Path to save the new augmented feather file.
    """
    print("--- 1. Loading DataFrames ---")
    
    # Load your current lookup table
    try:
        msg_df = pd.read_feather(msg_df_path)
        print(f"Successfully loaded {len(msg_df):,} records from {msg_df_path}")
    except Exception as e:
        print(f"Error loading {msg_df_path}: {e}")
        return

    # Load the main Mass Spec Gym dataset
    try:
        gym_df = pd.read_feather(gym_df_path)
        print(f"Successfully loaded {len(gym_df):,} records from {gym_df_path}")
    except Exception as e:
        print(f"Error loading {gym_df_path}: {e}")
        return

    print("\n--- 2. Preparing Data for Merge ---")
    
    # Define the columns we want to add from the main dataset.
    # We exclude columns that are already in msg_df or are redundant (like mzs/intensities).
    columns_to_add = [
        'identifier',
        'inchikey',
        'formula',
        'precursor_formula',
        'parent_mass',
        'instrument_type',
        'collision_energy',
        'fold',
        'simulation_challenge'
    ]
    
    # Ensure the selected columns exist in the dataframe
    missing_cols = [col for col in columns_to_add if col not in gym_df.columns]
    if missing_cols:
        print(f"Warning: The following columns were not found in {gym_df_path} and will be skipped: {missing_cols}")
        columns_to_add = [col for col in columns_to_add if col in gym_df.columns]

    gym_subset = gym_df[columns_to_add]
    print(f"Selected {len(columns_to_add)} columns to merge from the main dataset.")

    print("\n--- 3. Merging DataFrames ---")
    
    # Perform a left merge to keep all records from your original msg_df
    # and add information from the main gym_df where spectrum IDs match.
    augmented_df = pd.merge(
        left=msg_df,
        right=gym_subset,
        left_on='spectrum_id',
        right_on='identifier',
        how='left'
    )

    # Drop the redundant 'identifier' column from the merge
    if 'identifier' in augmented_df.columns:
        augmented_df.drop(columns=['identifier'], inplace=True)

    print(f"Merge complete. The new DataFrame has {len(augmented_df):,} records.")

    print("\n--- 4. Inspecting and Saving the Result ---")
    
    # Display the structure of the new, augmented DataFrame
    print("\nDataFrame Info:")
    augmented_df.info()

    print("\nFirst 5 records of the augmented DataFrame:")
    print(augmented_df.head())

    # Save the new DataFrame to a feather file
    try:
        augmented_df.to_feather(output_path)
        print(f"\nSuccessfully saved augmented data to: {output_path}")
    except Exception as e:
        print(f"\nError saving file to {output_path}: {e}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augment the msg_df lookup file with more metadata.")
    parser.add_argument("--msg_df_path", type=str, required=True, help="Path to your msg_df.feather file.")
    parser.add_argument("--gym_df_path", type=str, required=True, help="Path to the full Mass Spec Gym data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the new augmented_msg_df.feather file.")
    
    args = parser.parse_args()
    
    augment_lookup_file(args.msg_df_path, args.gym_df_path, args.output_path)