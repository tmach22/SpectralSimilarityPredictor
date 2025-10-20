import argparse
import os
import pandas as pd
from tqdm import tqdm

def create_contrastive_set(pairs_df, spec_df, positive_threshold, negative_threshold):
    """
    Merges spectrum pairs with molecule IDs and filters for positive/negative pairs.

    Args:
        pairs_df (pd.DataFrame): DataFrame with spectrum_id_a, spectrum_id_b, and similarity.
        spec_df (pd.DataFrame): DataFrame mapping spec_id to mol_id.
        positive_threshold (float): Similarity score above which pairs are considered positive.
        negative_threshold (float): Similarity score below which pairs are considered negative.

    Returns:
        pd.DataFrame: A new DataFrame with mol_id_a, mol_id_b, and a binary label.
    """
    # Create a mapping from spec_id to mol_id for efficient lookup
    spec_to_mol = spec_df.set_index('spec_id')['mol_id']
    
    # Map spectrum IDs to molecule IDs
    # Using.get() avoids KeyErrors for spec_ids that might not be in the map
    pairs_df['mol_id_a'] = pairs_df['name_main'].map(spec_to_mol.get)
    pairs_df['mol_id_b'] = pairs_df['name_sub'].map(spec_to_mol.get)
    
    # Drop rows where a mapping was not found
    pairs_df.dropna(subset=['mol_id_a', 'mol_id_b'], inplace=True)

    pairs_df['mol_id_a'] = pairs_df['mol_id_a'].astype('int64')
    pairs_df['mol_id_b'] = pairs_df['mol_id_b'].astype('int64')
    
    # Filter for positive and negative pairs based on thresholds
    positive_pairs = pairs_df[pairs_df['cosine_similarity'] >= positive_threshold].copy()
    positive_pairs['label'] = 1
    
    negative_pairs = pairs_df[pairs_df['cosine_similarity'] <= negative_threshold].copy()
    negative_pairs['label'] = 0
    
    # Combine into a single dataframe
    contrastive_df = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
    
    # Keep only the necessary columns
    return contrastive_df[['mol_id_a', 'mol_id_b', 'label']]

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for contrastive loss pre-training.")
    
    # --- Paths ---
    parser.add_argument("--train_pairs_path", type=str, required=True, help="Path to the training pairs feather file.")
    parser.add_argument("--val_pairs_path", type=str, required=True, help="Path to the validation pairs feather file.")
    parser.add_argument("--test_pairs_path", type=str, required=True, help="Path to the testing pairs feather file.")
    parser.add_argument("--spec_df_path", type=str, required=True, help="Path to the spec_df.pkl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    
    # --- Thresholds ---
    parser.add_argument("--positive_threshold", type=float, default=0.8, help="Similarity threshold for positive pairs.")
    parser.add_argument("--negative_threshold", type=float, default=0.2, help="Similarity threshold for negative pairs.")
    
    args = parser.parse_args()

    print(f"Arguments: {args}")

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data Files ---")
    try:
        train_pairs_df = pd.read_feather(args.train_pairs_path)
        val_pairs_df = pd.read_feather(args.val_pairs_path)
        test_pairs_df = pd.read_feather(args.test_pairs_path)
        spec_df = pd.read_pickle(args.spec_df_path)
        print("All files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Details: {e}")
        return
        
    # --- 2. Process Training Data ---
    print(f"\n--- 2. Processing Training Pairs ---")
    print(f"Positive threshold: >= {args.positive_threshold}, Negative threshold: <= {args.negative_threshold}")
    
    contrastive_train_df = create_contrastive_set(train_pairs_df, spec_df, args.positive_threshold, args.negative_threshold)
    
    print(f"Generated {len(contrastive_train_df):,} total training pairs.")
    print(f" - Positive pairs (label=1): {contrastive_train_df['label'].sum():,}")
    print(f" - Negative pairs (label=0): {len(contrastive_train_df) - contrastive_train_df['label'].sum():,}")

    # --- 3. Process Validation Data ---
    print(f"\n--- 3. Processing Validation Pairs ---")
    contrastive_val_df = create_contrastive_set(val_pairs_df, spec_df, args.positive_threshold, args.negative_threshold)
    
    print(f"Generated {len(contrastive_val_df):,} total validation pairs.")
    print(f" - Positive pairs (label=1): {contrastive_val_df['label'].sum():,}")
    print(f" - Negative pairs (label=0): {len(contrastive_val_df) - contrastive_val_df['label'].sum():,}")

    # --- 4. Process Testing Data ---
    print(f"\n--- 4. Processing Testing Pairs ---")
    contrastive_test_df = create_contrastive_set(test_pairs_df, spec_df, args.positive_threshold, args.negative_threshold)

    print(f"Generated {len(contrastive_test_df):,} total testing pairs.")
    print(f" - Positive pairs (label=1): {contrastive_test_df['label'].sum():,}")
    print(f" - Negative pairs (label=0): {len(contrastive_test_df) - contrastive_test_df['label'].sum():,}")

    # --- 4. Save Processed Data ---
    os.makedirs(args.output_dir, exist_ok=True)
    train_output_path = os.path.join(args.output_dir, 'pretrain_contrastive_train.feather')
    val_output_path = os.path.join(args.output_dir, 'pretrain_contrastive_validation.feather')
    test_output_path = os.path.join(args.output_dir, 'pretrain_contrastive_test.feather')
    
    print(f"\n--- 4. Saving Processed Files ---")
    contrastive_train_df.to_feather(train_output_path)
    contrastive_val_df.to_feather(val_output_path)
    contrastive_test_df.to_feather(test_output_path)
    print(f"Training pairs saved to: {train_output_path}")
    print(f"Validation pairs saved to: {val_output_path}")
    print(f"Testing pairs saved to: {test_output_path}")
    
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()