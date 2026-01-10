import pandas as pd
import argparse
import os

def create_relabelled_dataset(input_path, output_path, strict_thresh=0.8):
    print(f"Processing: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    try:
        df = pd.read_feather(input_path)
    except:
        df = pd.read_csv(input_path)
        
    initial_len = len(df)
    
    # --- THE LOGIC CHANGE ---
    # 1. New Positives: Cosine >= 0.8
    # 2. New Negatives: Cosine < 0.8 (Includes the old 0.7-0.79 range!)
    
    # We simply overwrite the label column
    # Note: We rely on 'cosine_similarity' column being present and accurate
    df['label'] = df['cosine_similarity'].apply(lambda x: 1.0 if x >= strict_thresh else 0.0)
    
    # Stats
    n_pos = len(df[df['label'] == 1.0])
    n_neg = len(df[df['label'] == 0.0])
    ratio = n_neg / n_pos if n_pos > 0 else 0
    
    print(f"  Total Count:    {len(df)}")
    print(f"  New Positives (>= {strict_thresh}): {n_pos}")
    print(f"  New Negatives (< {strict_thresh}):  {n_neg}")
    print(f"  Imbalance Ratio:    1 Positive : {ratio:.2f} Negatives")
    
    # Save
    df.to_feather(output_path)
    print(f"  Saved to: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data_splits_relabelled")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create Train
    train_out = os.path.join(args.output_dir, "binary_08_relabelled_train.feather")
    create_relabelled_dataset(args.train_path, train_out)
    
    # Create Val
    val_out = os.path.join(args.output_dir, "binary_08_relabelled_val.feather")
    create_relabelled_dataset(args.val_path, val_out)

    # Create Test
    test_out = os.path.join(args.output_dir, "binary_08_relabelled_test.feather")
    create_relabelled_dataset(args.test_path, test_out)