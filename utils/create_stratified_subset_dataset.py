import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd  # <--- NEW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split # <--- NEW

# Setup paths
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from classifier_siamesemodel_new import SiameseSpectralSimilarityModel
from updated_train import merge_configs
from binary_data_loader import BinaryClassificationDataset, binary_collate_fn

def create_stratified_subset(input_path, fraction, output_dir, seed=42):
    """
    Helper: Loads a feather file and samples a fraction while preserving 
    the cosine similarity distribution and class balance.
    """
    print(f"   -> Loading {os.path.basename(input_path)}...")
    df = pd.read_feather(input_path)
    
    # 1. Create a Stratification Key
    # We combine Label (0/1) + Cosine Bin (0.0-1.0) to ensure the subset 
    # looks exactly like the full dataset.
    df['sim_bin'] = pd.cut(df['cosine_similarity'], bins=10, labels=False)
    # Key = "Label_BinIndex" (e.g., "1.0_9" or "0.0_2")
    df['stratify_key'] = df['label'].astype(str) + "_" + df['sim_bin'].astype(str)
    
    # 2. Stratified Split
    print(f"   -> Sampling {fraction*100}% stratified subset...")
    # We use train_test_split as a fast sampler
    subset_df, _ = train_test_split(
        df, 
        train_size=fraction, 
        stratify=df['stratify_key'], 
        random_state=seed
    )
    
    # 3. Clean up
    subset_df = subset_df.drop(columns=['sim_bin', 'stratify_key']).reset_index(drop=True)
    
    # 4. Save Temp File
    temp_filename = f"temp_subset_{fraction}_{os.path.basename(input_path)}"
    temp_path = os.path.join(output_dir, temp_filename)
    subset_df.to_feather(temp_path)
    
    print(f"   -> Created subset: {len(subset_df):,} pairs")
    print(f"   -> Saved to: {temp_path}")
    return temp_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data Paths
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of training data to use (0.0 to 1.0). Default 1.0 (Full).")
    parser.add_argument("--output_dir", type=str, default="./data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs")
    args = parser.parse_args()

    output_path = create_stratified_subset(args.dataset_path, args.subset_fraction, args.output_dir)
    print(f"Stratified Dataset created at location: {output_path}")