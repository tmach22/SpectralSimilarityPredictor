import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import copy
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

import os
import sys
cwd = Path.cwd()

data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)

model_dir = os.path.join(cwd, 'model')
print(f"Adding {model_dir} to sys.path")
sys.path.insert(0, model_dir)

# --- Import our custom classes ---
# Assuming this script is in the same directory as your model and data loader scripts
from siamesemodel import SiameseSpectralSimilarityModel

parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
# Add the parent directory to the Python path
sys.path.insert(0, script_dir)

from data_loader import SpectralSimilarityDataset, siamese_collate_fn

def merge_configs(base_config, custom_config):
    """
    Recursively merges the custom config into the base config.
    """
    merged_config = copy.deepcopy(base_config)
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    return merged_config

def test_model(args):
    """
    Main function to load a trained model and evaluate it on a test dataset.
    """
    # --- 1. Setup and Configuration ---
    print("--- 1. Setting up testing environment ---")
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
    print(f"Using device: {device}")

    # Load configurations to reconstruct the model architecture correctly
    print(f"Loading template configuration from: {args.template_config_path}")
    with open(args.template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    print(f"Loading custom configuration from: {args.custom_config_path}")
    with open(args.custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)

    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
    # --- 2. Data Loading ---
    print("\n--- 2. Initializing Test Dataset and DataLoader ---")
    
    test_dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.test_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        subset_size=args.subset_size  # Use the subset size argument
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # No need to shuffle for testing
        collate_fn=siamese_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Test dataset contains {len(test_dataset)} pairs.")

    # --- 3. Model Loading ---
    print("\n--- 3. Initializing Model and Loading Trained Weights ---")
    
    # First, instantiate the model architecture. This will also load the
    # original pre-trained MassFormer weights into the encoder.
    model = SiameseSpectralSimilarityModel(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path
    )

    # Now, load the state dictionary from our own training (e.g., 'best_model.pth')
    # This will overwrite the encoder weights with our fine-tuned ones and
    # load the trained weights for the similarity head.
    print(f"Loading fine-tuned weights from: {args.model_checkpoint_path}")
    model.load_state_dict(torch.load(args.model_checkpoint_path, map_location=device))
    model.to(device)
    
    # --- 4. Evaluation Loop ---
    print("\n--- 4. Starting Model Evaluation on Test Set ---")
    model.eval()  # Set the model to evaluation mode
    
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for batch_A, batch_B, similarities in tqdm(test_loader, desc="Testing"):
            # Move data to the selected device
            for key in batch_A: batch_A[key] = batch_A[key].to(device, non_blocking=True)
            for key in batch_B: batch_B[key] = batch_B[key].to(device, non_blocking=True)
            
            # Forward pass
            predictions = model(batch_A, batch_B)
            
            # Store predictions and ground truth values
            all_predictions.append(predictions.squeeze().cpu())
            all_ground_truth.append(similarities.cpu())

    # Concatenate all batches into single tensors
    all_predictions = torch.cat(all_predictions).numpy()
    all_ground_truth = torch.cat(all_ground_truth).numpy()

    # --- 5. Calculate and Report Metrics ---
    print("\n--- 5. Performance Metrics ---")
    
    # Mean Squared Error (MSE)
    mse = np.mean((all_predictions - all_ground_truth)**2)
    print(f"Mean Squared Error (MSE):     {mse:.6f}")

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(all_predictions - all_ground_truth))
    print(f"Mean Absolute Error (MAE):    {mae:.6f}")
    
    # Pearson Correlation Coefficient
    pearson_corr, p_value = pearsonr(all_predictions, all_ground_truth)
    print(f"Pearson Correlation (r):      {pearson_corr:.6f}")
    print(f"P-value:                      {p_value:.6e}")

    if args.save_results_path:
        print(f"\nSaving prediction results to {args.save_results_path}...")
        np.savez_compressed(
            args.save_results_path,
            predictions=all_predictions,
            ground_truth=all_ground_truth
        )
        print("Results saved successfully.")
    
    print("\n--- Testing Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the Siamese Spectral Similarity Model.")
    
    # --- Data Paths ---
    parser.add_argument("--test_pairs_path", type=str, required=True, help="Path to the test pairs feather file.")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to the spec_df.pkl file.")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to the mol_df.pkl file.")
    
    # --- Model & Config Paths ---
    parser.add_argument("--template_config_path", type=str, required=True, help="Path to the template.yml config file.")
    parser.add_argument("--custom_config_path", type=str, required=True, help="Path to the custom experiment config (e.g., demo_eval.yml).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original pre-trained MassFormer checkpoint (e.g., checkpoints/demo.pkl).")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to your trained Siamese model checkpoint (e.g., best_model.pth).")

    # --- Evaluation Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--subset_size", type=float, default=None, help="Use a subset of the data. E.g., 0.1 for 10% or 10000 for 10k pairs.")

    # --- System Configuration ---
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use. Set to -1 for CPU.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--save_results_path", type=str, default=None, help="Path to save the final predictions and ground truth arrays (e.g., results.npz).")

    args = parser.parse_args()
    test_model(args)