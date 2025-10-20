import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import os
from pathlib import Path
import copy
import numpy as np
from scipy.stats import pearsonr
import sys

# NEW: Import the necessary PEFT libraries
from peft import PeftModel

# --- (Your existing sys.path setup) ---
cwd = Path.cwd()
data_loader_dir = os.path.join(cwd, 'data_loaders')
sys.path.insert(0, data_loader_dir)
model_dir = os.path.join(cwd, 'model')
sys.path.insert(0, model_dir)

# --- Import our custom classes ---
from siamesemodel import SiameseSpectralSimilarityModel

parent_directory = os.path.dirname(cwd.parent)
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
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
    Main function to load a PEFT-tuned model and evaluate it on a test dataset.
    """
    # --- 1. Setup and Configuration ---
    print("--- 1. Setting up PEFT Testing Environment ---")
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
    print(f"Using device: {device}")

    # Load configurations to reconstruct the base model architecture
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
        subset_size=args.subset_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=siamese_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Test dataset contains {len(test_dataset)} pairs.")

    # --- 3. Model Loading ---
    print("\n--- 3. Initializing Base Model and Loading PEFT Adapter ---")
    
    # First, instantiate the base model architecture. This loads the original
    # pre-trained MassFormer weights into the encoder.
    base_model = SiameseSpectralSimilarityModel(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path
    )

    # Then, load the state dictionary from our Phase 1 training. This loads the
    # trained weights for the similarity head.
    print(f"Loading Phase 1 weights from: {args.phase1_checkpoint_path}")
    base_model.load_state_dict(torch.load(args.phase1_checkpoint_path, map_location=device))
    
    # =============================================================================
    # *** NEW: Load the trained PEFT adapter on top of the base model ***
    # =============================================================================
    print(f"Loading PEFT adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    
    # --- 4. Evaluation Loop ---
    print("\n--- 4. Starting Model Evaluation on Test Set ---")
    model.eval()
    
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for batch_A, batch_B, similarities in tqdm(test_loader, desc="Testing"):
            for key in batch_A: batch_A[key] = batch_A[key].to(device, non_blocking=True)
            for key in batch_B: batch_B[key] = batch_B[key].to(device, non_blocking=True)
            
            predictions = model(batch_A, batch_B)
            
            all_predictions.append(predictions.squeeze().cpu())
            all_ground_truth.append(similarities.cpu())

    all_predictions = torch.cat(all_predictions).numpy()
    all_ground_truth = torch.cat(all_ground_truth).numpy()

    # --- 5. Calculate and Report Metrics ---
    print("\n--- 5. Performance Metrics ---")
    
    mse = np.mean((all_predictions - all_ground_truth)**2)
    print(f"Mean Squared Error (MSE):     {mse:.6f}")

    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

    mae = np.mean(np.abs(all_predictions - all_ground_truth))
    print(f"Mean Absolute Error (MAE):    {mae:.6f}")
    
    pearson_corr, p_value = pearsonr(all_predictions, all_ground_truth)
    print(f"Pearson Correlation (r):      {pearson_corr:.6f}")
    print(f"P-value:                      {p_value:.2e}")

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
    parser = argparse.ArgumentParser(description="Test the PEFT-tuned Siamese Spectral Similarity Model.")
    
    # --- Data & Config Paths ---
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    
    # --- Model & Checkpoint Paths ---
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original MassFormer checkpoint (demo.pkl).")
    parser.add_argument("--phase1_checkpoint_path", type=str, required=True, help="Path to your Phase 1 trained model (best_model.pth).")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the directory containing the trained PEFT adapter.")

    # --- Evaluation Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset_size", type=float, default=None)

    parser.add_argument("--save_results_path", type=str, default=None, help="Path to save the final predictions and ground truth arrays (e.g., results.npz).")

    # --- System Configuration ---
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    test_model(args)