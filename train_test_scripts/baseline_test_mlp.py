import argparse
import os
import time
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# --- 1. Data Loader and Model Definition (must match training script) ---

class HDF5Dataset(Dataset):
    """A Dataset class for loading data from an HDF5 file."""
    def __init__(self, h5_path, group_name):
        self.h5_file = h5py.File(h5_path, 'r')
        self.X = self.h5_file[group_name]['X']
        self.y = self.h5_file[group_name]['y']
    
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        # Fetches a single data point
        return torch.from_numpy(self.X[index].astype(np.float32)), torch.tensor(self.y[index], dtype=torch.float32)

    def close(self):
        self.h5_file.close()

class MLPRegressor(nn.Module):
    """A simple MLP for regression, designed for GPU execution."""
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

def main():
    parser = argparse.ArgumentParser(description="Test a trained PyTorch MLP baseline model.")
    
    # --- Paths ---
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained PyTorch model file (e.g., baseline_mlp_pytorch_best.pt).")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the HDF5 file containing the dataset.")
    parser.add_argument("--group_name", type=str, default="validation", 
                        help="Name of the group in the HDF5 file to use for testing (e.g., 'validation' or 'test').")
    parser.add_argument("--output_csv", type=str, default=None, 
                        help="Optional: Path to save a CSV file with predictions and true values.")
    
    # --- Inference Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader.")

    args = parser.parse_args()

    # --- 1. Setup Device and Load Model ---
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device: {device} ---")

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Get input dimension from the dataset to correctly initialize the model
    try:
        with h5py.File(args.data_path, 'r') as hf:
            if args.group_name not in hf:
                print(f"Error: Group '{args.group_name}' not found in HDF5 file.")
                return
            input_dim = hf[args.group_name]['X'].shape[1]
    except Exception as e:
        print(f"Error reading HDF5 file to determine input dimension: {e}")
        return

    model = MLPRegressor(input_dim).to(device)
    print(f"\n--- 1. Loading trained model from {args.model_path} ---")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # --- 2. Load Test Data ---
    print(f"\n--- 2. Loading test data from {args.data_path} (Group: {args.group_name}) ---")
    test_dataset = HDF5Dataset(args.data_path, args.group_name)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    print(f"Test data shape: {test_dataset.X.shape}")

    # --- 3. Make Predictions ---
    print("\n--- 3. Making predictions on the test set ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Predicting"):
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    
    duration = time.time() - start_time
    print(f"Prediction completed in {duration:.2f} seconds.")

    y_pred = np.concatenate(all_preds)
    y_test = np.concatenate(all_labels)

    # --- 4. Evaluate Performance ---
    print("\n--- 4. Evaluating Model Performance ---")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    
    print("\n--- Final Performance Metrics ---")
    print(f"Mean Squared Error (MSE):     {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Pearson Correlation (r):      {pearson_corr:.6f}")

    # --- 5. Save Predictions (Optional) ---
    if args.output_csv:
        print(f"\n--- 5. Saving predictions to {args.output_csv} ---")
        results_df = pd.DataFrame({
            'true_similarity': y_test,
            'predicted_similarity': y_pred
        })
        try:
            results_df.to_csv(args.output_csv, index=False)
            print("Predictions saved successfully.")
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")

    test_dataset.close()
    print("\n--- Testing Process Complete ---")

if __name__ == '__main__':
    main()