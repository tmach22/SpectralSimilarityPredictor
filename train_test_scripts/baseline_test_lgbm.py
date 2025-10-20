import argparse
import os
import time
import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser(description="Test a trained LightGBM baseline model.")
    
    # --- Paths ---
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained LightGBM model file (e.g., baseline_lgbm_best.joblib).")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the HDF5 file containing the dataset.")
    parser.add_argument("--group_name", type=str, default="validation", 
                        help="Name of the group in the HDF5 file to use for testing (e.g., 'validation' or 'test').")
    parser.add_argument("--output_csv", type=str, default=None, 
                        help="Optional: Path to save a CSV file with predictions and true values.")

    args = parser.parse_args()

    # --- 1. Load Model ---
    print(f"\n--- 1. Loading trained model from {args.model_path} ---")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    try:
        model = joblib.load(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Load Test Data ---
    print(f"\n--- 2. Loading test data from {args.data_path} (Group: {args.group_name}) ---")
    try:
        with h5py.File(args.data_path, 'r') as hf:
            if args.group_name not in hf:
                print(f"Error: Group '{args.group_name}' not found in HDF5 file.")
                return
            # Use [:] to read the entire dataset from disk into a NumPy array
            X_test = hf[args.group_name]['X'][:]
            y_test = hf[args.group_name]['y'][:]
        
        print(f"Test data shape: {X_test.shape}")
        print("Data successfully loaded into memory.")

    except Exception as e:
        print(f"Error: Could not open or read HDF5 file at {args.data_path}.")
        print(f"Details: {e}")
        return

    # --- 3. Make Predictions ---
    print("\n--- 3. Making predictions on the test set ---")
    start_time = time.time()
    
    # Use the best iteration found during training for prediction
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    duration = time.time() - start_time
    print(f"Prediction completed in {duration:.2f} seconds.")

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

    print("\n--- Testing Process Complete ---")

if __name__ == '__main__':
    main()