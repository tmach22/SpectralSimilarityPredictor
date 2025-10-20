import argparse
import os
import time
import h5py
import lightgbm as lgb
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser(description="Train a LightGBM baseline model using pre-processed HDF5 data.")
    
    # --- Paths ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to the HDF5 file containing the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    
    # --- Model Hyperparameters ---
    parser.add_argument("--n_estimators", type=int, default=2000, help="Maximum number of boosting rounds.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for the model.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")

    args = parser.parse_args()

    # --- 1. Load Data into Memory ---
    print("\n--- 1. Loading Pre-processed Data from HDF5 into RAM ---")
    try:
        with h5py.File(args.data_path, 'r') as hf:
            # Use [:] to read the entire dataset from disk into a NumPy array in memory
            X_train = hf['train']['X'][:]
            y_train = hf['train']['y'][:]
            X_val = hf['validation']['X'][:]
            y_val = hf['validation']['y'][:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print("Data successfully loaded into memory.")

    except Exception as e:
        print(f"Error: Could not open or read HDF5 file at {args.data_path}.")
        print(f"Details: {e}")
        return

    # --- 2. Create LightGBM Datasets ---
    print("\n--- 2. Creating LightGBM Dataset objects ---")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # --- 3. Model Configuration and Training ---
    print(f"\n--- 3. Initializing and Training LightGBM Model ---")
    
    params = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': 32, # Use all available CPU cores
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks= [lgb.early_stopping(args.patience)]
    )
    
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    print(f"Best iteration found: {model.best_iteration}")

    # --- 4. Save the Trained Model ---
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'baseline_lgbm_best.joblib')
    print(f"\n--- 4. Saving Best Model to {model_path} ---")
    joblib.dump(model, model_path)

    # --- 5. Final Evaluation ---
    print("\n--- 5. Evaluating Best Model on Validation Set ---")
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_val, y_pred)
    
    print("\n--- Final Performance Metrics ---")
    print(f"Mean Squared Error (MSE):     {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Pearson Correlation (r):      {pearson_corr:.6f}")
    
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()