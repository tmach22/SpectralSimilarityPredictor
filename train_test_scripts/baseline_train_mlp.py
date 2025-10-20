import argparse
import os
import time
import numpy as np
import h5py
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm

# --- 1. GPU-Aware HDF5 DataLoader ---
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

# --- 2. PyTorch MLP Model Definition ---
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
    parser = argparse.ArgumentParser(description="Train a GPU-accelerated PyTorch MLP baseline.")
    
    # --- Paths ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to the HDF5 file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=50, help="Max number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader.")

    args = parser.parse_args()

    # --- 1. Setup Device and DataLoaders ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device: {device} ---")

    print("\n--- 1. Loading Data ---")
    train_dataset = HDF5Dataset(args.data_path, 'train')
    val_dataset = HDF5Dataset(args.data_path, 'validation')

    # Optimized DataLoader for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # --- 2. Model Initialization ---
    input_dim = train_dataset.X.shape[1]
    model = MLPRegressor(input_dim).to(device)
    print(f"\n--- Model Initialized with {sum(p.numel() for p in model.parameters())} parameters ---")

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- 3. Training Loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    model_path = os.path.join(args.output_dir, 'baseline_mlp_pytorch_best.pt')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = loss_function(predictions, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s")

        # --- Early Stopping and Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} (Val MSE: {best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
            break
            
    # --- 4. Final Evaluation ---
    print("\n--- 4. Evaluating Best Model on Validation Set ---")
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Final Evaluation"):
                X_batch = X_batch.to(device)
                predictions = model(X_batch)
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
        
        y_pred = np.concatenate(all_preds)
        y_val = np.concatenate(all_labels)
        
        mse = np.mean((y_val - y_pred)**2)
        rmse = np.sqrt(mse)
        pearson_corr, _ = pearsonr(y_val, y_pred)
        
        print("\n--- Final Performance Metrics ---")
        print(f"Mean Squared Error (MSE):     {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Pearson Correlation (r):      {pearson_corr:.6f}")

    except FileNotFoundError:
        print(f"Error: Could not find the saved model at {model_path}. Training may have failed to improve.")

    train_dataset.close()
    val_dataset.close()
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()