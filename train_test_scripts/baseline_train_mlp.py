import pandas as pd
import numpy as np
import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# --- 1. PYTORCH MODEL ---
class FingerprintMLP(nn.Module):
    def __init__(self, input_dim=4096):
        super(FingerprintMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2), # Added dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output logits (no sigmoid here, handled by Loss)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. DATA PREPARATION ---
def build_id_to_smiles_map(spec_path, mol_path):
    print("Building ID -> SMILES mapping...")
    spec_df = pd.read_pickle(spec_path)
    mol_df = pd.read_pickle(mol_path)

    if 'inchikey' in mol_df.columns:
        mol_df = mol_df.set_index('inchikey')
    
    id_to_smiles = {}
    
    # Vectorized lookup where possible is hard due to mapping logic, 
    # but we only do this once.
    for spec_id, row in tqdm(spec_df.iterrows(), total=len(spec_df), desc="Mapping Spectra"):
        smiles = None
        if 'smiles' in row and pd.notna(row['smiles']):
            smiles = row['smiles']
        elif 'inchikey' in row and pd.notna(row['inchikey']):
            ikey = row['inchikey']
            if ikey in mol_df.index:
                mol_data = mol_df.loc[ikey]
                smiles = mol_data['smiles'] if isinstance(mol_data, pd.Series) else mol_data.iloc[0]['smiles']
        
        if smiles:
            id_to_smiles[spec_id] = smiles
            
    print(f"Mapped {len(id_to_smiles)} spectra to structures.")
    return id_to_smiles

def generate_fingerprints_map(unique_ids, id_to_smiles_map):
    """
    Pre-computes FPs for a set of IDs to avoid redundant RDKit work.
    """
    print(f"Generating fingerprints for {len(unique_ids)} unique compounds...")
    fp_map = {}
    
    for uid in tqdm(unique_ids, desc="RDKit Gen"):
        if uid not in id_to_smiles_map:
            fp_map[uid] = np.zeros((2048,), dtype=np.float32)
            continue
            
        smiles = id_to_smiles_map[uid]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_map[uid] = arr
        else:
            fp_map[uid] = np.zeros((2048,), dtype=np.float32)
            
    return fp_map

def prepare_data(pairs_path, id_to_smiles_map, cache_dir, split_name="Train"):
    # Check if cached tensors exist
    cache_file = os.path.join(cache_dir, f"{split_name}_data.pt")
    if os.path.exists(cache_file):
        print(f"Loading cached {split_name} data from {cache_file}...")
        return torch.load(cache_file)

    print(f"--- Processing {split_name} Data from {os.path.basename(pairs_path)} ---")
    pairs_df = pd.read_feather(pairs_path)
    
    # 1. Identify all unique IDs in this split
    unique_ids = set(pairs_df['name_main']).union(set(pairs_df['name_sub']))
    
    # 2. Generate FPs for them (Single pass)
    fp_map = generate_fingerprints_map(unique_ids, id_to_smiles_map)
    
    # 3. Vectorized Construction using List Comprehension (Much faster than iterrows)
    print(f"Constructing feature matrix for {len(pairs_df)} pairs...")
    
    # Get arrays of IDs
    ids1 = pairs_df['name_main'].values
    ids2 = pairs_df['name_sub'].values
    
    # Determine Label Column
    label_col = 'label' if 'label' in pairs_df.columns else 'cosine_similarity'
    labels_raw = pairs_df[label_col].values
    
    # Build X
    # We iterate over the numpy arrays which is fast
    X_list = []
    for i in tqdm(range(len(ids1)), desc="Stacking Features"):
        fp1 = fp_map.get(ids1[i], np.zeros((2048,), dtype=np.float32))
        fp2 = fp_map.get(ids2[i], np.zeros((2048,), dtype=np.float32))
        X_list.append(np.concatenate([fp1, fp2]))
        
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    
    # Build y
    if label_col == 'cosine_similarity':
        y = torch.tensor((labels_raw >= 0.7).astype(np.float32)).unsqueeze(1)
    else:
        y = torch.tensor(labels_raw.astype(np.float32)).unsqueeze(1)
        
    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    torch.save((X, y), cache_file)
    print(f"Saved cache to {cache_file}")
    
    return X, y

# --- 3. TRAINING FUNCTION ---
def train_mlp_gpu(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 0. Global Map
    id_map = build_id_to_smiles_map(args.spec_data_path, args.mol_data_path)
    
    # 1. Prepare Data
    X_train_part, y_train_part = prepare_data(args.train_pairs_path, id_map, args.cache_dir, "train")
    X_val_part, y_val_part = prepare_data(args.val_pairs_path, id_map, args.cache_dir, "val")
    
    print("Merging Train and Validation tensors...")
    X_train = torch.cat([X_train_part, X_val_part], dim=0)
    y_train = torch.cat([y_train_part, y_val_part], dim=0)
    
    X_test, y_test = prepare_data(args.test_pairs_path, id_map, args.cache_dir, "test")

    # Move Test data to GPU for fast evaluation later
    # (Keep training data on CPU and move in batches to avoid OOM if dataset is huge)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Setup Model
    model = FingerprintMLP(input_dim=4096).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Training Loop
    print(f"\n--- Starting Training on {device} ---")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

    # 4. Evaluation
    print("\n--- Evaluating on Test Set ---")
    model.eval()
    with torch.no_grad():
        # Process test set in batches to avoid OOM
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)
        all_preds = []
        all_probs = []
        
        for batch_X, _ in tqdm(test_loader, desc="Inference"):
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = y_test.numpy()
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    
    print(f"\n--- Final GPU MLP Baseline Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC:  {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    res_df = pd.DataFrame({
        'y_true': y_true.flatten(), 
        'y_pred': y_pred.flatten(), 
        'y_prob': y_prob.flatten()
    })
    save_path = os.path.join(args.output_dir, "mlp_gpu_results.csv")
    res_df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, default="./results/baseline_mlp_gpu")
    parser.add_argument("--cache_dir", type=str, default="./data_cache_mlp", help="Where to save processed tensors")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    args = parser.parse_args()
    train_mlp_gpu(args)