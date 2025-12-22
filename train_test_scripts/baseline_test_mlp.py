import pandas as pd
import numpy as np
import argparse
import os
import sys
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

def compute_fp(smiles):
    """Generates a 2048-bit Morgan Fingerprint (radius=2)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Generate bit vector and convert to numpy array
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        else:
            return np.zeros((2048,), dtype=np.int8)
    except:
        return np.zeros((2048,), dtype=np.int8)

def prepare_data(pairs_path, mol_path, split_name="Train"):
    print(f"--- Loading {split_name} Data ---")
    
    # 1. Load Dataframes
    pairs_df = pd.read_feather(pairs_path)
    mol_df = pd.read_pickle(mol_path)
    
    # Map InChIKey -> SMILES
    # Assuming mol_df index is InChIKey or it has a column 'inchikey'
    if 'inchikey' in mol_df.columns:
        mol_df = mol_df.set_index('inchikey')
    
    # 2. Pre-compute Fingerprints for all relevant molecules
    # (Optimization: only compute unique molecules in this split)
    relevant_keys = set(pairs_df['inchikey1']).union(set(pairs_df['inchikey2']))
    print(f"Generating fingerprints for {len(relevant_keys)} unique molecules...")
    
    fp_cache = {}
    for key in tqdm(relevant_keys):
        if key in mol_df.index:
            smiles = mol_df.loc[key]['smiles']
            # Handle case where index might have duplicates
            if isinstance(smiles, pd.Series): smiles = smiles.iloc[0]
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((2048,), dtype=np.float32) # Float for MLP
                from rdkit.Chem import DataStructs
                DataStructs.ConvertToNumpyArray(fp, arr)
                fp_cache[key] = arr
            else:
                fp_cache[key] = np.zeros((2048,), dtype=np.float32)
        else:
            fp_cache[key] = np.zeros((2048,), dtype=np.float32)

    # 3. Build X (Features) and y (Labels)
    print(f"Building feature matrix for {len(pairs_df)} pairs...")
    X_list = []
    y_list = []
    
    # Determine label column
    label_col = 'label' if 'label' in pairs_df.columns else 'cosine_similarity'
    
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        fp1 = fp_cache.get(row['inchikey1'], np.zeros((2048,), dtype=np.float32))
        fp2 = fp_cache.get(row['inchikey2'], np.zeros((2048,), dtype=np.float32))
        
        # FEATURE ENGINEERING:
        # Standard approach for Siamese-like MLP: Concatenate [FP1, FP2]
        # Alternative: [FP1, FP2, FP1*FP2, |FP1-FP2|] -> But let's stick to concat for a direct baseline
        features = np.concatenate([fp1, fp2])
        X_list.append(features)
        
        # Label handling
        if label_col == 'cosine_similarity':
            label = 1 if row[label_col] >= 0.7 else 0
        else:
            label = int(row[label_col])
        y_list.append(label)
        
    return np.array(X_list), np.array(y_list)

def train_mlp_baseline(args):
    # 1. Prepare Data
    X_train, y_train = prepare_data(args.train_pairs_path, args.mol_data_path, "Train")
    X_test, y_test = prepare_data(args.test_pairs_path, args.mol_data_path, "Test")
    
    print(f"Train Input Shape: {X_train.shape}")
    print(f"Test Input Shape: {X_test.shape}")
    
    # 2. Define Model (scikit-learn MLP)
    # Architecture: Input(4096) -> 1024 -> 512 -> 256 -> 1
    # This mimics a reasonably deep dense baseline
    print("\n--- Training MLP Classifier ---")
    clf = MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256),
        activation='relu',
        solver='adam',
        alpha=1e-4, # Weight decay
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=20, # Equivalent to epochs
        early_stopping=True,
        verbose=True,
        random_state=42
    )
    
    # 3. Train
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\n--- Evaluating on Test Set ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"\nResults for MLP (Morgan Fingerprints):")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC:  {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    results_df.to_csv(os.path.join(args.output_dir, "mlp_fingerprint_results.csv"), index=False)
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/baseline_mlp")
    
    args = parser.parse_args()
    train_mlp_baseline(args)