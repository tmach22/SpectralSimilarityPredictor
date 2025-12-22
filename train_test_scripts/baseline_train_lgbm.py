import pandas as pd
import numpy as np
import argparse
import os
import sys
import pickle
import lightgbm as lgb
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# --- 1. DATA PREPARATION HELPERS ---
def build_id_to_smiles_map(spec_path, mol_path):
    print("Building ID -> SMILES mapping...")
    spec_df = pd.read_pickle(spec_path)
    mol_df = pd.read_pickle(mol_path)

    # --- STEP A: Build Mol_ID -> SMILES Map ---
    print("Indexing Molecule Data...")
    mol_id_to_smiles = {}
    
    # Try to find the join column in mol_df
    # It might be the index, or a column named 'mol_id' / 'id'
    mol_join_col = None
    if 'mol_id' in mol_df.columns:
        mol_join_col = 'mol_id'
    elif 'id' in mol_df.columns:
        mol_join_col = 'id'
        
    for idx, row in tqdm(mol_df.iterrows(), total=len(mol_df), desc="Indexing Molecules"):
        # Determine Key (Integer ID)
        if mol_join_col:
            mid = row[mol_join_col]
        else:
            mid = idx # Fallback to using the dataframe index
            
        # Determine Value (SMILES)
        smiles = None
        if 'smiles' in row and pd.notna(row['smiles']):
            smiles = row['smiles']
        elif 'canonical_smiles' in row and pd.notna(row['canonical_smiles']):
            smiles = row['canonical_smiles']
            
        if mid is not None and smiles:
            mol_id_to_smiles[mid] = smiles

    # --- STEP B: Link Spec_ID -> Mol_ID -> SMILES ---
    print("Linking Spectra to Structures...")
    id_to_smiles = {}
    
    # Verify columns exist based on your file info
    if 'spec_id' not in spec_df.columns or 'mol_id' not in spec_df.columns:
        print("WARNING: 'spec_id' or 'mol_id' column missing. Trying heuristics...")
        # Fallback logic if names slightly differ
        spec_col = 'spec_id' if 'spec_id' in spec_df.columns else spec_df.columns[0]
        mol_ref_col = 'mol_id' if 'mol_id' in spec_df.columns else 'mol_index'
    else:
        spec_col = 'spec_id'
        mol_ref_col = 'mol_id'

    for _, row in tqdm(spec_df.iterrows(), total=len(spec_df), desc="Mapping"):
        sid = row[spec_col]      # e.g., MassSpecGymID...
        mid = row[mol_ref_col]   # e.g., 20678
        
        if mid in mol_id_to_smiles:
            id_to_smiles[sid] = mol_id_to_smiles[mid]
            
    print(f"Mapped {len(id_to_smiles)} spectra to structures.")
    if len(id_to_smiles) == 0:
        raise ValueError("FATAL: Mapping failed. Check if mol_id in spec_df matches mol_df index/columns.")
        
    return id_to_smiles

def generate_fingerprints_map(unique_ids, id_to_smiles_map):
    print(f"Generating fingerprints for {len(unique_ids)} unique spectra...")
    fp_map = {}
    
    for uid in tqdm(unique_ids, desc="RDKit Gen"):
        if uid not in id_to_smiles_map:
            # Zero vector if no structure found
            fp_map[uid] = np.zeros((2048,), dtype=np.uint8)
            continue
            
        smiles = id_to_smiles_map[uid]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((2048,), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fp_map[uid] = arr
            else:
                fp_map[uid] = np.zeros((2048,), dtype=np.uint8)
        except:
            fp_map[uid] = np.zeros((2048,), dtype=np.uint8)
            
    return fp_map

def prepare_data(pairs_path, id_to_smiles_map, cache_dir, split_name="Train"):
    # Cache file v3 to ensure we don't load old bad data
    cache_file = os.path.join(cache_dir, f"{split_name}_lgbm_data_v3.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached {split_name} data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"--- Processing {split_name} Data from {os.path.basename(pairs_path)} ---")
    pairs_df = pd.read_feather(pairs_path)
    
    unique_ids = set(pairs_df['name_main']).union(set(pairs_df['name_sub']))
    fp_map = generate_fingerprints_map(unique_ids, id_to_smiles_map)
    
    print(f"Constructing feature matrix for {len(pairs_df)} pairs...")
    ids1 = pairs_df['name_main'].values
    ids2 = pairs_df['name_sub'].values
    label_col = 'label' if 'label' in pairs_df.columns else 'cosine_similarity'
    labels_raw = pairs_df[label_col].values
    
    # Vectorized List Comprehension
    X_list = [
        np.concatenate([
            fp_map.get(id1, np.zeros((2048,), dtype=np.uint8)),
            fp_map.get(id2, np.zeros((2048,), dtype=np.uint8))
        ])
        for id1, id2 in tqdm(zip(ids1, ids2), total=len(ids1), desc="Stacking")
    ]
        
    X = np.array(X_list, dtype=np.float32)
    
    if label_col == 'cosine_similarity':
        y = (labels_raw >= 0.7).astype(int)
    else:
        y = labels_raw.astype(int)
        
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"Saved cache to {cache_file}")
    
    return X, y

# --- 2. TRAINING FUNCTION ---
def train_lgbm(args):
    print(f"--- Starting LightGBM Baseline Training ---")
    
    # 0. Global Map
    id_map = build_id_to_smiles_map(args.spec_data_path, args.mol_data_path)
    
    # 1. Prepare Data
    X_train, y_train = prepare_data(args.train_pairs_path, id_map, args.cache_dir, "train")
    X_val, y_val = prepare_data(args.val_pairs_path, id_map, args.cache_dir, "val")
    X_test, y_test = prepare_data(args.test_pairs_path, id_map, args.cache_dir, "test")

    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    
    # 2. Setup LGBM Model
    print("\n--- Initializing LGBM Classifier ---")
    clf = lgb.LGBMClassifier(
        n_estimators=2000,      
        learning_rate=0.05,     
        num_leaves=31,          
        objective='binary',
        n_jobs=-1,              
        random_state=42
    )
    
    # 3. Train
    print("Training...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # 4. Evaluation
    print("\n--- Evaluating on Test Set ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"\n--- Final LGBM Baseline Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC:  {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs(args.output_dir, exist_ok=True)
    res_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob})
    res_df.to_csv(os.path.join(args.output_dir, "lgbm_results.csv"), index=False)
    print(f"Saved results to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--val_pairs_path", type=str, required=True)
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
       
    parser.add_argument("--output_dir", type=str, default="./baseline_lgbm")
    # New default cache dir
    parser.add_argument("--cache_dir", type=str, default="./data_cache_lgbm_v3")
    
    args = parser.parse_args()
    train_lgbm(args)