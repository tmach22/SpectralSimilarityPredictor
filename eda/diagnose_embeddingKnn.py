import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics.pairwise import paired_cosine_distances

# --- 1. SETUP SYS.PATH ---
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, "train_test_scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from classifier_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def extract_pair_features(model, dataloader, device, desc="Extracting"):
    """
    Extracts embeddings for pairs (A, B) and returns features + labels.
    """
    model.eval()
    
    list_emb_A = []
    list_emb_B = []
    list_labels = []
    
    with torch.no_grad():
        for batch_A, batch_B, _, labels in tqdm(dataloader, desc=desc):
            # Move to GPU
            for k in batch_A: 
                if isinstance(batch_A[k], torch.Tensor): batch_A[k] = batch_A[k].to(device)
            for k in batch_B:
                if isinstance(batch_B[k], torch.Tensor): batch_B[k] = batch_B[k].to(device)
            
            # Encode
            emb_A = model.encoder({'gf_v2_data': batch_A})
            emb_B = model.encoder({'gf_v2_data': batch_B})
            
            list_emb_A.append(emb_A.cpu().numpy())
            list_emb_B.append(emb_B.cpu().numpy())
            list_labels.append(labels.numpy())
            
    # Concatenate
    embs_A = np.vstack(list_emb_A)
    embs_B = np.vstack(list_emb_B)
    labels = np.concatenate(list_labels)
    
    return embs_A, embs_B, labels

def run_quality_check(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Embedding Quality Check ---")
    
    # 1. Load Data
    print("Initializing Datasets...")
    # Train is needed to train the k-NN probe
    train_dataset = BinaryClassificationDataset(args.train_pairs_path, args.spec_data_path, args.mol_data_path)
    spec_meta_dim = train_dataset[0][2].shape[1]
    
    # IMPORTANT: Use small batch size if memory is tight, shuffle doesn't matter here
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=binary_collate_fn, num_workers=4
    )
    
    test_dataset = BinaryClassificationDataset(args.test_pairs_path, args.spec_data_path, args.mol_data_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, num_workers=4
    )

    # 2. Load Model
    print("Loading Model...")
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, 
        spec_meta_dim=spec_meta_dim
    ).to(device)

    # Load Binary Weights (The Finetuned Encoder)
    if os.path.exists(args.binary_model_path):
        print(f"Loading weights: {args.binary_model_path}")
        state_dict = torch.load(args.binary_model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("WARNING: Using random weights!")
    
    # 3. Extract Embeddings
    print("\n--- extracting Train Features ---")
    # To save time/memory for k-NN, you might want to use a subset of training data
    # But let's try full set first.
    tr_A, tr_B, tr_y = extract_pair_features(model, train_loader, device, desc="Train")
    
    print("\n--- extracting Test Features ---")
    te_A, te_B, te_y = extract_pair_features(model, test_loader, device, desc="Test")
    
    print(f"Train Shapes: {tr_A.shape}")
    print(f"Test Shapes: {te_A.shape}")

    # --- TEST 1: The Raw Cosine Check (The "Structure" Test) ---
    print("\n[TEST 1] Raw Latent Cosine Similarity Analysis")
    # Calculate Cosine Distance (0=Same, 1=Diff) between A and B
    # Note: paired_cosine_distances returns distance. Similarity = 1 - Dist
    tr_cos_dist = paired_cosine_distances(tr_A, tr_B)
    te_cos_dist = paired_cosine_distances(te_A, te_B)
    
    tr_cos_sim = 1.0 - tr_cos_dist
    te_cos_sim = 1.0 - te_cos_dist
    
    # Evaluate how well Raw Cosine predicts the Label
    roc_train = roc_auc_score(tr_y, tr_cos_sim)
    roc_test = roc_auc_score(te_y, te_cos_sim)
    
    print(f"Train ROC-AUC (Raw Cosine): {roc_train:.4f}")
    print(f"Test ROC-AUC (Raw Cosine):  {roc_test:.4f}")
    
    print("--> Interpretation: If this is High (>0.8), your Encoder is excellent.")
    print("--> Interpretation: If this is Low, your Encoder hasn't learned to cluster pairs.")

    # --- TEST 2: The k-NN Probe (The "Capacity" Test) ---
    print("\n[TEST 2] k-NN Probe on Difference Vectors")
    # Feature Engineering: Absolute Difference |A - B|
    # This captures the magnitude of difference in every dimension
    X_train = np.abs(tr_A - tr_B)
    X_test = np.abs(te_A - te_B)
    
    print("Training k-NN Classifier (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, tr_y)
    
    print("Predicting Test Set...")
    knn_preds = knn.predict(X_test)
    
    acc_knn = accuracy_score(te_y, knn_preds)
    f1_knn = f1_score(te_y, knn_preds)
    
    print(f"k-NN Probe Accuracy: {acc_knn:.4f}")
    print(f"k-NN Probe F1 Score: {f1_knn:.4f}")
    
    # --- Comparison / Conclusion ---
    print("\n" + "="*30)
    print("DIAGNOSTIC REPORT")
    print("="*30)
    print(f"1. Encoder Intrinsic Quality (Raw Cosine ROC): {roc_test:.4f}")
    print(f"2. Max Information Extractable (k-NN Probe Acc): {acc_knn:.4f}")
    
    print("\nCONCLUSION:")
    if roc_test > 0.8:
        print("PASS: The Encoder has successfully organized the latent space.")
        print("      High similarity pairs have high cosine similarity.")
    else:
        print("FAIL: The Encoder latent space is messy (Low Cosine ROC).")
        print("      It may need harder negative mining or better pre-training.")
        
    if acc_knn > (roc_test + 0.1): # Heuristic buffer
        print("NOTE: k-NN performed significantly better than Raw Cosine.")
        print("      This implies the info is there, but it's non-linear.")
        print("      Your Classification Head (MLP) might need to be deeper/stronger.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_pairs_path", type=str, required=True)
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    # Config
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--binary_model_path", type=str, required=True)
    # Sys
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    run_quality_check(args)