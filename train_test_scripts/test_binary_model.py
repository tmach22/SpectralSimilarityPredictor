import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

# --- 1. SETUP SYS.PATH ---
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, "train_test_scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Import custom modules
try:
    from classifier_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Binary Model Testing ---")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    print(f"Loading test data from: {args.test_pairs_path}")
    test_dataset = BinaryClassificationDataset(
        args.test_pairs_path, 
        args.spec_data_path, 
        args.mol_data_path
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=binary_collate_fn, 
        num_workers=4
    )
    
    # Get metadata dim dynamically
    spec_meta_dim = test_dataset[0][2].shape[1]
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Initialize Model ---
    print("Initializing Model Architecture...")
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path, # This loads the base architecture
        spec_meta_dim=spec_meta_dim
    ).to(device)

    # --- 4. Load Trained Binary Weights ---
    print(f"Loading trained weights from: {args.binary_model_path}")
    state_dict = torch.load(args.binary_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 5. Inference Loop ---
    print("Running Inference...")
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_A, batch_B, batch_meta, labels in tqdm(test_loader, desc="Testing"):
            # Move to device
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            
            # Forward pass
            logits = model(batch_A, batch_B, batch_meta).squeeze()
            
            # Apply Sigmoid (since we removed it from the model class)
            probs = torch.sigmoid(logits)
            
            # Threshold at 0.5
            preds = (probs > 0.5).float()
            
            # Store
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 6. Calculate Metrics ---
    print("\n--- Performance Metrics ---")
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"                 Pred Neg    Pred Pos")
    print(f"True Neg (Low)   {cm[0][0]:<10}  {cm[0][1]:<10}")
    print(f"True Pos (High)  {cm[1][0]:<10}  {cm[1][1]:<10}")

    # --- 7. Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "test_results_v2.csv")
    
    df_results = pd.DataFrame({
        'prob_similarity': all_probs,
        'predicted_label': all_preds,
        'true_label': all_labels
    })
    df_results.to_csv(save_path, index=False)
    print(f"\nDetailed predictions saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data Params
    parser.add_argument("--test_pairs_path", type=str, required=True, help="Path to the binary test set feather file")
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Config Params
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Base MassFormer architecture checkpoint")
    
    # Model Params
    parser.add_argument("--binary_model_path", type=str, required=True, help="Path to your trained 'best_binary_model.pth'")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    test_binary(args)