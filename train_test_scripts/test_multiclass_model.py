import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- 1. SETUP SYS.PATH ---
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from multiclass_classifier_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from multiclass_data_loader import MulticlassDataset, multiclass_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_multiclass(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Multiclass Model Testing (3 Classes) ---")
    
    # --- 2. Load Data ---
    print(f"Loading test data from: {args.test_pairs_path}")
    test_dataset = MulticlassDataset(
        args.test_pairs_path, 
        args.spec_data_path, 
        args.mol_data_path
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=multiclass_collate_fn, 
        num_workers=4
    )
    
    spec_meta_dim = test_dataset[0][2].shape[1]
    print(f"Test dataset size: {len(test_dataset)}")

    # --- 3. Initialize Model ---
    print("Initializing Model Architecture...")
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim,
        num_classes=3 # <--- 3 CLASSES
    ).to(device)

    # --- 4. Load Trained Weights ---
    print(f"Loading trained weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 5. Inference Loop ---
    print("Running Inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_A, batch_B, batch_meta, labels in tqdm(test_loader, desc="Testing"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            
            # Forward pass
            logits = model(batch_A, batch_B, batch_meta)
            
            # Get Predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    # --- 6. Calculate Metrics ---
    print("\n--- Performance Metrics ---")
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Low (<0.65)', 'Med (0.65-0.85)', 'High (>=0.85)']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # --- 7. Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "multiclass_test_results.csv")
    
    # Convert probs to list for saving
    probs_list = [p.tolist() for p in all_probs]
    
    df_results = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'probs': probs_list
    })
    df_results.to_csv(save_path, index=False)
    print(f"\nDetailed predictions saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data Params
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    # Config Params
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    
    # Model Params
    parser.add_argument("--model_path", type=str, required=True, help="Path to 'best_multiclass_model_3class.pth'")
    parser.add_argument("--output_dir", type=str, default="./multiclass_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    test_multiclass(args)