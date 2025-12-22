import torch
import torch.nn as nn
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_squared_error
from scipy.stats import pearsonr

# --- 1. SETUP SYS.PATH ---
from pathlib import Path
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders'))
sys.path.insert(0, os.path.join(cwd, 'model'))
sys.path.insert(0, os.path.join(cwd, 'train_test_scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    from multitrain_siamesemodel import SiameseSpectralSimilarityModel
    from updated_train import merge_configs
    from multitrain_data_loader import MulticlassDataset, multiclass_collate_fn
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_multitask(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Multi-Task Model Testing ---")
    
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
        num_classes=3 
    ).to(device)

    # --- 4. Load Trained Weights ---
    print(f"Loading trained weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 5. Inference Loop ---
    print("Running Inference...")
    
    # Storage for Classification (Task A)
    pred_classes = []
    true_classes = []
    class_probs = []
    
    # Storage for Regression (Task B)
    pred_tanimoto = []
    true_tanimoto = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Unpack 5 items
            b_A, b_B, b_meta, b_labels, b_tanimotos = batch
            
            # Move to device
            for k in b_A: b_A[k] = b_A[k].to(device)
            for k in b_B: b_B[k] = b_B[k].to(device)
            b_meta = b_meta.to(device)
            
            # Forward Pass (Returns Tuple)
            spec_logits, struct_preds = model(b_A, b_B, b_meta)
            
            # --- Task A Process ---
            probs = torch.softmax(spec_logits, dim=1)
            preds = torch.argmax(spec_logits, dim=1)
            
            pred_classes.extend(preds.cpu().numpy())
            true_classes.extend(b_labels.numpy())
            class_probs.extend(probs.cpu().numpy())
            
            # --- Task B Process ---
            pred_tanimoto.extend(struct_preds.squeeze().cpu().numpy())
            true_tanimoto.extend(b_tanimotos.numpy())

    # --- 6. Calculate Metrics ---
    print("\n" + "="*40)
    print("TASK A: SPECTRAL CLASSIFICATION RESULTS")
    print("="*40)
    
    acc = accuracy_score(true_classes, pred_classes)
    macro_f1 = f1_score(true_classes, pred_classes, average='macro')
    weighted_f1 = f1_score(true_classes, pred_classes, average='weighted')

    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(true_classes, pred_classes, target_names=['Low (<0.65)', 'Med (0.65-0.85)', 'High (>=0.85)']))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(true_classes, pred_classes))

    print("\n" + "="*40)
    print("TASK B: STRUCTURAL REGRESSION RESULTS")
    print("="*40)
    
    mse = mean_squared_error(true_tanimoto, pred_tanimoto)
    rmse = np.sqrt(mse)
    pearson_r, _ = pearsonr(true_tanimoto, pred_tanimoto)
    
    print(f"MSE:         {mse:.4f}")
    print(f"RMSE:        {rmse:.4f}")
    print(f"Pearson r:   {pearson_r:.4f}")

    # --- 7. Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "multitask_test_results.csv")
    
    df_results = pd.DataFrame({
        'true_spectral_class': true_classes,
        'pred_spectral_class': pred_classes,
        'true_tanimoto': true_tanimoto,
        'pred_tanimoto': pred_tanimoto
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
    parser.add_argument("--model_path", type=str, required=True, help="Path to 'best_multitask_model.pth'")
    parser.add_argument("--output_dir", type=str, default="./multitask_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    test_multitask(args)