import torch
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score
from pathlib import Path

# --- 1. SETUP SYS.PATH ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'bilinear_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'bilinear_model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

try:
    # UPDATED IMPORTS for Bilinear Architecture
    from bilinear_siamese_model import BilinearSiameseModel
    from binary_data_loader import DecoupledBinaryDataset, decoupled_collate_fn
    
    def merge_configs(base, custom):
        import copy
        merged = copy.deepcopy(base)
        for k, v in custom.items():
            if isinstance(v, dict) and k in merged:
                merged[k] = merge_configs(merged[k], v)
            else:
                merged[k] = v
        return merged
        
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_bilinear(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    input_filename = Path(args.test_pairs_path).stem 
    output_filename = f"results_bilinear_{input_filename}.csv"
    save_path = os.path.join(args.output_dir, output_filename)
    
    print(f"--- Starting Bilinear Model Testing ---")
    print(f"Target Dataset: {input_filename}")
    print(f"Output will be saved to: {save_path}")

    # --- 2. Load Data (Decoupled Loader) ---
    print(f"Loading Dataset and Metadata...")
    
    # A. Load the PyTorch Dataset
    test_dataset = DecoupledBinaryDataset(
        args.test_pairs_path, 
        args.spec_data_path, 
        args.mol_data_path
    )
    
    # B. Load Raw Metadata (for ID mapping in results)
    df_meta = pd.read_feather(args.test_pairs_path)
    if len(test_dataset) != len(df_meta):
        print(f"Trimming metadata to match valid dataset entries...")
        df_meta = df_meta.iloc[:len(test_dataset)]

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # Must be False to align with df_meta
        collate_fn=decoupled_collate_fn, 
        num_workers=4
    )
    
    # Check dimensions
    spec_meta_dim = test_dataset[0][2].shape[1]
    print(f"Dataset size: {len(test_dataset)} pairs | Meta Dim: {spec_meta_dim}")

    # --- 3. Initialize Model ---
    print("Initializing Bilinear Model...")
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = BilinearSiameseModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim
    ).to(device)

    # --- 4. Load Weights ---
    print(f"Loading Trained Weights: {os.path.basename(args.binary_model_path)}")
    state_dict = torch.load(args.binary_model_path, map_location=device)
    
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    elif 'best_model_sd' in state_dict: state_dict = state_dict['best_model_sd']
        
    model.load_state_dict(state_dict)
    model.eval()

    # --- 5. Inference Loop ---
    print("Running Inference...")
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # UPDATED: Unpack 5 items
        for batch_A, batch_B, batch_meta, batch_md, labels in tqdm(test_loader, desc="Testing"):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            batch_md = batch_md.to(device)
            
            # UPDATED: Forward pass with separated Mass Diff
            logits = model(batch_A, batch_B, batch_meta, batch_md).view(-1)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 6. Metrics ---
    print("\n--- Performance Metrics ---")
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    
    try: roc = roc_auc_score(all_labels, all_probs)
    except: roc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"                Pred Neg    Pred Pos")
    print(f"True Neg (Low)   {cm[0][0]:<10}  {cm[0][1]:<10}")
    print(f"True Pos (High)  {cm[1][0]:<10}  {cm[1][1]:<10}")

    # --- 7. Save Results ---
    print(f"\nSaving results to CSV...")
    results_df = pd.DataFrame()
    results_df['name_main'] = df_meta['name_main'].values
    results_df['name_sub'] = df_meta['name_sub'].values
    if 'cosine_similarity' in df_meta.columns:
        results_df['cosine_similarity'] = df_meta['cosine_similarity'].values
    
    results_df['true_label'] = all_labels
    results_df['prob_similarity'] = all_probs
    results_df['predicted_label'] = all_preds
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"Done. Saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--binary_model_path", type=str, required=True, help="Path to trained bilinear model .pth")
    parser.add_argument("--output_dir", type=str, default="./results/bilinear_predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    test_bilinear(args)