import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- SETUP ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'wide_model'))
sys.path.insert(0, os.path.join(cwd, 'model', 'wide_model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from binary_data_loader import BinaryClassificationDataset, binary_collate_fn
from model_stage2 import StageTwoClassificationModel

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(custom_path, 'r', encoding='utf-8') as f:
        custom_config = yaml.safe_load(f)
        
    for section, subdict in custom_config.items():
        if isinstance(subdict, dict):
            if section not in config:
                config[section] = {}
            for k, v in subdict.items():
                config[section][k] = v
        else:
            config[section] = subdict
    return config

def test_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    # --- HARDWARE OPTIMIZATIONS ---
    num_workers = full_config.get('run', {}).get('num_workers', 6)
    use_pin_memory = full_config.get('run', {}).get('pin_memory', True)

    # 1. Init Data (Strictly NO upsampling and NO shuffling for test sets)
    print(f"Loading Test Dataset from {args.test_pairs_path}...")
    test_dataset = BinaryClassificationDataset(
        pairs_feather_path=args.test_pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        upsample_hard_positives=False 
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=binary_collate_fn, pin_memory=use_pin_memory, num_workers=num_workers
    )

    # Load the exact test DataFrame to grab the specific string IDs later
    df_test = pd.read_feather(args.test_pairs_path)

    # 2. Init Model
    model = StageTwoClassificationModel(
        model_config=full_config['model'], 
        stage1_checkpoint=args.stage1_checkpoint
    ).to(device)
    
    print(f"Loading trained Stage 2 weights from {args.stage2_checkpoint}...")
    model.load_state_dict(torch.load(args.stage2_checkpoint, map_location=device))
    model.eval()

    # 3. Tracking Lists
    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []
    all_cos_sims = []

    # 4. Evaluation Loop
    print("\nStarting Evaluation...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating Test Set")
        
        for batch_A, batch_B, spec_meta, mass_A, mass_B, labels in test_bar:
            # Move to device asynchronously
            batch_A = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_B.items()}
            
            spec_meta = spec_meta.to(device, non_blocking=True)
            mass_A_dev = mass_A.to(device, non_blocking=True)
            mass_B_dev = mass_B.to(device, non_blocking=True)
            
            # Forward pass through the full Wide-Bottleneck MLP
            logits = model(batch_A, batch_B, spec_meta, mass_A_dev, mass_B_dev)
            
            # --- CALCULATE COSINE SIMILARITY ---
            # We explicitly pass the graphs through the frozen encoder to get the metrics
            emb_A = F.relu(model.encoder({'gf_v2_data': batch_A}))
            emb_B = F.relu(model.encoder({'gf_v2_data': batch_B}))
            cos_sims = F.cosine_similarity(emb_A, emb_B, dim=1).cpu().numpy()
            
            # Convert logits to probabilities using Sigmoid
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            
            # Convert probabilities to hard class labels (threshold = 0.5)
            preds = (probs >= 0.5).astype(float) # Cast to float for 1.0 / 0.0 formatting
            
            # Store results
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(probs)
            all_pred_labels.extend(preds)
            all_cos_sims.extend(cos_sims)

    # 5. Calculate Metrics
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_true_labels, all_pred_probs)
    except ValueError:
        roc_auc = float('nan')

    print("\n" + "="*50)
    print("FINAL TEST SET METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*50 + "\n")

    # 6. Save Predictions to EXACT CSV Format
    results_df = pd.DataFrame({
        'name_main': df_test['name_main'].values[:len(all_pred_probs)],
        'name_sub': df_test['name_sub'].values[:len(all_pred_probs)],
        'cosine_similarity': all_cos_sims,
        'true_label': [float(lbl) for lbl in all_true_labels],
        'prob_similarity': all_pred_probs,
        'predicted_label': all_pred_labels
    })
    
    csv_path = os.path.join(args.output_dir, "stage2_test_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved correctly formatted predictions to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="Base metric-learning weights")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Trained classification head weights")
    
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    parser.add_argument("--batch_size", type=int, default=128) 
    
    args = parser.parse_args()
    test_stage2(args)