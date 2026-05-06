import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# Mute PyTorch Nested Tensor warnings
warnings.filterwarnings("ignore", message=".*nested tensors.*")

# --- SETUP PATHS ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'fiar')) 
sys.path.insert(0, os.path.join(cwd, 'model', 'fiar'))
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model', 'bifurcate'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007', 'massformer', 'src'))
sys.path.insert(0, os.path.join(str(cwd), 'model', 'flare'))

try:
    from phase2_5_dataloader import PairedSiameseDataset, siamese_collate_fn
    # [UPDATED] Import the Phase 2.5 architecture
    from phase2_5_model import Phase2_5_SiameseNetwork
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    sys.exit(1)

def load_and_merge_configs(template_path="template.yml", custom_path="demo.yml"):
    with open(template_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    if os.path.exists(custom_path):
        with open(custom_path, 'r', encoding='utf-8') as f: custom_config = yaml.safe_load(f)
        for section, subdict in custom_config.items():
            if isinstance(subdict, dict):
                if section not in config: config[section] = {}
                for k, v in subdict.items(): config[section][k] = v
            else: config[section] = subdict
    return config

def test_binary_classifier(args):
    # Device Assignment Logic
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        print(f"[*] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("[-] WARNING: CUDA is not available. Falling back to CPU.")

    os.makedirs(args.output_dir, exist_ok=True)
    full_config = load_and_merge_configs(args.template_config, args.custom_config)

    print(f"\n[*] Initializing Test Dataset from {args.test_pairs}...")
    test_dataset = PairedSiameseDataset(feather_path=args.test_pairs, graphs_path=args.graphs_path)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, 
        shuffle=False, collate_fn=siamese_collate_fn, num_workers=6, pin_memory=True
    )

    print("\n[*] Initializing Phase 2.5 Siamese Network (Frozen Backbone)...")
    if not os.path.exists(args.phase2_checkpoint):
        raise FileNotFoundError(f"[-] Backbone Checkpoint not found at {args.phase2_checkpoint}")
        
    # [UPDATED] Initialize Phase 2.5 Model
    model = Phase2_5_SiameseNetwork(full_config, args.phase2_checkpoint, device, max_fragments=10).to(device)
    
    print(f"[*] Loading Trained Weights from: {args.model_ckpt}")
    if not os.path.exists(args.model_ckpt):
        raise FileNotFoundError(f"[-] Model Checkpoint not found at {args.model_ckpt}")
        
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device), strict=False)
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    
    print("\n[*] Commencing Evaluation...")
    test_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch_idx, (batch_A, batch_B, targets) in enumerate(test_bar):
            if batch_A is None or batch_B is None: continue
            
            batch_A = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_A.items()}
            batch_B = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_B.items()}

            # tau=0.01 matches the hardened temperature from training
            pred_sims = model(batch_A, batch_B, tau=0.01)
            all_preds.extend(pred_sims.cpu().numpy().flatten())

    prob_similarity = np.array(all_preds)
    
    # Apply the 0.70 Binary Classification Threshold
    THRESHOLD = 0.67
    predicted_label = (prob_similarity >= THRESHOLD).astype(float)

    # 4. Load Original Metadata to match true labels and build CSV
    print(f"\n[*] Loading original pairs data to map True Labels...")
    if str(args.test_pairs).endswith('.feather'):
        pairs_df = pd.read_feather(args.test_pairs)
    else:
        pairs_df = pd.read_csv(args.test_pairs)

    if len(pairs_df) != len(prob_similarity):
        raise ValueError(f"Row count mismatch! pairs_df: {len(pairs_df)}, predictions: {len(prob_similarity)}.")

    # Extract True Label (Entropy Label)
    if 'entropy_label' in pairs_df.columns:
        true_label = pairs_df['entropy_label'].values.astype(float)
    elif 'label' in pairs_df.columns:
        true_label = pairs_df['label'].values.astype(float)
    else:
        raise KeyError("Could not find 'entropy_label' or 'label' in the dataset.")

    # Calculate Classification Metrics
    acc = accuracy_score(true_label, predicted_label)
    prec = precision_score(true_label, predicted_label, zero_division=0)
    rec = recall_score(true_label, predicted_label, zero_division=0)
    f1 = f1_score(true_label, predicted_label, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(true_label, prob_similarity)
    except ValueError:
        roc_auc = float('nan') # In case of only one class present in test set

    print("\n" + "="*45)
    print(" 🏆 BINARY CLASSIFICATION METRICS (Threshold: 0.7) 🏆")
    print("="*45)
    print(f" Accuracy        : {acc:.4f}")
    print(f" Precision       : {prec:.4f}")
    print(f" Recall          : {rec:.4f}")
    print(f" F1-Score        : {f1:.4f}")
    print(f" ROC-AUC Score   : {roc_auc:.4f}")
    print("="*45)

    print(f"[*] Compiling requested CSV format...")
    
    # Create the DataFrame strictly following the requested format
    final_df = pd.DataFrame({
        'name_main': pairs_df['name_main'],
        'name_sub': pairs_df['name_sub'],
        'cosine_similarity': pairs_df.get('cosine_similarity', np.nan),
        'true_label': true_label,
        'prob_similarity': prob_similarity,
        'predicted_label': predicted_label
    })
    
    results_path = os.path.join(args.output_dir, "test_binary_predictions_phase2_5.csv")
    final_df.to_csv(results_path, index=False)
    print(f"[+] Formatted predictions successfully saved to: {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs", type=str, required=True)
    parser.add_argument("--graphs_path", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, required=True, help="Path to deterministic_student_best.pt")
    # [UPDATED] Help string points to Phase 2.5 weights
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to desaf_phase2_5_entropy_best.pt")
    parser.add_argument("--output_dir", type=str, default="results/binary_evaluation")
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    test_binary_classifier(args)