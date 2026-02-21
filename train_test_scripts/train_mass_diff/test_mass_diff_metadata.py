import torch
import yaml
import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'mass_gate'))
sys.path.insert(0, os.path.join(cwd, 'model', 'mass_gate'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

from classifier_siamesemodel_md_metadata import SiameseSpectralSimilarityModel
from binary_data_loader_metadata import BinaryClassificationDataset, binary_collate_fn

def merge_configs(base, custom):
    import copy
    merged = copy.deepcopy(base)
    for k, v in custom.items():
        if isinstance(v, dict) and k in merged:
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged

def test_binary(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Testing: Mass Diff as Feature ---")

    test_ds = BinaryClassificationDataset(args.test_pairs_path, args.spec_data_path, args.mol_data_path)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=binary_collate_fn)
    
    spec_meta_dim = test_ds[0][2].shape[1]
    
    with open(args.template_config_path, 'r') as f: template_config = yaml.safe_load(f)
    with open(args.custom_config_path, 'r') as f: custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)

    model = SiameseSpectralSimilarityModel(
        model_config=full_config.get('model', {}),
        checkpoint_path=args.checkpoint_path,
        spec_meta_dim=spec_meta_dim
    ).to(device)

    model.load_state_dict(torch.load(args.binary_model_path, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch_A, batch_B, batch_meta, labels in tqdm(test_loader):
            for k in batch_A: batch_A[k] = batch_A[k].to(device)
            for k in batch_B: batch_B[k] = batch_B[k].to(device)
            batch_meta = batch_meta.to(device)
            
            logits = model(batch_A, batch_B, batch_meta)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save results
    res_df = pd.DataFrame({'label': all_labels, 'prob': all_probs})
    os.makedirs(args.output_dir, exist_ok=True)
    res_df.to_csv(os.path.join(args.output_dir, "test_results.csv"), index=False)
    print("Results saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pairs_path", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--template_config_path", type=str, required=True)
    parser.add_argument("--custom_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--binary_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results_md_test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    test_binary(args)