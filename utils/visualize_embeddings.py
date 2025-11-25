import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yaml
import argparse
import pickle
import copy
from tqdm import tqdm
from pathlib import Path
import os
import sys

# --- 1. SETUP SYS.PATH (COPIED FROM YOUR SCRIPTS) ---
cwd = Path.cwd()
print(f"Current working directory: {cwd}")
data_loader_dir = os.path.join(cwd, 'data_loaders')
print(f"Adding {data_loader_dir} to sys.path")
sys.path.insert(0, data_loader_dir)
train_dir = os.path.join(cwd, 'train_test_scripts')
print(f"Adding {train_dir} to sys.path")
sys.path.insert(0, train_dir)
model_dir = os.path.join(cwd, 'model')
print(f"Adding {model_dir} to sys.path")
sys.path.insert(0, model_dir)
parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
sys.path.insert(0, script_dir)

# --- 2. LOCAL IMPORTS (must come AFTER sys.path setup) ---
try:
    from updated_siamesemodel import MassFormerEncoder
    from gf_data_utils import gf_preprocess, collator
    from updated_train import merge_configs
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Import error: {e}")
    sys.exit(1)

import plotly.figure_factory as ff # Used for density plots

# --- 3. HELPER DATASET (Same as before) ---

class AllMoleculesDataset(Dataset):
    def __init__(self, mol_data_path):
        super().__init__()
        self.mol_df = pd.read_pickle(mol_data_path)
        print(f"Loaded {len(self.mol_df)} unique molecules from {mol_data_path}.")

    def __len__(self):
        return len(self.mol_df)

    def __getitem__(self, idx):
        mol = self.mol_df.iloc[idx]['mol']
        mol_id = self.mol_df.iloc[idx]['mol_id']
        graph_data = gf_preprocess(mol, idx)
        return graph_data, mol_id

def all_molecules_collate_fn(batch):
    graphs, mol_ids = zip(*batch)
    collated_graphs = collator(graphs)
    return collated_graphs, mol_ids

# --- 4. CORE FUNCTIONS (Same as before) ---

def load_massformer_encoder(template_config_path, custom_config_path, checkpoint_path):
    print("--- Loading Model Configuration ---")
    with open(template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)
    with open(custom_config_path, 'r') as f:
        custom_config = yaml.safe_load(f)
    full_config = merge_configs(template_config, custom_config)
    model_config = full_config.get('model', {})
    
    print("\n--- Initializing MassFormerEncoder ---")
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MassFormerEncoder(
        model_config=model_config,
        checkpoint_path=checkpoint_path
    )
    
    model.to(device)
    model.eval()
    return model, device

def get_all_embeddings(model, device, mol_data_path):
    print("\n--- Generating Embeddings for All Unique Molecules ---")
    
    dataset = AllMoleculesDataset(mol_data_path)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=all_molecules_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    embedding_lookup = {}
    with torch.no_grad():
        for batch_data, mol_ids in tqdm(loader, desc="Generating Embeddings"):
            for key in batch_data: 
                batch_data[key] = batch_data[key].to(device)
            
            wrapped_batch = {'gf_v2_data': batch_data}
            embeddings = model(wrapped_batch)
            
            for mol_id, emb in zip(mol_ids, embeddings.cpu()):
                embedding_lookup[mol_id] = emb

    print(f"Successfully generated {len(embedding_lookup)} unique embeddings.")
    return embedding_lookup

# --- 5. NEW MAIN FUNCTION ---

def main(args):
    # --- 1. Load Model and Generate All Embeddings ---
    model, device = load_massformer_encoder(
        args.template_config_path, 
        args.custom_config_path, 
        args.checkpoint_path
    )
    
    all_embeddings_dict = get_all_embeddings(model, device, args.mol_data_path)

    # --- 2. Load Pair Data and Calculate Cosine Distances ---
    print("\n--- Processing Pairs and Calculating **Cosine Distances** ---")
    
    print(f"Loading pairs from {args.pairs_path}...")
    pair_df = pd.read_feather(args.pairs_path)

    if args.subset_size < 1.0:
        n_samples = int(len(pair_df) * args.subset_size)
        print(f"Using a subset of {n_samples} pairs ({args.subset_size * 100}%)")
        pair_df = pair_df.sample(n=n_samples, random_state=42)

    print(f"Loading spectrum lookup from {args.spec_data_path}...")
    spec_df = pd.read_pickle(args.spec_data_path)
    spec_to_mol_id_map = spec_df.set_index('spec_id')['mol_id'].to_dict()

    # --- !!! MODIFIED SCRIPT LOGIC !!! ---
    # Store the distances and ground truths
    distances = []
    ground_truths = []
    
    for row in tqdm(pair_df.itertuples(), total=len(pair_df), desc="Calculating Distances"):
        spec_id_a = row.name_main
        spec_id_b = row.name_sub
        
        mol_id_a = spec_to_mol_id_map.get(spec_id_a)
        mol_id_b = spec_to_mol_id_map.get(spec_id_b)
        
        if mol_id_a and mol_id_b:
            emb_a = all_embeddings_dict.get(mol_id_a)
            emb_b = all_embeddings_dict.get(mol_id_b)
            
            if emb_a is not None and emb_b is not None:
                # 1. Get Ground Truth Similarity
                sim_gt = row.cosine_similarity
                
                # 2. Get Embedding Cosine Similarity
                sim_emb = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
                
                # 3. Calculate Cosine Distance
                dist_emb = 1.0 - sim_emb
                
                ground_truths.append(sim_gt)
                distances.append(dist_emb)

    print(f"Successfully processed {len(distances)} pairs.")

    # --- 3. Create DataFrame and Bins ---
    print("\n--- Binning Data ---")
    
    plot_df = pd.DataFrame({
        'Ground_Truth_Similarity': ground_truths,
        'Embedding_Cosine_Distance': distances
    })
    
    # Define bins based on your hypothesis
    bins = [0.0, 0.4, 0.8, 1.0]
    labels = ['Low (0.0-0.4)', 'Medium (0.4-0.8)', 'High (0.8-1.0)']
    
    plot_df['Similarity_Bin'] = pd.cut(
        plot_df['Ground_Truth_Similarity'], 
        bins=bins, 
        labels=labels, 
        right=True,
        include_lowest=True
    ).dropna()

    # --- 4. Interactive Density Plot (Ridge Plot) ---
    print("\n--- Creating Interactive Density Plot ---")
    
    # Prepare data for the figure_factory
    hist_data = [
        plot_df[plot_df['Similarity_Bin'] == 'High (0.8-1.0)']['Embedding_Cosine_Distance'],
        plot_df[plot_df['Similarity_Bin'] == 'Medium (0.4-0.8)']['Embedding_Cosine_Distance'],
        plot_df[plot_df['Similarity_Bin'] == 'Low (0.0-0.4)']['Embedding_Cosine_Distance']
    ]
    
    group_labels = ['High (0.8-1.0)', 'Medium (0.4-0.8)', 'Low (0.0-0.4)']
    
    fig = ff.create_distplot(
        hist_data, 
        group_labels,
        show_hist=False,
        show_rug=False
    )
    
    fig.update_layout(
        title='Distribution of Embedding Cosine Distances by Ground Truth Similarity',
        xaxis_title='Embedding Cosine Distance (1.0 - Cosine Similarity)',
        yaxis_title='Density',
        legend_title='Similarity Bin',
        template='plotly_dark'
    )
    
    output_filename = 'embedding_cosine_distance_distribution.html'
    fig.write_html(output_filename)
    
    print(f"\n--- All Done! ---")
    print(f"Interactive plot saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot embedding cosine distance distributions by similarity.")
    
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to the .feather file of pairs (e.g., ...train.feather)")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to the spectrum lookup .pkl file (spec_df_2.pkl)")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to the molecule graph .pkl file (mol_df_2.pkl)")

    parser.add_argument("--template_config_path", type=str, required=True, help="Path to the template.yml config file.")
    parser.add_argument("--custom_config_path", type=str, required=True, help="Path to the custom experiment config (e.g., demo_eval.yml).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the base MassFormer .pkl checkpoint.")
    
    parser.add_argument("--subset_size", type=float, default=0.2, help="Fraction of pairs to plot (e.g., 0.2 for 20%)")

    args = parser.parse_args()
    main(args)