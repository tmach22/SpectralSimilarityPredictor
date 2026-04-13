import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import yaml
import io
import pandas as pd
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# --- SETUP PATHS ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'flare'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))

try:
    from data_loader import CrossModalPretrainDataset, cross_modal_collate_fn
    from cross_modal_pretrainer_new import CrossModalFlarePretrainer
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

def visualize_motifs(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    full_config = load_and_merge_configs(args.template_config, args.custom_config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[*] Initializing Dataset for Visualization...")
    dataset = CrossModalPretrainDataset(args.spec_data_path, args.mol_data_path)
    
    # ====================================================================
    # EXACT MOLECULE EXTRACTION BY SPEC_ID
    # ====================================================================
    if not args.target_spec_id:
        print("[!] Error: You must provide a --target_spec_id")
        sys.exit(1)
        
    print(f"[*] Searching for spec_id: {args.target_spec_id}...")
    
    target_idx = None
    for i in range(len(dataset.spec_df)):
        if dataset.spec_df.iloc[i]['spec_id'] == args.target_spec_id:
            target_idx = i
            break
            
    if target_idx is None:
        print(f"[!] Error: Could not find spec_id {args.target_spec_id} in the dataset.")
        sys.exit(1)
        
    print(f"[+] Found {args.target_spec_id} at dataset index {target_idx}")
    
    # Extract the single item from the dataset
    item = dataset[target_idx]
    if item is None or item[0] is None or item[1] is None:
        print("[!] Error: The dataset returned invalid data for this item (missing graph or brics).")
        sys.exit(1)
        
    single_graph, single_brics, single_peaks, single_mask = item
    
    row = dataset.spec_df.iloc[target_idx]
    target_mol = dataset.mol_lookup.get('mol').get(row['mol_id'])
    true_num_atoms = target_mol.GetNumAtoms()
    print(f"[*] Molecule loaded. True atoms: {true_num_atoms}")
    
    # We must collate it into a batch of size 1 so the model can process it
    print("[*] Collating into Batch of Size 1...")
    batch = cross_modal_collate_fn([item])
    batched_graphs, A_brics, _, _ = batch
    
    # ====================================================================
    # MODEL INFERENCE
    # ====================================================================
    print(f"[*] Loading Trained Stage 1 Weights from: {args.stage1_ckpt}")
    model = CrossModalFlarePretrainer(
        model_config=full_config['model'], 
        num_motifs=15
    ).to(device)
    
    model.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))
    model.eval()

    if isinstance(batched_graphs, dict):
        batched_graphs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batched_graphs.items()}
    
    A_brics = A_brics.to(device)
    
    with torch.no_grad():
        Z_graph, S, _ = model.forward_graph(batched_graphs, A_brics, tau=0.5)
        
    # Because batch size is 1, S has shape [1, N_atoms, Motifs]
    S_np = S.squeeze(0).cpu().numpy()            
    
    for atom in target_mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    # ==========================================
    # COMBINED PLOT: Heatmap + Color-Coded Structure
    # ==========================================
    print(f"[*] Generating Combined Heatmap and Structure Plot...")
    
    # 1. Prepare Motif Hard Assignments and Colors
    # Slice off the padding. Assuming dim 0 is the virtual node if used in your architecture.
    real_atom_S = S_np[1:true_num_atoms+1, :] 
    atom_assignments = np.argmax(real_atom_S, axis=1)
    print(f"[*] Atom-to-Motif Assignments: {atom_assignments}")
    
    cmap_colors = plt.get_cmap('tab20').colors
    highlight_atoms = []
    highlight_colors = {}
    
    for atom_idx, motif_bin in enumerate(atom_assignments):
        highlight_atoms.append(atom_idx)
        color_tuple = cmap_colors[motif_bin % 20] # Safe modulo in case motifs > 20
        highlight_colors[atom_idx] = tuple(color_tuple[:3])

    # 2. Render RDKit Molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette() 
    
    mc = Chem.Draw.rdMolDraw2D.PrepareMolForDrawing(target_mol)
    drawer.DrawMolecule(
        mc, 
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=[] 
    )
    drawer.FinishDrawing()
    
    img_bytes = drawer.GetDrawingText()
    mol_img = Image.open(io.BytesIO(img_bytes))

    # 3. Setup Matplotlib 1x2 Subplot Grid
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.2]})
    
    sns.heatmap(real_atom_S, cmap="viridis", annot=False, ax=axes[0])
    axes[0].set_title(f"Assignment Probabilities (S) | {args.target_spec_id}", fontsize=14)
    axes[0].set_xlabel("Learned Motifs (0-14)", fontsize=12)
    axes[0].set_ylabel("Atom Index", fontsize=12)
    
    axes[1].imshow(mol_img)
    axes[1].axis('off') 
    axes[1].set_title(f"Structure Colored by Motif Assignment", fontsize=14)
    
    plt.tight_layout()
    combined_path = os.path.join(args.output_dir, f"{args.target_spec_id}_motif_analysis.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"[+] Plot saved to {combined_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--spec_data_path", type=str, required=True)
    parser.add_argument("--mol_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/motif_visualizations")
    
    # The new target argument
    parser.add_argument("--target_spec_id", type=str, required=True, help="The exact MassSpecGymID to visualize")
    
    parser.add_argument("--template_config", type=str, default="template.yml")
    parser.add_argument("--custom_config", type=str, default="demo.yml")
    
    args = parser.parse_args()
    visualize_motifs(args)