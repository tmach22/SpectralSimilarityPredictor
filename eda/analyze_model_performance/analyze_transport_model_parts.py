import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import BRICS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import yaml
import os
import sys
from pathlib import Path

# --- 1. SETUP PATHS ---
cwd = Path.cwd()
parent_directory = os.path.dirname(cwd.parent)
sys.path.insert(0, os.path.join(cwd, 'data_loaders', 'flare'))
sys.path.insert(0, os.path.join(cwd, 'model', 'flare'))
sys.path.insert(0, os.path.join(parent_directory, 'tmach007/massformer/src/massformer'))

try:
    from data_loader import CrossModalPretrainDataset, cross_modal_collate_fn
    from cross_modal_pretrainer import CrossModalFlarePretrainer
except ImportError as e:
    print(f"[-] Error importing custom modules: {e}")
    sys.exit(1)

# Paths
SPEC_DATA_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"
MOL_DATA_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df.pkl"
STAGE1_CHECKPOINT = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/trained_model/flare/massformer_flare_stage1_best.pt"

def load_and_merge_configs(template_path="/data/nas-gpu/wang/tmach007/massformer/config/template.yml", custom_path="/data/nas-gpu/wang/tmach007/massformer/config/demo/demo_eval.yml"):
    with open(template_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    if os.path.exists(custom_path):
        with open(custom_path, 'r', encoding='utf-8') as f: custom_config = yaml.safe_load(f)
        for section, subdict in custom_config.items():
            if isinstance(subdict, dict):
                if section not in config: config[section] = {}
                for k, v in subdict.items(): config[section][k] = v
            else: config[section] = subdict
    return config

def get_brics_ground_truth(mol):
    """
    Simulates theoretical fragmentation by physically 'cutting' the BRICS bonds 
    and grouping the remaining connected atoms into Ground Truth Motifs.
    """
    brics_bonds = set()
    for bond in BRICS.FindBRICSBonds(mol):
        u, v = bond[0]
        brics_bonds.add(tuple(sorted((u, v))))

    # Build adjacency list ignoring cleavable bonds
    adj = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if tuple(sorted((u, v))) not in brics_bonds:
            adj[u].append(v)
            adj[v].append(u)

    # Find connected components (The rigid structural fragments)
    visited = set()
    fragments = []
    for i in range(mol.GetNumAtoms()):
        if i not in visited:
            comp = []
            q = [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            fragments.append(comp)

    # Map atom index to its Fragment ID
    atom_to_frag = np.zeros(mol.GetNumAtoms())
    for frag_id, comp in enumerate(fragments):
        for atom_idx in comp:
            atom_to_frag[atom_idx] = frag_id
            
    return atom_to_frag

def visualize_learned_motifs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    # 1. Initialize Dataset
    dataset = CrossModalPretrainDataset(spec_data_path=SPEC_DATA_PATH, mol_data_path=MOL_DATA_PATH, max_peaks=60)
    
    mol_idx = 16 # Feel free to change this index to test different molecules!
    row = dataset.spec_df.iloc[mol_idx]
    mol_id = row['mol_id']
    smiles = dataset.mol_lookup.loc[mol_id, 'smiles']
    mol = Chem.MolFromSmiles(smiles)
    print(f"[*] Extracting Motif Predictions for SMILES: {smiles}")

    subset = torch.utils.data.Subset(dataset, [mol_idx])
    dataloader = DataLoader(subset, batch_size=1, collate_fn=cross_modal_collate_fn)
    
    # 2. Initialize Model & Load Weights
    config = load_and_merge_configs()
    model = CrossModalFlarePretrainer(model_config=config['model'], num_motifs=15).to(device)
    
    print(f"[*] Loading Trained Motif Weights from: {STAGE1_CHECKPOINT}")
    model.load_state_dict(torch.load(STAGE1_CHECKPOINT, map_location=device))
    model.eval()
    
    # 3. Extract the Learned Motif Assignments (FLARE)
    batch_graphs, A_brics, peaks, peak_mask = next(iter(dataloader))
    if isinstance(batch_graphs, dict):
        batch_graphs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch_graphs.items()}
    elif hasattr(batch_graphs, 'to'):
        batch_graphs = batch_graphs.to(device)
        
    A_brics = A_brics.to(device) if A_brics is not None else None
    
    with torch.no_grad():
        Z_graph, S, X_pool = model.forward_graph(batch_graphs, A_brics, tau=0.5)
        
    # Shape: [Num_Atoms, 15]. We use PCA to compress this into a 1D array for coloring.
    atom_assignments = S[0, 1:, :].cpu().numpy() 
    pca = PCA(n_components=1)
    flare_colors_raw = pca.fit_transform(atom_assignments).flatten()
    
    # 4. Extract Theoretical Ground Truth (BRICS)
    brics_colors_raw = get_brics_ground_truth(mol)
    
    # 5. Normalization for Color Mapping
    def normalize_to_colors(values, cmap_name='Set1'):
        # We use a categorical colormap (Set1) because motifs are distinct chunks
        norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
        cmap = plt.get_cmap(cmap_name)
        colors = {i: cmap(norm(val)) for i, val in enumerate(values)}
        return colors

    colors_nn = normalize_to_colors(flare_colors_raw, 'viridis')
    colors_gt = normalize_to_colors(brics_colors_raw, 'Set1')

    # 6. Visualization Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    def draw_to_ax(m, colors, ax, title):
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(600, 600)
        drawer.DrawMolecule(m, highlightAtoms=list(colors.keys()), highlightAtomColors=colors)
        drawer.FinishDrawing()
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(drawer.GetDrawingText()))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    draw_to_ax(mol, colors_gt, axes[0], "Theoretical 'Ground Truth' Cleavage\n(BRICS Fragmentation Rules)")
    draw_to_ax(mol, colors_nn, axes[1], "FLARE Learned Latent Motifs\n(Trained via Mass Spec Alignment)")
    
    plt.tight_layout()
    plt.savefig("flare_vs_groundtruth.png", dpi=300)
    print("\n[+] Visualization saved as 'flare_vs_groundtruth.png'")

if __name__ == "__main__":
    visualize_learned_motifs()