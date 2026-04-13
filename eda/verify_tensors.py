import torch

# Update this path if you are pointing to the full dataset now
data_path = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/phase1_graphs.pt"

print(f"[*] Loading dataset from {data_path}...")
dataset = torch.load(data_path)

print(f"\n[+] Total Graphs Loaded: {len(dataset)}")

total_edges = 0
total_cleavages = 0
bde_min = float('inf')
bde_max = float('-inf')
unique_smiles = set()

# We want to print a deep dive for 3 DIFFERENT molecules
deep_dives_done = 0
target_deep_dives = 3

for i, data in enumerate(dataset):
    # PyG directed edges = 2 * undirected bonds
    num_directed_edges = data.edge_index.shape[1] 
    total_edges += num_directed_edges
    total_cleavages += data.y_cleavage.sum().item()
    
    # Track BDE bounds (Column 0 is BDE, Column 1 is CE)
    if num_directed_edges > 0:
        bdes = data.edge_attr[:, 0]
        bde_min = min(bde_min, bdes.min().item())
        bde_max = max(bde_max, bdes.max().item())
    
    smiles = getattr(data, 'smiles', 'Unknown')
    
    # Deep Dive Logic: Only print if it's a molecule we haven't dived into yet!
    if smiles not in unique_smiles and deep_dives_done < target_deep_dives:
        # Get the collision energy of the first edge (it's the same for the whole graph)
        ce = data.edge_attr[0, 1].item() if num_directed_edges > 0 else "N/A"
        
        print(f"\n--- Deep Dive: Graph {i} ---")
        print(f"SMILES: {smiles}")
        print(f"Collision Energy (CE): {ce} eV")
        print(f"Nodes (Atoms): {data.num_nodes}")
        print(f"Edges (Directed): {num_directed_edges}")
        print(f"Number of cut bonds: {int(data.y_cleavage.sum().item() / 2)}") # Divide by 2 for undirected
        
        deep_dives_done += 1
        
    unique_smiles.add(smiles)

print("\n=======================================")
print("--- Dataset Global Statistics ---")
print(f"Total Unique Molecules: {len(unique_smiles)}")
print(f"Total Graphs (Experiments): {len(dataset)}")
print(f"Total Directed Edges: {total_edges}")
if total_edges > 0:
    print(f"Global Cleavage Ratio: {(total_cleavages / total_edges) * 100:.2f}% of edges were cut.")
print(f"BDE Range: [{bde_min:.1f}, {bde_max:.1f}] kcal/mol")
print("=======================================")

# Sanity Check Warnings
if len(dataset) > 0 and torch.isnan(dataset[0].edge_attr).any(): 
    print("[!] WARNING: NaN values detected in Edge Attributes!")