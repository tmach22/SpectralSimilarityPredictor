import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
import os

# [UPDATED] Import Phase2EdgeDataset along with the collate function
try:
    from phase2_dataloader import phase2_collate_fn, Phase2EdgeDataset
except ImportError:
    print("[-] Warning: Could not import phase2_collate_fn or Phase2EdgeDataset.")
    sys.exit(1)

class PairedSiameseDataset(Dataset):
    def __init__(self, feather_path, graphs_path):
        print(f"[*] Loading paired metadata from: {feather_path}")
        self.pairs_df = pd.read_feather(feather_path)
        
        print(f"[*] Initializing Base Phase2EdgeDataset from: {graphs_path}")
        # [THE FIX] We let your existing dataset handle the heavy lifting!
        self.base_dataset = Phase2EdgeDataset(processed_graphs_path=graphs_path)
        
        print("[*] Building spec_id -> Base Index lookup dictionary...")
        self.spec_to_idx = {}
        
        # We map the spec_id to the integer index of the base dataset
        for i, g in enumerate(self.base_dataset.graphs):
            key = getattr(g, 'spec_id', None)
            if key is not None:
                self.spec_to_idx[key] = i
                
        print(f"[+] Successfully mapped {len(self.spec_to_idx)} graphs.")

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        spec_A = row['name_main']
        spec_B = row['name_sub']
        
        # =================================================================
        # [CRITICAL UPDATE] Target the new Spectral Entropy Similarity column
        # =================================================================
        target_sim = torch.tensor(row['entropy_similarity'], dtype=torch.float32)
        
        # Look up the integer indices for the base dataset
        idx_A = self.spec_to_idx.get(spec_A)
        idx_B = self.spec_to_idx.get(spec_B)
        
        if idx_A is None or idx_B is None:
            return None 

        # =================================================================
        # [THE FIX] Delegate the fetch to the base dataset!
        # This automatically computes 'attn_bias', 'spatial_pos', etc.
        # =================================================================
        graph_A = self.base_dataset[idx_A].clone()
        graph_B = self.base_dataset[idx_B].clone()
        
        # Inject the siamese batch index
        graph_A.idx = idx
        graph_B.idx = idx
        # =================================================================

        return graph_A, graph_B, target_sim

def siamese_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
        
    graphs_A = [item[0] for item in batch]
    graphs_B = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch])
    
    batch_A = phase2_collate_fn(graphs_A)
    batch_B = phase2_collate_fn(graphs_B)
    
    return batch_A, batch_B, targets


# ==============================================================================
# ISOLATED TESTING BLOCK
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the Siamese Dataloader")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to stratified_binary_07_dataset_train.feather")
    parser.add_argument("--graphs_path", type=str, required=True, help="Path to phase3_graphs.pt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    args = parser.parse_args()

    if not os.path.exists(args.pairs_path):
        print(f"[!] Error: Pairs file not found at {args.pairs_path}")
        sys.exit(1)
    if not os.path.exists(args.graphs_path):
        print(f"[!] Error: Graphs file not found at {args.graphs_path}")
        sys.exit(1)

    print("\n--- Initiating Dataloader Test ---")
    
    test_dataset = PairedSiameseDataset(feather_path=args.pairs_path, graphs_path=args.graphs_path)
    print(f"\n[+] Dataset initialized successfully with {len(test_dataset)} pairs.")

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=siamese_collate_fn
    )

    print("\n--- Fetching a Test Batch ---")
    try:
        batch_iterator = iter(test_loader)
        batch_A, batch_B, targets = next(batch_iterator)
        
        while batch_A is None:
            print("[!] Batch was empty (missing graphs), fetching next...")
            batch_A, batch_B, targets = next(batch_iterator)

        print("[+] Successfully fetched batch!")
        print("\n--- Batch Structure ---")
        # [UPDATED] Log output correctly reflects Entropy
        print(f"Targets (Entropy Similarities): {targets.shape}")
        
        print("\nBatch A (Molecule 1):")
        if isinstance(batch_A, dict):
            print(f"  - Node Features 'x': {batch_A.get('x', torch.tensor([])).shape}")
            if 'attn_bias' in batch_A:
                print(f"  - Attn Bias: {batch_A['attn_bias'].shape}  <-- [FIXED]")
        else:
            print(f"  - Node Features 'x': {batch_A.x.shape}")
            if hasattr(batch_A, 'attn_bias'):
                print(f"  - Attn Bias: {batch_A.attn_bias.shape}  <-- [FIXED]")
            
        print("\n[+] Dataloader test passed successfully! You are ready for Phase 2.5 Retraining.")

    except StopIteration:
        print("[-] Dataloader is empty.")
    except Exception as e:
        print(f"[!] Error during batch fetching: {e}")