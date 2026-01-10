import torch
import pandas as pd
import numpy as np
import sys

def inspect_file(path):
    print(f"--- Inspecting: {path} ---")
    
    try:
        # Load data (map_location ensures we don't need a GPU to peek)
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load file. {e}")
        return

    print(f"Object Type: {type(data)}")

    # CASE A: It is a Dictionary (Could be Model Weights OR Data Dict)
    if isinstance(data, dict):
        print(f"Keys found: {list(data.keys())}")
        
        # Check for Model Weights signature
        if any(k in data.keys() for k in ['state_dict', 'model_state', 'epoch', 'optimizer']):
            print("\n⚠️  WARNING: This looks like a MODEL CHECKPOINT.")
            print("   It likely contains weights, not the training data.")
            if 'state_dict' in data:
                print(f"   Model layers found: {len(data['state_dict'])}")
        
        # Check for Data signature
        elif any(k in data.keys() for k in ['x', 'y', 'features', 'labels', 'ids', 'smiles']):
            print("\n✅ HOPE: This looks like a DATASET.")
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"   Key '{k}': Tensor shape {v.shape}")
                elif isinstance(v, list):
                    print(f"   Key '{k}': List length {len(v)}")
                    print(f"   First item: {v[0]}")
        
        # Print a small sample of the dictionary content structure
        print("\n--- Content Sample ---")
        for i, (k, v) in enumerate(data.items()):
            if i > 2: break
            print(f"{k}: {type(v)}")

    # CASE B: It is a List (Likely a list of Data objects or Tuples)
    elif isinstance(data, list):
        print(f"\n✅ HOPE: This looks like a LIST DATASET.")
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"Sample Element [0]: {data[0]}")
            print(f"Type of Element: {type(data[0])}")

    # CASE C: It is a Tensor (Just raw features?)
    elif isinstance(data, torch.Tensor):
        print(f"\n⚠️  WARNING: This is a raw TENSOR.")
        print(f"Shape: {data.shape}")
        print("   If this contains features, we might lack the IDs (SMILES) to map them back.")

    else:
        print(f"Unknown structure: {data}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_pt.py <path_to_file.pt>")
    else:
        inspect_file(sys.argv[1])