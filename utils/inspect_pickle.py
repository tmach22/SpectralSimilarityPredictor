import pickle
import os
import sys
import pandas as pd
from collections.abc import Sized

def inspect_pickle(file_path):
    """
    Inspects a pickle file, reporting its size, object type, and structure 
    (size/shape) without executing any unknown code.
    """
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at path: {file_path}")
        return

    # 1. Report File Size
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print("-" * 50)
    print(f"File Path: {file_path}")
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
    print("-" * 50)

    # SAFETY WARNING
    print("WARNING: Loading pickle files from unknown sources can execute arbitrary code.")
    print("         Only proceed if you trust the source of this file.")
    print("-" * 50)

    try:
        # 2. Attempt to Load the Object
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 3. Report Object Type
        data_type = type(data)
        print(f"Object Type: {data_type}")
        
        # 4. Report Structure based on Type
        print("--- Structure Summary ---")
        
        if isinstance(data, pd.DataFrame):
            print(f"This is a Pandas DataFrame.")
            print(f"Shape (Rows, Columns): {data.shape}")
            print("First 5 column names:")
            print(list(data.columns[:5]))

            print("First 5 rows:")
            print(data.head())
            
        elif isinstance(data, dict):
            print(f"This is a dictionary (dict).")
            print(f"Total keys: {len(data)}")
            # Sample 5 keys or fewer
            sample_keys = list(data.keys())[:5]
            if sample_keys:
                print("Sample keys:")
                for key in sample_keys:
                    print(f"  - {key} (Type: {type(key).__name__})")
            
        elif isinstance(data, (list, tuple, set)) and isinstance(data, Sized):
            print(f"This is a sequence/collection.")
            print(f"Total items: {len(data)}")
            if len(data) > 0:
                print(f"Type of the first item: {type(data[0])}")
            
        elif isinstance(data, Sized):
             # Covers general sized objects not covered above
            print(f"Object size (length): {len(data)}")

        else:
            print("No specific size/shape information available for this type.")

        print("-" * 50)

    except pickle.UnpicklingError as e:
        print(f"ERROR: Could not unpickle the file. It may be corrupted or created with a newer/different protocol.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pickle_inspector.py <path/to/your/file.pkl>")
        sys.exit(1)
        
    pickle_file_path = sys.argv[1]
    inspect_pickle(pickle_file_path)