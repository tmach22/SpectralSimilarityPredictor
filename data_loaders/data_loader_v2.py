import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np ### NEW ###

from pathlib import Path
import os
import sys
cwd = Path.cwd()
print(f"Current working directory: {cwd}")
parent_directory = os.path.dirname(cwd.parent)
print(f"Parent directory: {parent_directory}")
script_dir = os.path.join(parent_directory, 'tmach007', 'massformer', 'src', 'massformer')
print(f"Adding {script_dir} to sys.path")
sys.path.insert(0, script_dir)

### MODIFIED ###
# Import the new utilities we need
from gf_data_utils import gf_preprocess, collator
# We import np_one_hot and EPS from the original MassFormer utils 
from misc_utils import np_one_hot, EPS 
### END MODIFIED ###

class SpectralSimilarityDataset(Dataset):
    """
    ### MODIFIED ###
    Custom PyTorch Dataset for loading pairs of molecular graphs,
    their shared metadata, and their spectral similarity score.
    ### END MODIFIED ###
    """
    ### MODIFIED ###
    def __init__(self, pairs_feather_path, spec_data_path, mol_data_path, subset_size=None):
        super().__init__()
        print(f"Loading pairs data from {pairs_feather_path}...")
        pairs_df = pd.read_feather(pairs_feather_path)
        
        print(f"Loading spectral data from {spec_data_path}...")
        # Load the full spec_df to set up metadata maps 
        spec_df = pd.read_pickle(spec_data_path)
        
        ### NEW ###
        # Set up all the metadata processing dictionaries and stats
        # This function is adapted from _setup_spec_metadata_dicts in dataset.py 
        print("Setting up metadata encoders...")
        self._setup_metadata_maps(spec_df) 
        ### END NEW ###

        # Create the first lookup table (spec_id -> full spec entry)
        self.spec_lookup = spec_df.set_index('spec_id')

        print(f"Loading molecular graph data from {mol_data_path}...")
        mol_df = pd.read_pickle(mol_data_path)
        # Create the second lookup table (mol_id -> graph_obj)
        self.mol_lookup = mol_df.set_index('mol_id')

        print("Filtering pairs to match available processed spectra...")
        initial_pair_count = len(pairs_df)
        valid_spec_ids = set(self.spec_lookup.index)
        print(f"Total valid spectra available: {len(valid_spec_ids)}")
        print(f"First 5 valid spec IDs: {list(valid_spec_ids)[:5]}")
        self.pairs_df = pairs_df[
            pairs_df['name_main'].isin(valid_spec_ids) &
            pairs_df['name_sub'].isin(valid_spec_ids)
        ].reset_index(drop=True)
        final_pair_count = len(self.pairs_df)
        print(f"Filtering complete. Kept {final_pair_count} of {initial_pair_count} pairs.")

        if subset_size is not None:
            print(f"--- Using a subset of the data: {subset_size} ---")
            if isinstance(subset_size, float) and 0 < subset_size <= 1.0:
                self.pairs_df = self.pairs_df.sample(frac=subset_size, random_state=42).reset_index(drop=True)
            elif isinstance(subset_size, int) and subset_size > 0:
                num_samples = min(subset_size, final_pair_count)
                self.pairs_df = self.pairs_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
            else:
                raise ValueError("subset_size must be a float between 0 and 1, or a positive integer.")
            print(f"New dataset size: {len(self.pairs_df)} pairs.")
    ### END MODIFIED ###

    ### NEW ###
    def _setup_metadata_maps(self, spec_df):
        """
        Infers metadata vocabularies and stats from the spec_df.
        This logic is adapted from _setup_spec_metadata_dicts in dataset.py.
        It uses the column names from your 'augmented_msg_df.feather' file.
        """
        # We must make assumptions about these config values, based on dataset.py
        # and your project's data (e.g., augmented_msg_df.feather)
        self.ce_key = "ace" # Use the column name from your file
        self.preproc_ce = "normalize" # Assumes normalization, as in dataset.py 
        
        # Infer vocabularies directly from the loaded spec_df 
        inst_type_list = sorted(spec_df["inst_type"].unique().tolist())
        prec_type_list = sorted(spec_df["prec_type"].unique().tolist())
        frag_mode_list = sorted(spec_df["frag_mode"].unique().tolist()) 

        # Create the string-to-index maps 
        self.inst_type_c2i = {s: i for i, s in enumerate(inst_type_list)}
        self.num_inst_type = len(inst_type_list)
        
        self.prec_type_c2i = {s: i for i, s in enumerate(prec_type_list)}
        self.num_prec_type = len(prec_type_list)
        
        self.frag_mode_c2i = {s: i for i, s in enumerate(frag_mode_list)}
        self.num_frag_mode = len(frag_mode_list)

        # Calculate stats for collision energy normalization 
        self.mean_ce = spec_df[self.ce_key].mean()
        self.std_ce = spec_df[self.ce_key].std()
        print(f"Metadata setup complete. CE Mean: {self.mean_ce:.2f}, CE Std: {self.std_ce:.2f}")

    def _process_ce(self, col_energy):
        """ Processes the collision energy float. Based on ce_func in dataset.py. """
        if self.preproc_ce == "normalize":
            normalized_ce = (col_energy - self.mean_ce) / (self.std_ce + EPS)
            # Then, create a 1D tensor from this value
            col_energy_meta = torch.tensor(
                [normalized_ce], 
                dtype=torch.float32
            )
        else: # "none" or other
            col_energy_meta = torch.tensor([col_energy], dtype=torch.float32)
        return col_energy_meta

    def _get_spec_meta(self, spec_entry):
        """
        Processes a single spectrum's metadata into a tensor.
        This logic is adapted from get_spec_feats in dataset.py.
        """
        # 1. Get raw values (using your column names)
        col_energy = spec_entry[self.ce_key]
        inst_type = spec_entry["inst_type"]
        prec_type = spec_entry["prec_type"]
        frag_mode = spec_entry["frag_mode"] # Using 'instrument' as proxy for frag_mode

        # 2. Process Collision Energy
        col_energy_meta = self._process_ce(col_energy)

        # 3. Process Categorical Data (One-Hot Encoding) 
        inst_type_idx = self.inst_type_c2i[inst_type]
        inst_type_meta = torch.as_tensor(
            np_one_hot(
                inst_type_idx,
                num_classes=self.num_inst_type),
            dtype=torch.float32)

        prec_type_idx = self.prec_type_c2i[prec_type]
        prec_type_meta = torch.as_tensor(
            np_one_hot(
                prec_type_idx,
                num_classes=self.num_prec_type),
            dtype=torch.float32)
            
        frag_mode_idx = self.frag_mode_c2i[frag_mode]
        frag_mode_meta = torch.as_tensor(
            np_one_hot(
                frag_mode_idx,
                num_classes=self.num_frag_mode),
            dtype=torch.float32)

        # 4. Assemble final vector (matches line 696 in dataset.py) 
        # Note: The original implementation concatenates col_energy_meta twice.
        # We replicate that behavior here to be identical.
        spec_meta_list = [
            col_energy_meta,
            inst_type_meta,
            prec_type_meta,
            frag_mode_meta,
            col_energy_meta]
        
        # Concatenate all features and add a batch dimension
        spec_meta = torch.cat(spec_meta_list, dim=0).unsqueeze(0)
        return spec_meta
    ### END NEW ###

    def __len__(self):
        return len(self.pairs_df)

    ### MODIFIED ###
    def __getitem__(self, idx):
        # 1. Get the instruction for this specific training example
        pair_info = self.pairs_df.iloc[idx]
        id_A = pair_info['name_main']
        id_B = pair_info['name_sub']
        similarity = torch.tensor(pair_info['cosine_similarity'], dtype=torch.float32)

        # 2. Look up the full spectrum entry for molecule A
        #    (We only need one, since metadata is shared for the pair)
        spec_entry_A = self.spec_lookup.loc[id_A]
        spec_entry_B = self.spec_lookup.loc[id_B]
        
        # 3. Use the spec_id to find the mol_id for each molecule
        mol_id_A = spec_entry_A['mol_id']
        mol_id_B = spec_entry_B['mol_id']

        # 4. Retrieve the PRE-COMPUTED graph objects (FAST)
        mol_A = self.mol_lookup.loc[mol_id_A, 'mol']
        mol_B = self.mol_lookup.loc[mol_id_B, 'mol']

        graph_A = gf_preprocess(mol_A, idx)
        graph_B = gf_preprocess(mol_B, idx)
        
        # 5. Get the "physics-aware" metadata tensor
        spec_meta = self._get_spec_meta(spec_entry_A)
        
        # 6. Return the complete set of data
        return graph_A, graph_B, spec_meta, similarity
    ### END MODIFIED ###

### MODIFIED ###
# Create an updated collate function for our new "physics-aware" task
def siamese_collate_fn(batch):
    # Unzip the batch of 4-tuples
    graphs_A, graphs_B, spec_metas, similarities = zip(*batch)
    
    # Collate the two graph batches separately
    batch_A = collator(graphs_A)
    batch_B = collator(graphs_B)
    
    # Stack the metadata and similarity tensors
    batch_meta = torch.cat(spec_metas, dim=0)
    batch_sim = torch.stack(similarities, 0)
    
    return batch_A, batch_B, batch_meta, batch_sim
### END MODIFIED ###

# =============================================================================
# Testing Block: Run this script directly to test the DataLoader
# =============================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test the GNN data loader with rich features.")
    parser.add_argument("--pairs_path", type=str, required=True, help="Path to the feather file.")
    parser.add_argument("--mol_data_path", type=str, required=True, help="Path to the pre-built graph data.")
    parser.add_argument("--spec_data_path", type=str, required=True, help="Path to the spectrum to molecule mapping data.")
    parser.add_argument("--subset_size", type=float, default=1.0, help="Number of pairs to load for testing.")
    args = parser.parse_args()
    print(f"Args: {args}")
    
    print("--- Initializing Dataset ---")
    dataset = SpectralSimilarityDataset(
        pairs_feather_path=args.pairs_path,
        spec_data_path=args.spec_data_path,
        mol_data_path=args.mol_data_path,
        subset_size=args.subset_size
    )
    print(f"Dataset initialized successfully. Total number of pairs: {len(dataset)}\n")

    # --- 2. Test retrieving a single item ---
    print("--- Testing __getitem__ (retrieving one sample) ---")
    ### MODIFIED ###
    # Get the first sample from the dataset (now 4 items)
    graph_A, graph_B, spec_meta, similarity_score = dataset[0]
    
    print(f"Sample 0 loaded.")
    print(f"  - Type of Graph A: {type(graph_A)}")
    print(f"  - Graph A (sample keys): {list(graph_A.keys())[:5]}...")
    print(f"  - Type of Graph B: {type(graph_B)}")
    print(f"  - Graph B (sample keys): {list(graph_B.keys())[:5]}...")
    print(f"  - Spec Meta Shape: {spec_meta.shape}")
    print(f"  - Similarity Score: {similarity_score.item():.4f} (dtype: {similarity_score.dtype})\n")
    ### END MODIFIED ###

    # --- 3. Test the DataLoader for batching ---
    print("--- Testing DataLoader (batching multiple samples) ---")

    siamese_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=siamese_collate_fn)
    
    ### MODIFIED ###
    # Get the first batch from the new loader (now 4 items)
    batch_A, batch_B, batch_meta, batch_similarity = next(iter(siamese_loader))
    
    print("First batch loaded successfully.")
    print("\n--- Batch A (a dictionary of padded tensors) ---")
    for key, value in batch_A.items():
        print(f"  - {key}: {value.shape}")
        
    print("\n--- Batch B (a dictionary of padded tensors) ---")
    for key, value in batch_B.items():
        print(f"  - {key}: {value.shape}")
        
    print(f"\n--- Spec Meta Tensor ---")
    print(f"  - Shape: {batch_meta.shape}") # Should be [batch_size, num_meta_features]

    print(f"\n--- Similarity Tensor ---")
    print(f"  - Shape: {batch_similarity.shape}")
    print(f"  - Values: {batch_similarity.numpy().round(4)}")