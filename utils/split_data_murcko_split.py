import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import os

# --- Configuration ---
# 1. Input file paths
SPECTRA_FILE = '/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/msg_df.feather'
PAIRS_FILE = '/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/oracle_pairs_all.feather'

# 2. Output directory for the final data splits
OUTPUT_DIRECTORY = '/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/data_splits/'

# 3. Define the split ratios
VALIDATION_SIZE = 0.1  # 10% for validation
TEST_SIZE = 0.1      # 10% for testing
# Training size will be the remaining 80%

# --- Main Script ---
print("--- Starting Rigorous Scaffold-Based Data Splitting ---")

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# --- Step 1: Load and Merge Initial Data ---
print("Loading and merging data files...")
try:
    spectra_df = pd.read_feather(SPECTRA_FILE)
    pairs_df = pd.read_feather(PAIRS_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find a required file. Details: {e}")
    exit()

# Create a mapping from spectrum_id to smiles for efficient lookup
id_to_smiles = spectra_df.set_index('spectrum_id')['smiles'].to_dict()
# --- Step 2: Get All Unique Molecules and Calculate Scaffolds ---
print("Extracting unique molecules and calculating Murcko scaffolds...")
unique_smiles = spectra_df['smiles'].dropna().unique()
scaffold_to_smiles = defaultdict(list)

for smiles in tqdm(unique_smiles, desc="Generating Scaffolds"):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_to_smiles[scaffold].append(smiles)
        except ValueError:
            # Handle cases where scaffold generation fails
            scaffold_to_smiles[""].append(smiles) # Group molecules without scaffolds

unique_scaffolds = list(scaffold_to_smiles.keys())
print(f"Found {len(unique_smiles)} unique molecules with {len(unique_scaffolds)} unique scaffolds.")

# --- Step 3: Split the Scaffolds into Train, Validation, and Test Sets ---
print("Splitting scaffolds into train, validation, and test sets...")
train_scaffolds, temp_scaffolds = train_test_split(
    unique_scaffolds, 
    test_size=(VALIDATION_SIZE + TEST_SIZE), 
    random_state=42
)
validation_scaffolds, test_scaffolds = train_test_split(
    temp_scaffolds,
    test_size=(TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)),
    random_state=42
)

print(f"Scaffold splits: {len(train_scaffolds)} train, {len(validation_scaffolds)} validation, {len(test_scaffolds)} test.")

# --- Step 4: Assign All Molecules to Their Respective Splits ---
print("Assigning molecules to splits based on their scaffolds...")
train_smiles = set([s for scaffold in train_scaffolds for s in scaffold_to_smiles[scaffold]])
validation_smiles = set([s for scaffold in validation_scaffolds for s in scaffold_to_smiles[scaffold]])
test_smiles = set([s for scaffold in test_scaffolds for s in scaffold_to_smiles[scaffold]])

# --- Step 5: Assign Pairs to Final Datasets ---
print("Assigning all ~24 million pairs to the final datasets. This may take a few minutes...")
train_indices, validation_indices, test_indices = [], [], []

# Use tqdm for progress tracking on the large pairs DataFrame
for i, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Assigning Pairs"):
    smiles_main = id_to_smiles.get(row['name_main'])
    smiles_sub = id_to_smiles.get(row['name_sub'])

    if smiles_main and smiles_sub:
        # Assign to a split only if BOTH molecules belong to that split
        if smiles_main in train_smiles and smiles_sub in train_smiles:
            train_indices.append(i)
        elif smiles_main in validation_smiles and smiles_sub in validation_smiles:
            validation_indices.append(i)
        elif smiles_main in test_smiles and smiles_sub in test_smiles:
            test_indices.append(i)

# --- Step 6: Create and Save the Final DataFrames ---
print("Creating and saving the final data splits...")
train_pairs_df = pairs_df.loc[train_indices]
validation_pairs_df = pairs_df.loc[validation_indices]
test_pairs_df = pairs_df.loc[test_indices]

train_pairs_df.to_feather(os.path.join(OUTPUT_DIRECTORY, 'train_pairs.feather'))
validation_pairs_df.to_feather(os.path.join(OUTPUT_DIRECTORY, 'validation_pairs.feather'))
test_pairs_df.to_feather(os.path.join(OUTPUT_DIRECTORY, 'test_pairs.feather'))

print("\n--- Data Splitting Complete ---")
print(f"Training pairs: {len(train_pairs_df)}")
print(f"Validation pairs: {len(validation_pairs_df)}")
print(f"Test pairs: {len(test_pairs_df)}")
print(f"Data splits saved to: {OUTPUT_DIRECTORY}")