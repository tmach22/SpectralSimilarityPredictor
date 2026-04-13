import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# --- 1. CONFIGURATION ---
NIST_REF_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/NIST_dataset.tsv"
TEST_SET_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/data_splits/stratified_binary_08_dataset_test.feather"
SPEC_LOOKUP_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"
MOL_LOOKUP_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df.pkl"
OUTPUT_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/data_splits/stratified_binary_08_unseen_nist.feather"

def get_skeleton_from_smi(smiles):
    if pd.isna(smiles): return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol: return Chem.MolToInchiKey(mol)[:14]
    except: return None
    return None

# --- 2. BUILD NIST BLACKLIST ---
print("Building NIST training blacklist...")
nist_df = pd.read_csv(NIST_REF_PATH, sep='\t')
# Using a set for O(1) lookup speed
nist_skeletons = set()
for smiles in tqdm(nist_df['smiles'].unique(), desc="Hashing NIST Skeletons"):
    skel = get_skeleton_from_smi(smiles)
    if skel: nist_skeletons.add(skel)

# --- 3. MAP MASS SPEC GYM IDs TO SKELETONS ---
print("Loading MassSpecGym lookups...")
spec_df = pd.read_pickle(SPEC_LOOKUP_PATH)
mol_df = pd.read_pickle(MOL_LOOKUP_PATH)

# Merge to get spec_id and inchikey connectivity
id_mapping = spec_df.merge(mol_df[['mol_id', 'inchikey_s']], on='mol_id', how='left')
# Map spec_id -> first 14 chars of InChIKey
id_to_skeleton = id_mapping.set_index('spec_id')['inchikey_s'].str[:14].to_dict()

# --- 4. FILTER THE FEATHER DATASET ---
print("Loading and filtering the test feather file...")
test_df = pd.read_feather(TEST_SET_PATH)

def is_seen_by_iceberg(row):
    # Retrieve the skeleton for both sides of the pair
    skel_main = id_to_skeleton.get(row['name_main'])
    skel_sub = id_to_skeleton.get(row['name_sub'])
    
    # Check if either molecule was in the NIST training set
    return (skel_main in nist_skeletons) or (skel_sub in nist_skeletons)

tqdm.pandas(desc="Filtering NIST Overlap")
is_contaminated = test_df.progress_apply(is_seen_by_iceberg, axis=1)

# Keep only those not contaminated
test_df_unseen = test_df[~is_contaminated].copy()

# --- 5. SAVE AS FEATHER ---
print(f"\nOriginal count: {len(test_df)}")
print(f"Removed count: {is_contaminated.sum()}")
print(f"Unseen count: {len(test_df_unseen)}")

# Feather requires a reset index to avoid issues with non-standard index types
test_df_unseen.reset_index(drop=True).to_feather(OUTPUT_PATH)
print(f"Successfully saved clean feather to: {OUTPUT_PATH}")