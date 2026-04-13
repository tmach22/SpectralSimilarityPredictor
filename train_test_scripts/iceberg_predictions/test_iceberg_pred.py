import pandas as pd
import numpy as np
import os
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from tqdm import tqdm

# --- 1. CONFIGURATION ---
TEST_PAIRS_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/data_splits/stratified_binary_07_unseen_nist.feather"
PREDICTED_LIB_PATH = "/data/nas-gpu/wang/tmach007/ms-pred/results/msg_iceberg_predictions/iceberg_predicted_spec_df.pkl"
ORIGINAL_SPEC_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"
OUTPUT_CSV_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/results/predictions/binary_classifier/msg_iceberg_predictions/iceberg_classification_results.csv"

SIMILARITY_THRESHOLD = 0.7 

# --- 2. LOAD DATA & CREATE MAPPINGS ---
print("Loading datasets...")
test_df = pd.read_feather(TEST_PAIRS_PATH)
pred_lib_df = pd.read_pickle(PREDICTED_LIB_PATH)
spec_df = pd.read_pickle(ORIGINAL_SPEC_PATH)

# Step A: Map spec_id -> mol_id using the original spec file
print("Building ID mapping...")
spec_to_mol = spec_df.set_index('spec_id')['mol_id'].to_dict()

# Step B: Index predicted library by mol_id
# We sort peaks here to prevent the matchms AssertionError
print("Indexing predicted library by mol_id...")
mol_to_spec_obj = {}
for _, row in tqdm(pred_lib_df.iterrows(), total=len(pred_lib_df)):
    mid = row['mol_id']
    if not row['peaks'] or len(row['peaks']) == 0: continue
    
    # Extract and Sort Peaks
    mz, intensities = zip(*row['peaks'])
    mz_array = np.array(mz, dtype=float)
    int_array = np.array(intensities, dtype=float)
    
    sort_idx = np.argsort(mz_array)
    
    spec_obj = Spectrum(
        mz=mz_array[sort_idx],
        intensities=int_array[sort_idx],
        metadata={'precursor_mz': float(row['prec_mz'])}
    )
    mol_to_spec_obj[mid] = spec_obj

# --- 3. SIMILARITY CALCULATION ---
mod_cosine = ModifiedCosine(tolerance=0.05)
results = []

print(f"Calculating similarity for {len(test_df)} pairs...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    sid_main = row['name_main']
    sid_sub = row['name_sub']
    
    # Find the mol_id for these specific spec_ids
    mid_main = spec_to_mol.get(sid_main)
    mid_sub = spec_to_mol.get(sid_sub)
    
    # Retrieve the predicted spectra using the mol_id
    spec_main = mol_to_spec_obj.get(mid_main)
    spec_sub = mol_to_spec_obj.get(mid_sub)

    if spec_main and spec_sub:
        score_data = mod_cosine.pair(spec_main, spec_sub)
        sim_score = float(score_data['score'])
        
        results.append({
            'name_main': sid_main,
            'name_sub': sid_sub,
            'cosine_similarity': sim_score,
            'true_label': float(row['label']),
            'predicted_label': 1.0 if sim_score >= SIMILARITY_THRESHOLD else 0.0
        })

# --- 4. SAVE & REPORT ---
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_CSV_PATH}")
accuracy = (results_df['true_label'] == results_df['predicted_label']).mean()
print(f"Classification Accuracy: {accuracy:.4f}")