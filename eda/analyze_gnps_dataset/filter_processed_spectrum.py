import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from matchms.importing import load_from_mgf
from matchms.similarity import ModifiedCosine
from matchms.filtering import normalize_intensities, select_by_relative_intensity
from matchms import Spectrum
from matchms.logging_functions import set_matchms_logger_level

# --- 1. CONFIGURATION ---
set_matchms_logger_level("ERROR")

GNPS_BATCH_DIR = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/ALL_GNPS_cleaned/" 
MSG_SPEC_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"
MSG_MOL_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/mol_df.pkl"

# New directory to store individual batch results
OUTPUT_DIR = "gnps_hard_negative_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COSINE_THRESHOLD = 0.7
TOLERANCE = 0.05

# --- 2. LOAD & PREPARE MASS SPEC GYM REFERENCE ---
print("Loading MassSpecGym reference library...")
msg_spec = pd.read_pickle(MSG_SPEC_PATH)
msg_mol = pd.read_pickle(MSG_MOL_PATH)

msg_merged = msg_spec.merge(msg_mol[['mol_id', 'inchikey_s']], on='mol_id', how='left')

print("Building MSG matchms library...")
msg_library = []
for _, row in tqdm(msg_merged.iterrows(), total=len(msg_merged)):
    mz, inten = zip(*row['peaks'])
    spec = Spectrum(mz=np.array(mz, dtype=float), 
                    intensities=np.array(inten, dtype=float), 
                    metadata={'spec_id': row['spec_id'], 
                              'inchikey': row['inchikey_s'],
                              'precursor_mz': row['prec_mz']})
    spec = normalize_intensities(spec)
    spec = select_by_relative_intensity(spec, intensity_from=0.001)
    msg_library.append(spec)

# --- 3. MINING LOOP (UPDATED FOR BATCH SAVING) ---
mc = ModifiedCosine(tolerance=TOLERANCE)
mgf_files = glob.glob(os.path.join(GNPS_BATCH_DIR, "*.mgf"))

for mgf_file in mgf_files:
    batch_name = os.path.basename(mgf_file)
    output_filename = os.path.join(OUTPUT_DIR, f"pairs_{batch_name.replace('.mgf', '.tsv')}")
    
    # RESUME CHECK: Skip if this batch has already been processed
    if os.path.exists(output_filename):
        print(f"Skipping {batch_name} (Results already exist).")
        continue

    print(f"\nMining batch: {batch_name}")
    gnps_spectra = list(load_from_mgf(mgf_file))
    batch_pairs = []
    
    for query in tqdm(gnps_spectra, desc=f"Processing {batch_name}"):
        query = normalize_intensities(query)
        query = select_by_relative_intensity(query, intensity_from=0.001)
        if query is None or len(query.peaks.mz) < 10: 
            continue
            
        q_inchikey = query.get("inchikey")
        if q_inchikey: 
            q_inchikey = q_inchikey[:14]
        
        q_mz = query.get("precursor_mz")
        if q_mz is None:
            continue
            
        for ref in msg_library:
            r_inchikey = ref.get("inchikey")
            if q_inchikey and r_inchikey and q_inchikey == r_inchikey:
                continue
            
            if abs(q_mz - ref.get("precursor_mz")) > 1.0:
                continue
            
            score = mc.pair(query, ref)
            if score and score['score'] >= COSINE_THRESHOLD:
                batch_pairs.append({
                    'gnps_id': query.get("scans") or query.get("title"),
                    'msg_spec_id': ref.get("spec_id"),
                    'cosine_similarity': score['score'],
                    'gnps_inchikey': q_inchikey,
                    'msg_inchikey': r_inchikey,
                    'gnps_smiles': query.get("smiles")
                })

    # SAVE BATCH RESULTS IMMEDIATELY
    if batch_pairs:
        df_batch = pd.DataFrame(batch_pairs)
        df_batch.to_csv(output_filename, sep='\t', index=False)
        print(f"Done. Found {len(batch_pairs)} pairs in {batch_name}.")
    else:
        # Create an empty file so the resume check knows this batch was checked
        open(output_filename, 'a').close()
        print(f"No pairs found in {batch_name}.")

print("\nAll batches complete.")