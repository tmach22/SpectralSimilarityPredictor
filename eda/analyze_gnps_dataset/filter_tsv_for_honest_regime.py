import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from matchms.importing import load_from_mgf
from matchms.logging_functions import set_matchms_logger_level

# --- 1. CONFIGURATION ---
set_matchms_logger_level("ERROR")

# --- 1. SETUP ARGUMENTS ---
parser = argparse.ArgumentParser(description="Filter a single GNPS TSV batch using a pinpointed MGF file.")
parser.add_argument("--input_tsv", required=True, help="Path to the individual GNPS TSV batch")
parser.add_argument("--input_mgf", required=True, help="Path to the corresponding MGF file for this batch")
parser.add_argument("--output_dir", default="gnps_hard_negative_honest_batches", help="Where to save the result")
args = parser.parse_args()

# --- 2. CONFIGURATION ---
# Use the absolute path to your MSG spec lookup
MSG_SPEC_PATH = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/mass_spec_gym_data/spec_df.pkl"

MIN_MASS_DIFF = 1.0
MAX_MASS_DIFF = 100.0

os.makedirs(args.output_dir, exist_ok=True)

# --- 3. LOAD MSG LOOKUP ---
print("Loading MassSpecGym metadata...")
msg_spec = pd.read_pickle(MSG_SPEC_PATH).set_index('spec_id')

# --- 4. PINPOINTED GNPS MASS LOOKUP ---
# Only scan the specific MGF file provided in the arguments
gnps_mass_lookup = {}
print(f"Pinpointing mass metadata in: {os.path.basename(args.input_mgf)}...")

# GNPS MGFs use 'scans' or 'title' as the primary ID
for spec in tqdm(load_from_mgf(args.input_mgf), desc="Reading MGF"):
    sid = str(spec.get("scans") or spec.get("title"))
    pmz = spec.get("precursor_mz")
    if pmz is not None:
        gnps_mass_lookup[sid] = float(pmz)

# --- 5. PROCESS THE SINGLE TSV FILE ---
batch_name = os.path.basename(args.input_tsv)
output_path = os.path.join(args.output_dir, f"honest_{batch_name}")

print(f"\nFiltering TSV: {batch_name}")
df = pd.read_csv(args.input_tsv, sep='\t')

if df.empty:
    print("Input TSV is empty. Exiting.")
    exit()

honest_pairs = []
for _, row in df.iterrows():
    gnps_id = str(row['gnps_id'])
    msg_id = row['msg_spec_id']
    
    # Get Precursor Mass from our pinpointed lookup
    m_gnps = gnps_mass_lookup.get(gnps_id)
    
    # Get Precursor Mass from the MSG index
    m_msg = msg_spec.at[msg_id, 'prec_mz'] if msg_id in msg_spec.index else None
    
    if m_gnps is not None and m_msg is not None:
        mass_diff = abs(m_gnps - m_msg)
        
        # Apply the Honest Zone constraint
        if MIN_MASS_DIFF <= mass_diff <= MAX_MASS_DIFF:
            row_dict = row.to_dict()
            row_dict['gnps_prec_mz'] = m_gnps
            row_dict['msg_prec_mz'] = m_msg
            row_dict['mass_diff'] = mass_diff
            honest_pairs.append(row_dict)

# --- 6. SAVE RESULTS ---
if honest_pairs:
    pd.DataFrame(honest_pairs).to_csv(output_path, sep='\t', index=False)
    print(f"Success: Kept {len(honest_pairs)} pairs. Saved to {output_path}")
else:
    print(f"No pairs in {batch_name} met the Honest Zone criteria.")