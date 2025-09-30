import pandas as pd
import numpy as np

# Load your spectra dataframe
msg_df = pd.read_feather('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/msg_df.feather')

# Select and rename columns to be more compatible with the MassFormer pipeline's expectations
# We only need the unique identifier and the structural/adduct information.
mol_data = msg_df[['spectrum_id', 'smiles', 'adduct', 'peaks', 'precursor_mz', 'instrument']].copy()
mol_data.rename(columns={'spectrum_id': 'spec_id', 'precursor_mz': 'prec_mz', 'instrument': 'inst'}, inplace=True)

# The MassFormer model was trained with collision energy as a feature.
# Your dataframe does not have this. We will add a placeholder column.
# A common default or median value like 35.0 is a reasonable choice.
mol_data['col_energy'] = '35.0 eV'

mol_data['inst_type'] = ''
mol_data.loc[mol_data['inst'] == "Orbitrap", "inst_type"] = "FT"
mol_data.loc[mol_data['inst'] == "QTOF", "inst_type"] = "QTOF" # Assuming all are MS2 spectra

mol_data['ion_mode'] = 'P'

mol_data['frag_mode'] = 'CID'  # Assuming all are CID spectra

mol_data['prec_type'] = None
mol_data['ri'] = None
# The MassFormer pipeline also expects a 'dataset' column. We'll add one.
mol_data['dataset'] = 'custom_dataset'

# Ensure there are no duplicate molecules to avoid redundant processing
mol_data.drop_duplicates(subset=['spec_id'], inplace=True)

def format_peaks(peak_list):
    # If the entry is NOT a list (e.g., it's NaN, None, etc.), return an empty string.
    if not isinstance(peak_list, np.ndarray) or len(peak_list) == 0:
        print("Warning: Found non-list entry in peaks column. Converting to empty string.")
        return None
    peaks = []
    for mz, intensity in peak_list:
        if mz is not None and intensity is not None:
            peaks.append(f"{mz} {intensity}\n")
    return "".join(peaks)
    
print("Formatting peaks column...")
mol_data['peaks'] = mol_data['peaks'].apply(format_peaks)

print("Prepared DataFrame with unique molecules:")
print(mol_data.head())
print(f"\nTotal unique molecules to process: {len(mol_data)}")

# Save the preprocessed dataframe to a new feather file
mol_data.to_json('/data/nas-gpu/wang/tmach007/massformer/data/df/custom_dataset_df.json', orient='records', indent=4)

# This DataFrame is now ready to be processed by the MassFormer scripts.