import pandas as pd
import numpy as np

# Load your spectra dataframe
msg_df = pd.read_feather('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/merged_spec_sim_dataset.feather')
# (Assuming you load df here)

print(f"Initial shape: {msg_df.shape}")

# 1. Select and Copy
mol_data = msg_df[['spectrum_id', 'smiles', 'adduct', 'peaks', 'precursor_mz', 'instrument', 'collision_energy']].copy()
mol_data.rename(columns={'spectrum_id': 'spec_id', 'precursor_mz': 'prec_mz', 'instrument': 'inst'}, inplace=True)

# 2. Collision Energy Fix
# Downstream regex expects "35.0 eV". It does NOT handle "-1.0 eV".
# We must ensure unknowns remain None/NaN.
def format_ce(val):
    try:
        if pd.isna(val) or val < 0:
            return None # Will become null in JSON, handled as NaN downstream
        return f'{float(val):.1f} eV'
    except:
        return None

mol_data['col_energy'] = mol_data['collision_energy'].apply(format_ce)

# 3. Instrument Type
# Your manual mapping is fine, but data_utils has a robust map too.
# We initialize with empty string, then apply your specific overrides.
mol_data['inst_type'] = '' 
# Note: data_utils.py maps "Orbitrap" -> "FT" automatically, but doing it here is safe too.
mol_data.loc[mol_data['inst'].str.contains("Orbitrap", case=False, na=False), "inst_type"] = "FT"
mol_data.loc[mol_data['inst'].str.contains("QTOF", case=False, na=False), "inst_type"] = "QTOF"

# 4. Ion Mode and Frag Mode
mol_data['ion_mode'] = 'P'
mol_data['frag_mode'] = 'CID' 
# Optimization: If you have HCD data (common in Orbitraps), you might want to flag it
# mol_data.loc[mol_data['inst_type'] == 'FT', 'frag_mode'] = 'HCD' 

# 5. Adducts
# Downstream `parse_prec_type_str` handles "1+" and "1-". 
# Ensure your adducts don't have weird spacing.
mol_data['prec_type'] = mol_data['adduct'].astype(str).str.strip()

# 6. Initialize placeholders
mol_data['ri'] = None
mol_data['spec_type'] = None
mol_data['col_gas'] = None
mol_data['dataset'] = 'custom_dataset'

# Remove original column
mol_data.drop(columns=['collision_energy'], inplace=True)

# 7. Deduplicate
# Ensure unique spec_ids
mol_data.drop_duplicates(subset=['spec_id'], inplace=True)

# 8. Peaks Formatting
# Downstream expects: "mz1 int1\nmz2 int2\n"
def format_peaks(peak_list):
    if peak_list is None: return None
    # Handle the "Nuclear Fix" logic in case it's still an object array
    try:
        # If it's a numpy array, ensure it's iterable
        if hasattr(peak_list, 'tolist'):
            peak_list = peak_list.tolist()
        
        if len(peak_list) == 0: return None

        peaks_str_list = []
        for mz, intensity in peak_list:
            peaks_str_list.append(f"{float(mz)} {float(intensity)}")
        
        return "\n".join(peaks_str_list)
    except Exception as e:
        print(f"Error formatting peaks: {e}")
        return None

print("Formatting peaks column...")
mol_data['peaks'] = mol_data['peaks'].apply(format_peaks)

# Filter out rows where peaks failed formatting
print(f"Rows before dropping empty peaks: {len(mol_data)}")
mol_data = mol_data.dropna(subset=['peaks'])
print(f"Rows after dropping empty peaks: {len(mol_data)}")

print("Prepared DataFrame example:")
print(mol_data[['spec_id', 'col_energy', 'inst_type', 'prec_type']].head())

# Save
output_path = '/data/nas-gpu/wang/tmach007/massformer/data/df/custom_dataset_df.json'
print(f"Saving to {output_path}...")
mol_data.to_json(output_path, orient='records', indent=4)
print("Done.")

# --- WARNING CHECK ---
# Check for atoms that will be dropped by data_utils.py
# ELEMENT_LIST = ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F']
print("\n--- COMPATIBILITY CHECK ---")
try:
    from rdkit import Chem
    def has_bad_atoms(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return True
        valid_atoms = {'H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F'}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in valid_atoms:
                return True # Found a bad atom
        return False

    bad_atom_count = mol_data['smiles'].apply(has_bad_atoms).sum()
    if bad_atom_count > 0:
        print(f"WARNING: {bad_atom_count} molecules contain atoms other than [H, C, O, N, P, S, Cl, F].")
        print("These will be DROPPED by 'data_utils.check_mol_props'.")
        print("Please update 'ELEMENT_LIST' in 'data_utils.py' if you wish to keep them.")
    else:
        print("Atom types look compatible with data_utils.")
except ImportError:
    print("Could not run atom check (RDKit not installed in this env).")