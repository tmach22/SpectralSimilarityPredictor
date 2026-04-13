import os
import tqdm
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import (
    default_filters, 
    normalize_intensities, 
    select_by_relative_intensity,
    require_minimum_number_of_peaks
)
from matchms.logging_functions import set_matchms_logger_level
# Only show "ERROR" or higher to stop the "Overwriting" warnings
set_matchms_logger_level("ERROR")

def apply_massspecgym_logic(spectrum):
    """
    Applies the exact normalization from MassSpecGym:
    1. Standardizes metadata keys.
    2. Scales intensities relative to the base peak (0.0 to 1.0).
    3. Scales Collision Energy (eV) to a 0-1 range (dividing by 100).
    4. Filters out noise peaks (<1% relative intensity).
    5. Discards spectra with <5 peaks.
    """
    if spectrum is None:
        return None

    # 1. Standard Metadata Cleaning (harmonizes keys like 'pepmass' -> 'precursor_mz')
    s = default_filters(spectrum)

    # Strict SMILES Filter
    # We check both the string content and common placeholders like "N/A"
    smiles = s.get("smiles")
    if not smiles or str(smiles).strip().upper() in ["N/A", "NONE", "NULL", ""]:
        return None

    # 2. Intensity Normalization (0.0 to 1.0)
    # MassSpecGym uses relative intensity to the base peak.
    s = normalize_intensities(s)

    # 3. Peak Filtering (MassSpecGym threshold: > 1% of base peak)
    s = select_by_relative_intensity(s, intensity_from=0.01, intensity_to=1.0)

    # 4. Information Density Check (MassSpecGym requires >= 5 peaks)
    s = require_minimum_number_of_peaks(s, n_required=5)
        
    if s is None:
        return None

    # 5. Collision Energy Normalization
    # The paper scales CE to [0, 1]. If it's in eV, we divide by 100.
    ce = s.get("collision_energy")
    if ce is None or ce == "" or ce == "N/A":
        final_ce = 30.0
    else:
        try:
            # Handle potential list formats from matchms
            final_ce = float(ce[0]) if isinstance(ce, (list, tuple)) else float(ce)
        except (ValueError, TypeError):
            final_ce = 30.0
            
    s.set("collision_energy", final_ce)

    return s

def run_serial_pipeline(input_file, output_dir, batch_size=10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spectra_gen = load_from_mgf(input_file)
    current_batch = []
    batch_id = 0
    total_count = 0

    with tqdm.tqdm(desc="MassSpecGym Processing", unit="batch") as pbar:
        for spec in spectra_gen:
            processed_spec = apply_massspecgym_logic(spec)
            
            if processed_spec:
                current_batch.append(processed_spec)
                total_count += 1
            
            # Save the ENTIRE LIST of 10,000 spectra to ONE file
            if len(current_batch) >= batch_size:
                output_path = os.path.join(output_dir, f"batch_{batch_id}.mgf")
                
                # We pass the list 'current_batch'. MatchMS will iterate 
                # through this list and write all 10,000 to 'batch_X.mgf'
                with open(output_path, 'w', encoding='utf-8') as f:
                    # Pass the list and the file handle 'f'
                    save_as_mgf(current_batch, f)
                
                batch_id += 1
                current_batch = [] # Clear the list for the next 10,000
                pbar.update(batch_size)

        # Final batch: Save whatever is left in the list
        if current_batch:
            output_path = os.path.join(output_dir, f"batch_{batch_id}.mgf")
            save_as_mgf(current_batch, output_path)
            pbar.update(len(current_batch))

    print(f"Done! Saved {total_count} total spectra across {batch_id + 1} files.")

if __name__ == "__main__":
    run_serial_pipeline("/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/ALL_GNPS_cleaned.mgf", "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/ALL_GNPS_cleaned/")