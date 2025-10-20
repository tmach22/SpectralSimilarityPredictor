import pandas as pd
import numpy as np
from pathlib import Path

def format_spectra_file(input_path: Path, output_path: Path):
    """
    Loads a feather file with separate fragment_mz and fragment_intensities
    columns and transforms it to match the mgf_df.feather format.
    """
    print(f"Loading input data from: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    df = pd.read_feather(input_path)
    print("\n--- Initial DataFrame structure ---")
    print(df.head())
    print("\nColumns:", df.columns.tolist())

    # --- 1. Create the 'peaks' column ---
    # The core of the transformation is to combine the two fragment columns
    # into a single list of [mz, intensity] pairs for each row.
    print("\nCombining 'fragment_mz' and 'fragment_intensities' into 'peaks' column...")
    
    def create_peak_list(row):
        # Use zip to pair each mz with its corresponding intensity
        mzs = row['fragment_mz']
        intensities = row['fragment_intensities']
        if mzs is None or intensities is None:
            return None
        return [[mz, intensity] for mz, intensity in zip(mzs, intensities)]

    df['peaks'] = df.apply(create_peak_list, axis=1)

    # --- 3. Select and reorder columns to match the target format ---
    target_columns = [
        'spectrum_id',
        'peaks',
        'precursor_mz',
        'smiles',
        'adduct',
        'inchikey',
        'instrument'
    ]
    
    print(f"\nReformatting to target columns: {target_columns}")
    final_df = df[target_columns]

    # --- 4. Save the transformed DataFrame ---
    print(f"Saving formatted data to: {output_path}")
    final_df.to_feather(output_path)

    print("\n--- Final DataFrame structure ---")
    print(final_df.head())
    print("\nTransformation complete.")


if __name__ == '__main__':
    # Define the input and output file paths
    INPUT_FILE = Path('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/merged_spectra.feather')
    OUTPUT_FILE = Path('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/mgf_df_mona.feather')
    
    format_spectra_file(INPUT_FILE, OUTPUT_FILE)