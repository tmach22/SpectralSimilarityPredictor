import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

# Import libraries for similarity calculations
try:
    from rdkit import Chem
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit is not installed. Please install it with 'conda install -c conda-forge rdkit'")

try:
    import matchms
    from matchms import Spectrum
    from matchms.similarity import CosineGreedy
except ImportError:
    raise ImportError("matchms is not installed. Please install it with 'pip install matchms'")

def create_oracle_pairs(input_path: Path, output_path: Path):
    """
    Generates a file of all possible pairs from an input spectra file,
    calculating both spectral and structural similarity for each pair.
    """
    print(f"Loading input data from: {input_path}")
    df = pd.read_feather(input_path)
    print(f"Loaded {len(df)} records.")

    # --- 1. Pre-compute fingerprints and create matchms Spectrum objects ---
    print("Preparing molecules and spectra for comparison...")

    mols = []
    fps = []
    spectra = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        mols.append(mol)
        
        # b. If the molecule is valid, calculate its Morgan fingerprint.
        if mol is not None:
            # Using a radius of 2 and 2048 bits is a standard choice.
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        else:
            # c. If SMILES is invalid, append None to keep indices aligned.
            fps.append(None)
            print(f"Warning: Could not parse SMILES '{row['smiles']}' for spectrum_id {row['spectrum_id']}. Tanimoto will be 0.")

        peak_list = row['peaks']
        
        # --- THIS IS THE FIX ---
        # Robustly parse the peaks to handle potential malformations.
        if peak_list is None or not isinstance(peak_list, list) or len(peak_list) == 0:
            spectra.append(None)
            continue

        try:
            # Separate mzs and intensities, validating each pair.
            mzs = []
            intensities = []
            for peak_pair in peak_list:
                # Ensure the pair is a list/tuple of length 2 before unpacking.
                if isinstance(peak_pair, (list, tuple)) and len(peak_pair) == 2:
                    mzs.append(peak_pair)
                    intensities.append(peak_pair[1])
            
            # If no valid peaks were found after filtering, skip this spectrum.
            if not mzs:
                spectra.append(None)
                continue

            # Create the Spectrum object from the clean, separated lists.
            spectrum = Spectrum(mz=np.array(mzs, dtype=np.float32),
                                intensities=np.array(intensities, dtype=np.float32),
                                metadata={'spectrum_id': row['spectrum_id']})
            spectra.append(spectrum)
        except (ValueError, TypeError) as e:
            # Catch any other unexpected conversion errors.
            print(f"Warning: Could not process peaks for spectrum_id {row['spectrum_id']}. Error: {e}. Skipping.")
            spectra.append(None)

    # --- 2. Generate all unique pairs ---
    print("\nGenerating all unique pairs...")
    indices = range(len(df))
    all_pairs_indices = list(combinations(indices, 2))
    print(f"Found {len(all_pairs_indices)} unique pairs to process.")

    # --- 3. Calculate similarities for each pair ---
    print("Calculating Tanimoto and Cosine similarities for all pairs...")
    
    results = []
    cosine_greedy = CosineGreedy(tolerance=0.1)

    for i, j in tqdm(all_pairs_indices, desc="Processing pairs"):
        spec_A, spec_B = spectra[i], spectra[j]
        fp_A, fp_B = fps[i], fps[j]

        cosine_score = 0.0
        if spec_A is not None and spec_B is not None:
            score = cosine_greedy.pair(spec_A, spec_B)
            # score can be None if there are no matching peaks
            if score is not None and 'score' in score:
                cosine_score = score['score']

        tanimoto_score = 0.0
        if fp_A is not None and fp_B is not None:
            tanimoto_score = TanimotoSimilarity(fp_A, fp_B)

        results.append({
            'name_main': df['spectrum_id'][i],
            'name_sub': df['spectrum_id'][j],
            'num_sites': 1,
            'tanimoto': tanimoto_score,
            'cosine_similarity': cosine_score
        })

    # --- 4. Create and save the final DataFrame ---
    print("\nAssembling and saving the final DataFrame...")
    final_df = pd.DataFrame(results)
    final_df = final_df[['name_main', 'name_sub', 'num_sites', 'tanimoto', 'cosine_similarity']]
    final_df.to_feather(output_path)

    print("\n--- Final DataFrame structure ---")
    print(final_df.head())
    print(f"\nSuccessfully created and saved '{output_path}'.")


if __name__ == '__main__':
    INPUT_FILE = Path('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/mgf_df_mona.feather')
    OUTPUT_FILE = Path('/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/oracle_pairs_all_mona.feather')
    
    create_oracle_pairs(INPUT_FILE, OUTPUT_FILE)