import pandas as pd
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine as scipy_cosine # Cosine distance = 1 - cosine similarity
from tqdm import tqdm
import random

# --- Helper Functions ---

def parse_peaks_object(peaks_obj):
    """
    Parses an object from the 'peaks' column, which is expected to be
    a list of lists, a numpy ndarray of shape (n_peaks, 2),
    OR a list/array of 1D arrays [array([mz, int]), ...].
    """
    try:
        # Check for None or pandas NaT/NaN
        if peaks_obj is None:
            return None, None
        # Handle float NaN (which can appear in object columns)
        if isinstance(peaks_obj, float) and np.isnan(peaks_obj):
            return None, None
        # Handle cases where it might be a numpy array containing NaNs
        if isinstance(peaks_obj, np.ndarray) and pd.isna(peaks_obj).any():
             return None, None
        
        # Check for empty list [] or empty array
        if len(peaks_obj) == 0:
            return [], []

        # --- NEW LOGIC to handle list of 1D arrays ---
        try:
            # np.vstack intelligently stacks a sequence of arrays vertically
            # This handles:
            # 1. list of lists: [[mz, int], [mz, int]]
            # 2. list of 1D arrays: [array([mz, int]), array([mz, int])]
            peaks_arr = np.vstack(peaks_obj)
        except ValueError:
            # Fallback for other structures (e.g., already a 2D array)
            peaks_arr = np.array(peaks_obj)
        # --- END NEW LOGIC ---

        # Check for empty array (e.g., []) after conversion
        if peaks_arr.size == 0:
            return [], []

        # Check for correct 2D shape (N, 2)
        if peaks_arr.ndim == 2 and peaks_arr.shape[1] == 2:
            mzs = peaks_arr[:, 0].tolist()
            intensities = peaks_arr[:, 1].tolist()
            return mzs, intensities
        else:
            # print(f"Warning: Unexpected peaks shape after processing: {peaks_arr.shape}")
            return None, None # Malformed data

    except Exception as e:
        # print(f"Warning: Error parsing peaks object '{peaks_obj}': {e}")
        return None, None # Error parsing

def parse_comma_sep_string(mzs_str, intensities_str):
    """ Parses comma-separated m/z and intensity strings. """
    try:
        mzs = [float(m) for m in mzs_str.split(',')]
        intensities = [float(i) for i in intensities_str.split(',')]
        return mzs, intensities
    except Exception as e:
        # print(f"Warning: Error parsing comma-sep string: {e}")
        return None, None

def process_spectrum_for_cosine(mzs, intensities, precision=10):
    """ Processes lists of m/z and intensities into a sparse vector dict for cosine calculation. """
    if mzs is None or intensities is None: return {}
    try:
        intensities_sqrt = np.sqrt(np.array(intensities, dtype=float))
        binned_spectrum = {}
        for mz, intensity in zip(mzs, intensities_sqrt):
            if intensity > 0:
                bin_index = int(np.round(mz * precision)) # Round for stability
                binned_spectrum[bin_index] = binned_spectrum.get(bin_index, 0) + intensity
        # Normalize (L2)
        norm = np.sqrt(sum(v**2 for v in binned_spectrum.values()))
        if norm > 0:
            for bin_index in binned_spectrum: binned_spectrum[bin_index] /= norm
        return binned_spectrum
    except Exception as e:
        # print(f"Warning: Error processing spectrum for cosine: {e}")
        return {}


def fast_cosine_similarity(spec_dict_a, spec_dict_b):
    """ Calculates cosine similarity between two sparse dict spectra (pre-normalized). """
    if not isinstance(spec_dict_a, dict) or not isinstance(spec_dict_b, dict): return 0.0
    dot_product = 0.0
    # Iterate over smaller dict
    if len(spec_dict_a) < len(spec_dict_b):
        for bin_index in spec_dict_a:
            if bin_index in spec_dict_b: dot_product += spec_dict_a[bin_index] * spec_dict_b[bin_index]
    else:
        for bin_index in spec_dict_b:
            if bin_index in spec_dict_a: dot_product += spec_dict_a[bin_index] * spec_dict_b[bin_index]
    return dot_product

def plot_mirror_spectrum(mzs1, ints1, mzs2, ints2, title, subtitle, filename, max_mz=None):
    """ Creates and saves a mirror plot of two spectra. """
    try:
        # Ensure data is valid for plotting
        if not mzs1 or not ints1: ints1 = [0]; mzs1 = [0]
        if not mzs2 or not ints2: ints2 = [0]; mzs2 = [0]

        # Normalize intensities relative to base peak for visualization
        max_int1 = max(ints1) if ints1 else 1
        norm_ints1 = [i / max_int1 * 100 for i in ints1] if max_int1 > 0 else []

        max_int2 = max(ints2) if ints2 else 1
        norm_ints2 = [-i / max_int2 * 100 for i in ints2] if max_int2 > 0 else [] # Negative for mirror

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot stems
        markerline1, stemlines1, baseline1 = ax.stem(mzs1, norm_ints1, linefmt='b-', markerfmt=' ', basefmt=' ')
        plt.setp(stemlines1, 'linewidth', 1, 'color', 'blue')

        markerline2, stemlines2, baseline2 = ax.stem(mzs2, norm_ints2, linefmt='r-', markerfmt=' ', basefmt=' ')
        plt.setp(stemlines2, 'linewidth', 1, 'color', 'red')

        # Add horizontal line at y=0
        ax.axhline(0, color='black', linewidth=0.5)

        # Formatting
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative Intensity (%)")
        ax.set_ylim(-110, 110) # Provide some padding
        if max_mz:
            ax.set_xlim(0, max_mz)
        else:
            ax.set_xlim(0, max(max(mzs1) if mzs1 else 0, max(mzs2) if mzs2 else 0) * 1.05)

        ax.set_title(title + "\n" + subtitle)

        # Add legend elements manually if needed (e.g., colored lines)
        ax.plot([], [], color='blue', label='Spectrum A')
        ax.plot([], [], color='red', label='Spectrum B')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Error generating plot {filename}: {e}")
        return False

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Visualize experimental variance in mass spectra for the same molecule.")
    parser.add_argument("--input_meta", type=str, required=True,
                        help="Path to the metadata feather file (e.g., augmented_msg_df.feather) containing peaks/mzs/intensities.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the mirror plot images.")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of example pairs to generate for each variance type.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for selecting examples.")
    # --- UPDATED ARGUMENT NAME ---
    parser.add_argument("--peaks_format", type=str, default="object_list", choices=["object_list", "comma_sep"],
                        help="Format of peak data ('object_list' for ndarray/list in 'peaks' col or 'comma_sep' for separate mzs,intensities cols).")


    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)

    # --- 1. Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- Starting Experimental Variance Visualization ---")
    print(f"Loading metadata from: {args.input_meta}")
    print(f"Saving plots to: {args.output_dir}")

    # --- 2. Load Data ---
    start_time = time.time()
    try:
        df = pd.read_feather(args.input_meta)
        print(f"Loaded {len(df):,} records in {time.time() - start_time:.2f}s")

        # --- Check for required columns ---
        required_id_cols = ['spectrum_id', 'inchikey', 'instrument', 'adduct', 'collision_energy']
        required_peak_cols = []
        if args.peaks_format == "object_list": # --- UPDATED ---
            required_peak_cols = ['peaks']
        else: # comma_sep
            required_peak_cols = ['mzs', 'intensities']

        missing_cols = [col for col in required_id_cols + required_peak_cols if col not in df.columns]
        if missing_cols:
             print(f"Error: Input file missing required columns: {missing_cols}")
             return

        # Handle NaNs in key columns BEFORE grouping
        key_cols = ['inchikey', 'instrument', 'adduct', 'collision_energy']
        # Fill NaNs with a placeholder string to treat them as a distinct category
        df[key_cols] = df[key_cols].fillna('Unknown')
        # Ensure energy is comparable (e.g., convert to string or handle floats carefully)
        # Converting float to string is safer for direct comparison
        df['collision_energy'] = df['collision_energy'].astype(str)

    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_meta}. Error: {e}")
        return

    # --- 3. Find Molecules with Multiple Spectra ---
    print("\nGrouping by molecule (InChIKey)...")
    # Drop entries where inchikey is 'Unknown' as they can't be grouped meaningfully
    df_known_mol = df[df['inchikey'] != 'Unknown'].copy()
    grouped_by_mol = df_known_mol.groupby('inchikey')
    
    # Keep groups (molecules) that have at least 2 spectra
    molecules_with_variance = {inchikey: group for inchikey, group in grouped_by_mol if len(group) >= 2}
    print(f"Found {len(molecules_with_variance)} molecules with 2 or more spectra.")

    if not molecules_with_variance:
        print("No molecules found with multiple spectra to compare. Exiting.")
        return

    # --- 4. Find and Visualize Examples ---
    # Define the experimental factors (filters) we want to check
    variance_types = {
        # "Varying Column": ["Constant Column 1", "Constant Column 2"]
        "Instrument": ("instrument", ["adduct", "collision_energy"]),
        "Adduct": ("adduct", ["instrument", "collision_energy"]),
        "Collision_Energy": ("collision_energy", ["instrument", "adduct"])
    }

    examples_found = {v_type: 0 for v_type in variance_types}

    print("\nSearching for variance examples...")
    # Iterate through molecules, shuffling for random examples if needed
    molecule_keys = list(molecules_with_variance.keys())
    random.shuffle(molecule_keys) # Process in random order

    # Iterate through each molecule group
    for inchikey in tqdm(molecule_keys, desc="Processing Molecules"):
        group = molecules_with_variance[inchikey] # Get all spectra for this one molecule
        spectra_indices = group.index.tolist()

        # Check all pairs (A vs B) *within this molecule's group*
        for i in range(len(spectra_indices)):
            for j in range(i + 1, len(spectra_indices)):
                idx1 = spectra_indices[i]
                idx2 = spectra_indices[j]
                spec1 = df.loc[idx1]
                spec2 = df.loc[idx2]

                # Check each variance type (Instrument, Adduct, Energy)
                for v_type, (varying_col, constant_cols) in variance_types.items():
                    # Stop searching for this type if enough examples found
                    if examples_found[v_type] >= args.num_examples:
                        continue

                    # Check if the varying column is different
                    if spec1[varying_col] != spec2[varying_col]:
                        # Check if the other (constant) columns are the same
                        if all(spec1[col] == spec2[col] for col in constant_cols):
                            # --- Found an Example ---
                            # This pair (spec1, spec2) is the SAME molecule,
                            # measured with the SAME (constant_cols),
                            # but DIFFERENT (varying_col).
                            
                            print(f"\nFound example for {v_type} Variance:")
                            print(f"  Molecule InChIKey: {inchikey}")
                            print(f"  Spectrum A: ID={spec1['spectrum_id']}, {varying_col}={spec1[varying_col]}, Const={ {col:spec1[col] for col in constant_cols} }")
                            print(f"  Spectrum B: ID={spec2['spectrum_id']}, {varying_col}={spec2[varying_col]}, Const={ {col:spec2[col] for col in constant_cols} }")

                            # --- UPDATED PARSING LOGIC ---
                            if args.peaks_format == "object_list":
                                mzs1, ints1 = parse_peaks_object(spec1['peaks'])
                                mzs2, ints2 = parse_peaks_object(spec2['peaks'])
                            else: # comma_sep
                                mzs1, ints1 = parse_comma_sep_string(spec1['mzs'], spec1['intensities'])
                                mzs2, ints2 = parse_comma_sep_string(spec2['mzs'], spec2['intensities'])
                            # --- END UPDATED LOGIC ---

                            if mzs1 is None or mzs2 is None:
                                print("  Skipping plot generation: Error parsing peak data.")
                                continue

                            # Calculate Cosine
                            spec_dict1 = process_spectrum_for_cosine(mzs1, ints1)
                            spec_dict2 = process_spectrum_for_cosine(mzs2, ints2)
                            cosine_sim = fast_cosine_similarity(spec_dict1, spec_dict2)
                            print(f"  Calculated Cosine Similarity: {cosine_sim:.4f}")

                            # Generate Plot
                            plot_title = f"{v_type} Variance Example (InChIKey: ...{inchikey[-10:]})"
                            plot_subtitle = (f"Spec A ({spec1['spectrum_id']} - {spec1[varying_col]}) vs "
                                             f"Spec B ({spec2['spectrum_id']} - {spec2[varying_col]}) | "
                                             f"Cosine Sim: {cosine_sim:.4f}")
                            filename_base = f"{v_type}_example_{examples_found[v_type]+1}_{spec1['spectrum_id']}_vs_{spec2['spectrum_id']}.png"
                            output_filepath = os.path.join(args.output_dir, filename_base)

                            plot_success = plot_mirror_spectrum(mzs1, ints1, mzs2, ints2,
                                                                plot_title, plot_subtitle, output_filepath)

                            if plot_success:
                                examples_found[v_type] += 1
                                print(f"  Saved plot to {output_filepath}")

        # Check if we found enough examples for all types
        if all(count >= args.num_examples for count in examples_found.values()):
            print("\nFound sufficient examples for all variance types.")
            break

    # --- 5. Final Summary ---
    print("\n--- Variance Visualization Complete ---")
    for v_type, count in examples_found.items():
        print(f"Generated {count}/{args.num_examples} examples for {v_type} variance.")
    print(f"Plots saved in: {args.output_dir}")


# --- Standalone Execution Block ---
if __name__ == '__main__':
    main()