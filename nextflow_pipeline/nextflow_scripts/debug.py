import pandas as pd
import numpy as np
import argparse
import sys
import traceback

# Try importing MatchMS
try:
    from matchms import Spectrum
    from matchms.similarity import ModifiedCosine
    print("MatchMS imported successfully.")
except ImportError:
    print("CRITICAL: MatchMS library not found!")
    sys.exit(1)

def debug_one_block(input_file):
    print(f"\n--- DEBUGGING FILE: {input_file} ---")
    
    # 1. Load Data
    try:
        df = pd.read_feather(input_file).head(50)
        print(f"Loaded {len(df)} rows for inspection.")
    except Exception as e:
        print(f"CRITICAL: Read failed. {e}")
        return

    # 2. Conversion Test
    print("\n--- TESTING CONVERSION LOGIC (NUCLEAR FIX) ---")
    spectra_objs = []
    
    for i, r in df.iterrows():
        print(f"Processing Row {i}...")
        
        # A. Check Precursor M/Z
        pmz = r.get('precursor_mz')
        if pmz is None or pd.isna(pmz):
            print(f"  -> SKIPPED: Precursor M/Z is missing/NaN")
            continue
            
        # B. Check Peaks
        p_raw = r['peaks']
        try:
            # --- THE FIX ---
            # 1. Force outer structure to list
            clean_list = list(p_raw) if hasattr(p_raw, '__len__') else None
            
            if clean_list is None:
                print(f"  -> SKIPPED: Peaks not iterable (Got {type(p_raw)})")
                continue
            
            # 2. Force conversion to 2D Float64 Matrix
            # This allocates NEW memory, solving the "pointer array" issue
            peaks_arr = np.array(clean_list, dtype=np.float64)

            # 3. Validate Shape
            if peaks_arr.ndim != 2 or peaks_arr.shape[1] < 2:
                print(f"  -> SKIPPED: Invalid shape {peaks_arr.shape}")
                continue

            # 4. Extract Columns & FORCE CONTIGUOUS MEMORY
            # This is what MatchMS demands
            mz = np.ascontiguousarray(peaks_arr[:, 0])
            intensity = np.ascontiguousarray(peaks_arr[:, 1])
            
            # SQRT
            intensity = np.sqrt(intensity)
            
            # Create Object
            meta = {
                'spectrum_id': str(r['spectrum_id']),
                'precursor_mz': float(pmz)
            }
            spec = Spectrum(mz=mz, intensities=intensity, metadata=meta)
            spectra_objs.append(spec)
            print(f"  -> Success: MatchMS Object created. (Peaks: {len(mz)})")
            
        except Exception as e:
            print(f"  -> FAILED: {e}")
            # print(traceback.format_exc()) # Uncomment if you need deeper trace
            continue

    print(f"\nTotal Valid Spectra Created: {len(spectra_objs)}")
    
    if len(spectra_objs) < 2:
        print("CRITICAL: Not enough valid spectra to calculate similarity.")
        return

    # 3. Similarity Test
    print("\n--- TESTING SIMILARITY CALCULATION ---")
    try:
        mc = ModifiedCosine(tolerance=0.05)
        
        # Compare first 2 spectra
        print("Calculating pair score...")
        pair_score = mc.pair(spectra_objs[0], spectra_objs[1])
        
        if pair_score is None:
            print("Result: None (No matches found within tolerance)")
        else:
            print(f"Result: {pair_score['score']}")
            
        print("SUCCESS! The logic is valid.")
        
    except Exception as e:
        print(f"CRITICAL: Similarity calculation failed. {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    args = parser.parse_args()
    
    debug_one_block(args.input_file)