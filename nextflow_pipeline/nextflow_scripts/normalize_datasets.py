import pandas as pd
import numpy as np
import argparse
import os
import logging

# --- CONFIGURATION ---
LOG_FILE = "normalization_log.txt"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )

def normalize_instrument(val):
    """Maps raw instrument strings to 4 Standard Categories."""
    s = str(val).upper().strip()
    
    # Priority 1: Orbitrap (High Res, Trap/FT)
    if any(x in s for x in ['ORBITRAP', 'EXACTIVE', 'EXPLORIS', 'HYBRID FT', 'ID-X']):
        return 'Orbitrap'
    # Priority 2: QTOF (High Res, TOF)
    if any(x in s for x in ['QTOF', 'Q-TOF', 'Q TOF', 'TRIPLETOF', 'SYNAPT', 'XEVO', 'MAXIS', 'MICROTOF', 'OTOF', 'TOF']):
        return 'QTOF'
    # Priority 3: Triple Quad (Low Res, Beam)
    if any(x in s for x in ['QQQ', 'TRIPLE QUAD', 'API', 'QUATTRO', 'TSQ', 'QTRAP', 'SCIENTIFIC Q-EXACTIVE FOCUS']): 
        return 'Triple Quad'
    # Priority 4: Ion Trap (Low Res, Trap)
    if any(x in s for x in ['ION TRAP', 'LTQ', 'LCQ', 'ESQUIRE', 'IT-TOF']):
        return 'Ion Trap'
        
    return 'Unknown'

def normalize_ce(val):
    """Snaps Collision Energy to nearest 5 eV."""
    try:
        if pd.isna(val) or val < 0: return -1.0
        # Logic: 18, 19, 20, 21, 22 -> All become 20.0
        return float(round(val / 5.0) * 5)
    except: return -1.0

def normalize_adduct(val):
    """Standardizes adduct string."""
    s = str(val).strip().upper()
    if not s or s == 'NONE' or s == 'NAN': return 'UNKNOWN'
    return s

def process_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    logging.info(f"Processing {filename}...")
    
    try:
        df = pd.read_feather(file_path)
        
        # 1. Normalize Instruments
        if 'instrument_type' in df.columns:
            df['instrument_type'] = df['instrument_type'].apply(normalize_instrument)
        elif 'instrument' in df.columns:
             df['instrument_type'] = df['instrument'].apply(normalize_instrument)
        else:
            df['instrument_type'] = 'Unknown'
            
        # 2. Normalize Collision Energy
        if 'collision_energy' in df.columns:
            df['collision_energy'] = df['collision_energy'].apply(normalize_ce)
        else:
            df['collision_energy'] = -1.0

        # 3. Normalize Adducts
        if 'adduct' in df.columns:
            df['adduct'] = df['adduct'].apply(normalize_adduct)
        else:
             df['adduct'] = 'UNKNOWN'

        # Save
        output_path = os.path.join(output_dir, f"normalized_{filename}")
        df.to_feather(output_path)
        logging.info(f"Saved normalized file to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs='+', required=True, help="List of feather files to process")
    parser.add_argument("--output_dir", required=True, help="Directory to save normalized files")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    for f in args.input_files:
        process_file(f, args.output_dir)