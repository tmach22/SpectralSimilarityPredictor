import argparse
import os
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from matchms.importing import load_from_mgf, load_from_msp, load_from_mzml
from matchms import Spectrum
from matchms.filtering import select_by_relative_intensity
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from pyteomics import mzml as py_mzml 
import re

# --- Helper: Salt Removal ---
def clean_smiles_string(smiles):
    if not smiles or not isinstance(smiles, str): return None
    if '.' in smiles:
        parts = smiles.split('.')
        return max(parts, key=len)
    return smiles

# --- Helper: RDKit Calculations ---
def get_rdkit_properties(smiles):
    if not smiles: return None, None, None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None, None, None, None
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_fp = fp_bit.ToBitString()
        formula = rdMolDescriptors.CalcMolFormula(mol)
        parent_mass = Descriptors.ExactMolWt(mol)
        inchikey = Chem.MolToInchiKey(mol)
        return morgan_fp, formula, parent_mass, inchikey
    except: return None, None, None, None

# --- Helper: Extract CE and Polarity Manually from mzML ---
def map_raw_metadata(file_path):
    """
    Scans raw mzML using pyteomics to map Scan Number -> {CE, Polarity}.
    Crucial for determining Positive Mode in raw files.
    """
    meta_map = {}
    try:
        with py_mzml.read(file_path) as reader:
            for scan in reader:
                # We typically only care about MS2 for the dataset, but extraction is cheap
                scan_id = scan.get('id', '')
                scan_num = None
                
                # 1. Parse Scan Number
                for part in scan_id.split():
                    if part.startswith('scan='):
                        scan_num = part.split('=')[1]
                        break
                
                if not scan_num: continue

                # 2. Parse Collision Energy
                ce = None
                if 'precursorList' in scan:
                    try:
                        activation = scan['precursorList']['precursor'][0]['activation']
                        for key in ['collision energy', 'collision energy ramp start']:
                            if key in activation:
                                ce = float(activation[key])
                                break
                    except: pass
                
                # 3. Parse Polarity / Ion Mode
                # Strategy A: Explicit CV Param 'positive scan' (Accession MS:1000130)
                is_positive = 'positive scan' in scan
                is_negative = 'negative scan' in scan
                
                # Strategy B: Filter String (e.g., "FTMS + c ESI...")
                if not is_positive and not is_negative:
                    filter_str = scan.get('scanList', {}).get('scan', [{}])[0].get('filter string', '')
                    if not filter_str and 'filter string' in scan:
                         filter_str = scan['filter string']
                    
                    if filter_str:
                        if '+ ' in str(filter_str): is_positive = True
                        if '- ' in str(filter_str): is_negative = True

                # Determine final mode string
                ion_mode = 'unknown'
                if is_positive: ion_mode = 'positive'
                elif is_negative: ion_mode = 'negative'

                meta_map[scan_num] = {'ce': ce, 'ion_mode': ion_mode}

    except Exception as e:
        print(f"  Warning: Could not map metadata for {os.path.basename(file_path)}: {e}")
    
    return meta_map

# --- 1. CASMI Key Loader ---
def load_casmi_key(csv_path):
    print(f"Loading CASMI Key from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=1)
        lookup = {}
        for _, row in df.iterrows():
            try:
                fname = str(row['File']).strip()
                mz = float(row['Precursor m/z (Da)'])
                rounded_mz = round(mz, 1) 
                
                info = {
                    'smiles': row['SMILES'],
                    'adduct': row['Adduct']
                }
                lookup[(fname, rounded_mz)] = info
            except: continue
        print(f"  Loaded {len(lookup)} Ground Truth entries.")
        return lookup
    except Exception as e:
        print(f"  ERROR loading Key: {e}")
        return {}

# --- Loaders (XML, JSON) ---
def parse_hmdb_xml(xml_path):
    context = ET.iterparse(xml_path, events=("end",))
    current_data = {}
    for event, elem in context:
        tag = elem.tag.split('}')[-1]
        if tag == 'database_id': current_data['id'] = elem.text
        elif tag == 'smiles': current_data['smiles'] = elem.text
        elif tag == 'precursor_mz':
            try: current_data['precursor_mz'] = float(elem.text)
            except: pass
        elif tag == 'instrument_type': current_data['instrument'] = elem.text
        elif tag == 'peak_list':
            mz, intensities = [], []
            for peak in elem:
                for sub in peak:
                    if sub.tag.endswith('mass_charge'):
                        try: mz.append(float(sub.text))
                        except: pass
                    elif sub.tag.endswith('intensity'):
                        try: intensities.append(float(sub.text))
                        except: pass
            if mz:
                # HMDB is usually [M+H]+ but strictly speaking we should check.
                # Assuming positive for now as HMDB MS/MS is mostly positive.
                meta = {
                    'id': current_data.get('id'), 
                    'smiles': current_data.get('smiles'), 
                    'precursor_mz': current_data.get('precursor_mz'), 
                    'instrument_type': current_data.get('instrument'), 
                    'adduct': '[M+H]+',
                    'ion_mode': 'positive' # HMDB Default
                }
                yield Spectrum(mz=np.array(mz), intensities=np.array(intensities), metadata=meta)
            current_data = {}
            elem.clear()

def parse_mona_json(file_path):
    with open(file_path, 'r') as f: data = json.load(f)
    
    for entry in data:
        # 1. Parse Metadata first
        meta_dict = {item['name'].lower(): item['value'] for item in entry.get('metaData', [])}
        
        # --- FILTER 1: IONIZATION MODE ---
        # Look for keys: 'ionization mode', 'ion mode'
        mode = meta_dict.get('ionization mode') or meta_dict.get('ion mode') or 'unknown'
        if 'neg' in mode.lower():
            continue # SKIP NEGATIVE MODE

        # --- FILTER 2: STRICT MS2 CHECK ---
        ms_level = meta_dict.get('ms level')
        if ms_level and ms_level != 'MS2':
            continue

        # --- FILTER 3: PRECURSOR M/Z ---
        pmz = meta_dict.get('precursor m/z')
        if not pmz: continue 
        try: pmz = float(pmz)
        except: continue
        
        spec_str = entry.get('spectrum', '')
        if not spec_str: continue
        
        mz, intensities = [], []
        try:
            for p in spec_str.split(' '):
                if ':' in p:
                    m, i = p.split(':')
                    mz.append(float(m))
                    intensities.append(float(i))
        except: continue
        
        if not mz: continue

        # Sort Peaks
        pairs = sorted(zip(mz, intensities), key=lambda x: x[0])
        mz, intensities = zip(*pairs)
        
        # Compound parsing
        compounds = entry.get('compound', [])
        smiles, inchikey = None, None
        if compounds:
            cmp = compounds[0]
            inchikey = cmp.get('inchiKey')
            for m in cmp.get('metaData', []):
                if m['name'] == 'SMILES': smiles = m['value']; break
        
        if not smiles: smiles = meta_dict.get('smiles')
            
        metadata = {
            'source_id': entry.get('id'), 
            'smiles': smiles, 
            'inchikey': inchikey, 
            'precursor_mz': pmz, 
            'adduct': meta_dict.get('precursor type'), 
            'instrument_type': meta_dict.get('instrument type'),
            'instrument': meta_dict.get('instrument'), 
            'collision_energy': meta_dict.get('collision energy'),
            'ion_mode': 'positive' # We filtered out negatives above
        }
        
        yield Spectrum(mz=np.array(mz), intensities=np.array(intensities), metadata=metadata)

# --- Main Loader Logic ---
def load_data(input_dir, fmt, casmi_key=None):
    spectra = []
    print(f"Scanning {input_dir} recursively for {fmt} files...")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Standard Loaders
                if fmt == 'mgf' and file.endswith('.mgf'): spectra.extend(list(load_from_mgf(file_path)))
                elif fmt == 'msp' and (file.endswith('.msp') or file.endswith('.txt')): spectra.extend(list(load_from_msp(file_path)))
                elif fmt == 'xml' and file.endswith('.xml'): 
                    for s in parse_hmdb_xml(file_path): spectra.append(s)
                elif fmt == 'json' and file.endswith('.json'): 
                    for s in parse_mona_json(file_path): spectra.append(s)
                
                # mzML Loader
                elif (fmt == 'mzml' or fmt == 'mgf') and file.endswith('.mzML'):
                    
                    # 1. Map Metadata (CE + Polarity)
                    raw_meta_map = map_raw_metadata(file_path)
                    
                    # 2. Load Spectra (FILTERED FOR MS2)
                    loaded = list(load_from_mzml(file_path, ms_level=2))
                    
                    if casmi_key:
                        fname = os.path.splitext(file)[0]
                        for s in loaded:
                            # A. Label from Key
                            pmz = s.get('precursor_mz')
                            if pmz:
                                info = casmi_key.get((fname, round(float(pmz), 1)))
                                if info: 
                                    s.set('smiles', info['smiles'])
                                    s.set('adduct', info['adduct'])
                                    s.set('dataset_origin', 'CASMI_2022')
                                    s.set('instrument_type', 'Thermo Q Exactive HF')
                                    s.set('instrument', 'Thermo Q Exactive HF') 
                            
                            # B. Inject Raw Metadata (CE + Polarity)
                            scan_num = s.get('scan_number')
                            if not scan_num:
                                title = str(s.get('title', ''))
                                if 'scan=' in title:
                                    scan_num = title.split('scan=')[1].split()[0].strip('"')
                            
                            if scan_num and str(scan_num) in raw_meta_map:
                                meta = raw_meta_map[str(scan_num)]
                                
                                # Inject CE if missing
                                if not s.get('collision_energy'):
                                    s.set('collision_energy', meta['ce'])
                                
                                # Inject Ion Mode
                                s.set('ion_mode', meta['ion_mode'])

                    spectra.extend(loaded)
            except Exception as e: print(f"Skipping {file}: {e}")
    return spectra

# --- Main Processing Logic ---
def process_spectrum(spectrum, min_peaks, dataset_name, sequential_index, debug=False):
    if spectrum is None: return None
    
    # --- FILTER 4: POSITIVE MODE CHECK ---
    # 1. Check Explicit Ion Mode (mapped from mzML or parsed from JSON)
    mode = spectrum.get('ion_mode')
    if mode and 'neg' in str(mode).lower():
        return None # Explicitly Negative
    
    # 2. Check Adduct (Fallback)
    # If we don't know the mode, we check the adduct.
    # If adduct ends in '-', we assume negative.
    adduct = spectrum.get('adduct')
    if adduct:
        adduct_str = str(adduct).strip()
        if adduct_str.endswith('-') or ']2-' in adduct_str or ']3-' in adduct_str:
            return None # Adduct indicates negative

    # 1. Intensity Filters
    try:
        spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001) 
        if len(spectrum.peaks.intensities) > 0:
            max_i = np.max(spectrum.peaks.intensities)
            if max_i > 0:
                new_i = spectrum.peaks.intensities / max_i
                spectrum = Spectrum(mz=spectrum.peaks.mz, intensities=new_i, metadata=spectrum.metadata)
    except: return None
    
    if len(spectrum.peaks.mz) < min_peaks: return None
    
    # 2. Chemical Metadata
    raw_smiles = spectrum.get('smiles')
    clean_smiles = clean_smiles_string(raw_smiles)
    if not clean_smiles: return None 
    
    morgan_fp, formula, parent_mass, inchikey = get_rdkit_properties(clean_smiles)
    if not morgan_fp: return None 
    
    # 3. SEQUENTIAL ID
    spec_id = f"{dataset_name}_{sequential_index}"
    
    peaks_list = [[float(mz), float(i)] for mz, i in zip(spectrum.peaks.mz, spectrum.peaks.intensities)]
    inst = spectrum.get('instrument') or spectrum.get('instrument_type') or 'Unknown'
    
    # 4. Collision Energy (Regex Cleanup)
    ce_raw = spectrum.get('collision_energy') or spectrum.get('collision energy')
    ce = None
    if ce_raw:
        # Regex to find float/int. Handles "35HCD", "20 eV"
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(ce_raw))
        if match:
            try: 
                val = float(match.group())
                # Filter negative Collision Energies immediately
                if val >= 0:
                    ce = val
            except: ce = None

    fold = 'train'
    is_sim = False
    if 'CASMI' in dataset_name:
        fold = 'holdout' 
        is_sim = True 
    
    return {
        'spectrum_id': spec_id,
        'peaks': peaks_list,
        'precursor_mz': float(spectrum.get('precursor_mz')) if spectrum.get('precursor_mz') else None,
        'smiles': clean_smiles,
        'adduct': adduct, 
        'instrument': inst,
        'morgan_fingerprint': morgan_fp,
        'inchikey': inchikey,
        'formula': formula,
        'precursor_formula': None,
        'parent_mass': float(parent_mass) if parent_mass else None,
        'instrument_type': inst,
        'collision_energy': ce,
        'fold': fold,
        'simulation_challenge': is_sim
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_peaks", type=int, default=10)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--casmi_key_path", type=str, default=None)
    args = parser.parse_args()

    casmi_key = None
    if args.casmi_key_path and os.path.exists(args.casmi_key_path):
        casmi_key = load_casmi_key(args.casmi_key_path)

    raw_spectra = load_data(args.input_dir, args.format, casmi_key)
    print(f"Found {len(raw_spectra)} raw spectra.")
    
    clean_data = []
    # Start ID index at 1
    for i, s in enumerate(tqdm(raw_spectra, desc="Processing")):
        processed = process_spectrum(s, args.min_peaks, args.dataset_name, i+1, debug=True)
        if processed:
            clean_data.append(processed)
            
    if clean_data:
        df = pd.DataFrame(clean_data)
        cols = ['spectrum_id', 'peaks', 'precursor_mz', 'smiles', 'adduct', 'instrument', 
                'morgan_fingerprint', 'inchikey', 'formula', 'precursor_formula', 
                'parent_mass', 'instrument_type', 'collision_energy', 'fold', 'simulation_challenge']
        for c in cols:
            if c not in df.columns: df[c] = None
        
        df = df[cols] 
        df.to_feather(args.output)
        print(f"Saved {len(df)} cleaned records to {args.output}")
    else:
        print("No valid spectra found.")