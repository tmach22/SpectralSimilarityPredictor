import pandas as pd
import glob
import os
from tqdm import tqdm

input_dir = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/brute_force/"
output_file = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/spectra_pairs/brute_force_combined.feather"

# Find all the .parquet files
file_list = glob.glob(os.path.join(input_dir, "block_*.parquet"))
print(f"Found {len(file_list)} block files to combine.")

all_dfs = []
for f in tqdm(file_list, desc="Loading blocks"):
    all_dfs.append(pd.read_parquet(f))

print("Combining all blocks...")
final_df = pd.concat(all_dfs, ignore_index=True)

print(f"Total pairs found: {len(final_df):,}")
print("Saving final .feather file...")
final_df.to_feather(output_file)
print("Done.")