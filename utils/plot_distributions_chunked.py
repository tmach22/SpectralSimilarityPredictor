import pandas as pd
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from tqdm import tqdm

def main(args):
    print(f"--- Distributed Distribution Plotting ---")
    
    # 1. Setup
    input_dir = args.input_dir
    search_pattern = os.path.join(input_dir, "*.feather")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No feather files found in {input_dir}")
        return
        
    print(f"Found {len(files)} chunks. Starting streaming analysis...")
    print(f"Sampling Rate: {args.sample_rate * 100}% (for visualization)")

    # 2. Streaming Aggregation
    global_sum = 0.0
    global_count = 0
    sampled_batches = []
    
    # We only need the similarity column
    # If your files have 'cosine_similarity', ensure we read only that if possible
    # Feather reads usually load the full table, but we drop others immediately.
    
    for file_path in tqdm(files, desc="Scanning Chunks"):
        try:
            # Load file
            df_chunk = pd.read_feather(file_path)
            
            # Extract Similarity Column (numpy array is lighter)
            if 'cosine_similarity' not in df_chunk.columns:
                continue
                
            sims = df_chunk['cosine_similarity'].to_numpy()
            
            # Update Global Stats (Exact Mean Calculation)
            global_count += len(sims)
            global_sum += np.sum(sims)
            
            # Subsample for Plotting (Reservoir Sampling)
            # We take a random subset based on the sample rate
            # If sample_rate is 0.01, we keep 1% of this chunk
            if args.sample_rate > 0:
                # Calculate how many to keep
                n_keep = int(len(sims) * args.sample_rate)
                if n_keep > 0:
                    # Random choice without replacement is expensive, 
                    # straightforward slicing or mask is faster for vis
                    mask = np.random.rand(len(sims)) < args.sample_rate
                    sampled_sims = sims[mask]
                    sampled_batches.append(sampled_sims)
            
            # Cleanup immediately
            del df_chunk, sims
            # Garbage collect occasionally to prevent fragmentation
            if len(sampled_batches) % 50 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    # 3. Final Calculations
    if global_count == 0:
        print("No data found.")
        return

    exact_mean = global_sum / global_count
    print(f"\n--- Statistics (Exact) ---")
    print(f"Total Pairs: {global_count:,}")
    print(f"Global Mean: {exact_mean:.4f}")

    # 4. Prepare Plotting Data
    print("\nConcatenating samples for plot...")
    if sampled_batches:
        plot_data = np.concatenate(sampled_batches)
        print(f"Plotting Data Size: {len(plot_data):,} samples (Proxy for full distribution)")
    else:
        print("No samples collected. Increase sample_rate.")
        return

    # 5. Plotting
    print(f"Generating plot... saving to {args.output_image}")
    
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".15"})
    plt.figure(figsize=(14, 8))

    # We use the sampled data for the histogram
    # Since it's a random sample, the SHAPE of the distribution is preserved perfectly.
    sns.histplot(
        x=plot_data,
        bins=50,             # Increased bins for better resolution on large data
        binrange=(0.0, 1.0),
        color="blue",
        element="bars",
        fill=True,
        alpha=0.6,
        edgecolor="#333",
        stat="density"       # Normalize to density so Y-axis isn't dependent on sample size
    )
    
    # Add the EXACT mean line (not the sample mean)
    plt.axvline(
        exact_mean, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'True Mean: {exact_mean:.3f}'
    )
    
    plt.title(
        f"Cosine Similarity Distribution (Sampled)\nTotal Population: {global_count:,} | Plot Sample: {len(plot_data):,}", 
        fontsize=16
    )
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.xlim(-0.05, 1.05)
    
    try:
        plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing cleaned feather chunks.")
    parser.add_argument("--output_image", type=str, required=True,
                        help="Path to save the output .png file.")
    parser.add_argument("--sample_rate", type=float, default=0.01, 
                        help="Fraction of data to load for plotting (default 0.01 = 1%).")

    args = parser.parse_args()
    main(args)