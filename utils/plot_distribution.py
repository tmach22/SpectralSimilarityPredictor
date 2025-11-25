import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    """
    Loads the 5-bin balanced dataset and plots the
    COMBINED cosine similarity distribution.
    """
    print(f"--- Starting Combined 5-Bin Distribution Plotting ---")
    
    # --- 1. Load Data ---
    print(f"Loading balanced dataset from: {args.input_file}")
    try:
        df = pd.read_feather(args.input_file)
    except Exception as e:
        print(f"Fatal Error: Could not read input file {args.input_file}. Error: {e}")
        return
        
    n_total = len(df)
    print(f"Loaded {n_total:,} total pairs.")
    
    # --- 2. Calculate Statistics ---
    mean_val = df['cosine_similarity'].mean()
    
    # --- 3. Create and Save the Plot ---
    print(f"\nGenerating combined plot... saving to {args.output_image}")
    
    # Set the style to match your example plot
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".15"})
    plt.figure(figsize=(14, 8)) # A slightly wider figure for clarity

    # Use histplot to create the bars and the KDE (smooth line)
    # We use 50 bins (10 per 0.2-width section) to show the "step"
    # and the internal distribution.
    sns.histplot(
        data=df,
        x="cosine_similarity",
        # binwidth=0.1,
        # binrange=(0.0, 1.0),
        bins=15,
        # kde=True,        # This adds the smooth blue line
        color="blue",
        element="bars",
        fill=True,
        alpha=0.6,
        edgecolor="#333" # Add a faint edge to the bars
    )
    
    # Add the red mean line, just like in your plot
    plt.axvline(
        mean_val, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {mean_val:.3f}'
    )
    
    # Set titles and labels to match your example
    plt.title(
        f"Cosine Similarity Distribution\nFinal Combined Dataset (n={n_total:,})", 
        fontsize=16
    )
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    
    # Set the x-axis limits
    plt.xlim(-0.05, 1.05)
    
    # Save the figure
    try:
        plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot: {e}")

# --- Standalone Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the combined cosine similarity distribution for the 5-bin dataset.")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to your BALANCED 5-BIN dataset (e.g., balanced_5bin_dataset.feather).")
    parser.add_argument("--output_image", type=str, required=True,
                        help="Path to save the output.png file (e.g., combined_5bin_distribution.png).")

    args = parser.parse_args()
    main(args)