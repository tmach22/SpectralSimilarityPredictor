import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def create_stratified_plot(results_path, output_path, title, n_samples_per_bin=10):
    """
    Loads model results and generates a stratified scatter plot showing a sample
    of low, medium, and high similarity pairs.
    """
    print(f"Loading results from: {results_path}")
    try:
        data = np.load(results_path)
        predictions = data['predictions']
        ground_truth = data['ground_truth']
    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found.")
        return

    print(f"Loaded {len(predictions)} total data points.")

    # Combine into a pandas DataFrame for easy filtering and sampling
    df = pd.DataFrame({
        'ground_truth': ground_truth,
        'prediction': predictions
    })

    # --- Define Similarity Bins ---
    bins = {
        'Low Similarity (0.0-0.3)': df[(df['ground_truth'] >= 0.0) & (df['ground_truth'] < 0.3)],
        'Medium Similarity (0.3-0.7)': df[(df['ground_truth'] >= 0.3) & (df['ground_truth'] < 0.7)],
        'High Similarity (0.7-1.0)': df[(df['ground_truth'] >= 0.7) & (df['ground_truth'] <= 1.0)]
    }

    # --- Sample from Each Bin ---
    sampled_dfs = []
    print("\nSampling data points from each similarity range:")
    for name, bin_df in bins.items():
        # Ensure we don't try to sample more points than available
        num_to_sample = min(n_samples_per_bin, len(bin_df))
        if len(bin_df) > 0:
            sampled_dfs.append(bin_df.sample(n=num_to_sample, random_state=44))
            # sampled_dfs.append(bin_df.sample(n=num_to_sample, random_state=30))
            print(f" - Sampled {num_to_sample} points from '{name}' (out of {len(bin_df)} available)")
        else:
            print(f" - No points available in '{name}' range.")

    if not sampled_dfs:
        print("No data points to plot after sampling. Exiting.")
        return
        
    # --- Create the Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define colors for each category
    colors = {
        'Low Similarity (0.0-0.3)': 'red',
        'Medium Similarity (0.3-0.7)': 'red',
        'High Similarity (0.7-1.0)': 'red'
    }

    # Plot each sampled group with a different color
    for i, (name, bin_df) in enumerate(bins.items()):
        if not sampled_dfs[i].empty:
            ax.scatter(
                sampled_dfs[i]['ground_truth'], 
                sampled_dfs[i]['prediction'], 
                alpha=0.7,
                s=50,  # Make markers a bit larger
                c=colors[name],
                # label=name
            )

    x = np.linspace(0, 1.1, 100)
    y = x

    # Plot the "perfect prediction" line (y=x)
    ax.plot(x, y, 'b--', label='Perfect Prediction (y=x)') # Changed to red for better contrast
    ax.figsize = (20, 20)

    # --- Styling and Labels ---
    ax.set_xlabel('Ground Truth Similarity', fontsize=12)
    ax.set_ylabel('Predicted Similarity', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # ax.set_aspect('equal', adjustable='box')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Create a more descriptive legend
    ax.legend(title='Similarity Bins', frameon=True, facecolor='white', framealpha=0.8)

    # --- Save and Show ---
    print(f"\nSaving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully.")
    
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a stratified scatter plot of model predictions.")
    
    parser.add_argument("--results_path", type=str, required=True, 
                        help="Path to the.npz file containing prediction results.")
    parser.add_argument("--output_path", type=str, default="stratified_similarity_plot.png", 
                        help="Path to save the output plot image.")
    parser.add_argument("--title", type=str, default="Model Performance on Sampled Data Points", 
                        help="Title for the plot.")
    parser.add_argument("--samples_per_bin", type=int, default=10,
                        help="Number of data points to sample from each similarity bin.")

    args = parser.parse_args()
    create_stratified_plot(args.results_path, args.output_path, args.title, args.samples_per_bin)