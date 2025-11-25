import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def analyze_false_positives(args):
    """
    Loads model predictions and original test data to analyze the distribution
    of true cosine similarity scores for False Positives.
    """
    # Set the classification threshold used during training/testing
    CLASSIFICATION_THRESHOLD = 0.7
    
    print("--- Starting False Positive Distribution Analysis ---")
    
    # 1. Load data
    try:
        print(f"Loading prediction results from: {args.results_csv_path}")
        df_results = pd.read_csv(args.results_csv_path)

        print(f"Loading original test pairs from: {args.test_feather_path}")
        df_pairs = pd.read_feather(args.test_feather_path)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: One of the required files was not found.")
        print(f"Please check the paths. Error: {e}")
        return

    # 2. Merge data on index (safe if created sequentially)
    # Reset index to ensure a clean merge if source files had different indices
    df_merged = pd.concat([df_pairs.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)

    # 3. Filter for False Positives (FP)
    # FP definition: Predicted Positive (1.0) AND True Label is Negative (0.0)
    df_fp = df_merged[
        (df_merged['predicted_label'] == 1.0) &
        (df_merged['true_label'] == 0.0)
    ].copy()

    total_fp = len(df_fp)
    print(f"\nTotal False Positives (FP) found: {total_fp}")
    
    if total_fp == 0:
        print("No False Positives to analyze. Analysis complete.")
        return

    # 4. Plot the distribution of true cosine_similarity for FPs
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_plot_path = os.path.join(output_dir, "false_positive_similarity_distribution.png")

    plt.figure(figsize=(10, 6))

    # Plot histogram of true cosine similarities for FPs
    plt.hist(
        df_fp['cosine_similarity'],
        bins=30,  # Higher bin count for fine-grained analysis
        range=(0.0, CLASSIFICATION_THRESHOLD),
        edgecolor='black',
        alpha=0.8,
        color='#CC5500' # Orange-Red for FPs
    )

    # Add the true classification threshold line for context
    plt.axvline(
        CLASSIFICATION_THRESHOLD,
        color='gray',
        linestyle='--',
        linewidth=2,
        label=f'True Classification Threshold ({CLASSIFICATION_THRESHOLD})'
    )

    # Customize plot
    plt.title('Distribution of True Cosine Similarity for False Positives (FP)')
    plt.xlabel('True Cosine Similarity (Must be < 0.7)')
    plt.ylabel('Count of False Positives (FP)')
    plt.grid(axis='y', alpha=0.4)
    plt.legend()
    
    plt.savefig(output_plot_path)
    plt.close()

    print(f"Plot saved to: {output_plot_path}")

    # 5. Calculate and print summary statistics
    print("\n--- False Positive (FP) Summary Statistics (True Similarity) ---")
    fp_summary = df_fp['cosine_similarity'].describe()
    print(fp_summary)

    # Check concentration near the threshold (e.g., 0.6 to 0.7)
    fp_near_threshold = len(df_fp[(df_fp['cosine_similarity'] >= 0.6) & (df_fp['cosine_similarity'] < 0.7)])
    print(f"\nFPs with True Similarity in [0.6, 0.7): {fp_near_threshold} ({fp_near_threshold/total_fp*100:.2f}% of all FPs)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze False Positive distribution based on true similarity scores.")
    parser.add_argument("--results_csv_path", type=str, required=True, help="Path to the model's test_results.csv file.")
    parser.add_argument("--test_feather_path", type=str, required=True, help="Path to the original binary test set feather file.")
    parser.add_argument("--output_dir", type=str, default="./analysis_plots", help="Directory to save the resulting plot.")

    args = parser.parse_args()
    analyze_false_positives(args)