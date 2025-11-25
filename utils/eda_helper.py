import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr # Added spearman

def calculate_and_log_metrics(df, taxon_name, summary_file, is_inter=False):
    """
    Calculates key EDA metrics for a given dataframe and logs them to a file.
    """
    header = f"\n==================================================\n"
    header += f"EDA METRICS FOR: {taxon_name}\n"
    header += f"==================================================\n"
    print(header)
    summary_file.write(header)
    
    if df.empty:
        report = "  Count: 0\n"
        print(report)
        summary_file.write(report)
        return

    # 1. Count
    count = len(df)
    report = f"  Count: {count:,}\n"
    
    # 2. Cosine Metrics
    # Replace inf/-inf with NaN, then drop NaNs for stats
    cosine_stats_data = df['cosine_similarity'].replace([np.inf, -np.inf], np.nan).dropna()
    if not cosine_stats_data.empty:
        mean_cosine = cosine_stats_data.mean()
        median_cosine = cosine_stats_data.median()
        std_cosine = cosine_stats_data.std()
        report += f"  Cosine Similarity (Mean / Median / Std): {mean_cosine:.4f} / {median_cosine:.4f} / {std_cosine:.4f} (on {len(cosine_stats_data):,} clean rows)\n"
    else:
        report += "  Cosine Similarity: N/A (no valid data)\n"

    # 3. Tanimoto Metrics (only if is_inter is True)
    if is_inter:
        tanimoto_stats_data = df['tanimoto'].replace([np.inf, -np.inf], np.nan).dropna()
        if not tanimoto_stats_data.empty:
            mean_tanimoto = tanimoto_stats_data.mean()
            median_tanimoto = tanimoto_stats_data.median()
            std_tanimoto = tanimoto_stats_data.std()
            report += f"  Tanimoto Similarity (Mean / Median / Std): {mean_tanimoto:.4f} / {median_tanimoto:.4f} / {std_tanimoto:.4f} (on {len(tanimoto_stats_data):,} clean rows)\n"
        else:
            report += "  Tanimoto Similarity: N/A (no valid data)\n"
        
        # 4. Correlation (only if is_inter is True)
        try:
            # .corr() can fail on all-NaN slices, so we drop NaNs first
            clean_df = df[['tanimoto', 'cosine_similarity']].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_df) > 50: # Need enough points for correlation
                pearson_corr, _ = pearsonr(clean_df['tanimoto'], clean_df['cosine_similarity'])
                spearman_corr, _ = spearmanr(clean_df['tanimoto'], clean_df['cosine_similarity'])
                report += f"  Tanimoto vs. Cosine Correlation (Pearson / Spearman): {pearson_corr:.4f} / {spearman_corr:.4f} (on {len(clean_df):,} clean rows)\n"
            else:
                report += "  Tanimoto vs. Cosine Correlation: N/A (not enough data)\n"
        except Exception as e:
            report += f"  Tanimoto vs. Cosine Correlation: ERROR ({e})\n"
            
    print(report)
    summary_file.write(report)

def plot_distributions(df, taxon_name, plot_dir, is_inter=False):
    """
    Plots and saves the Cosine distribution and (if is_inter) the Tanimoto vs. Cosine heatmap.
    """
    if df.empty:
        print(f"  [Plotting] Skipping {taxon_name} (empty dataframe).")
        return
        
    print(f"  [Plotting] Generating plots for {taxon_name}...")
    
    # Clean taxon name for filenames
    safe_taxon_name = taxon_name.replace(":", "").replace("→", "to").replace(" ", "_").replace("/", "_").replace("<", "lt").replace(">=", "gte")

    # 1. Plot Cosine Similarity Distribution
    try:
        # Replace inf with nan for plotting
        cosine_data = df['cosine_similarity'].replace([np.inf, -np.inf], np.nan).dropna()
        if not cosine_data.empty:
            plt.figure(figsize=(12, 7))
            sns.histplot(cosine_data, bins=50, kde=True, element="step", color="blue")
            mean_val = cosine_data.mean()
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            plt.title(f'Cosine Similarity Distribution\n{taxon_name} (n={len(cosine_data):,})')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f"{safe_taxon_name}_cosine_dist.png"))
            plt.close()
        else:
            print(f"  [Warning] Could not plot cosine distribution for {taxon_name}: No valid data.")
            
    except Exception as e:
        # Catch plotting errors
        print(f"  [Warning] Could not plot cosine distribution for {taxon_name}: {e}")

    # 2. Plot Tanimoto vs. Cosine Heatmap (only if is_inter is True)
    if is_inter:
        try:
            # Use a jointplot with hex bins for large data
            plot_data = df[['tanimoto', 'cosine_similarity']].replace([np.inf, -np.inf], np.nan).dropna()
            if not plot_data.empty and len(plot_data) > 100: # jointplot needs >1 point
                g = sns.jointplot(data=plot_data, x='tanimoto', y='cosine_similarity', kind='hex',
                                  height=8, gridsize=50, cmap='viridis', mincnt=1)
                g.fig.suptitle(f'Tanimoto vs. Cosine\n{taxon_name} (n={len(plot_data):,})', y=1.03)
                g.set_axis_labels('Tanimoto Similarity', 'Cosine Similarity')
                plt.savefig(os.path.join(plot_dir, f"{safe_taxon_name}_tanimoto_vs_cosine.png"))
                plt.close('all') # Close all figures from jointplot
            elif not plot_data.empty:
                 print(f"  [Warning] Skipping Tanimoto vs. Cosine plot for {taxon_name}: Not enough data ({len(plot_data)} rows).")
            else:
                print(f"  [Warning] Could not plot Tanimoto vs. Cosine for {taxon_name}: No valid data.")
        except Exception as e:
            print(f"  [Warning] Could not plot Tanimoto vs. Cosine for {taxon_name}: {e}")