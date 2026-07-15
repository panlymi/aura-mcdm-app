import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "aura-mcdm-matplotlib"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from monte_carlo_aura import run_monte_carlo_aura
import argparse

DEFAULT_CRITERIA_TYPES = (1, 1, -1, -1, 0)


def generate_exports(
    input_path="Full result AURA_new_weights.csv",
    *,
    iterations=10_000,
    seed=42,
    target=0.65,
    criteria_types=DEFAULT_CRITERIA_TYPES,
):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    alternatives = df['Alternatives'].values
    criterion_columns = df.columns[1 : 1 + len(criteria_types)].tolist()
    raw_data = df[criterion_columns].values
    
    if df['Rank'].dtype == object:
        df['Rank'] = df['Rank'].str.replace('\r', '').astype(int)
    baseline_ranks = df['Rank'].values
    
    n_alts = len(alternatives)
    print(f"Running {iterations:,} Monte Carlo simulations...")
    rank_matrix, correlations = run_monte_carlo_aura(
        raw_data,
        baseline_ranks,
        criteria_types,
        iterations=iterations,
        seed=seed,
        target_val=target,
    )
    
    # --- 1. Generate Excel Report ---
    print("Generating Excel Report...")
    report_data = []
    
    for j in range(n_alts):
        ranks = rank_matrix[:, j]
        name = alternatives[j]
        mean_rank = np.mean(ranks)
        rank_sd = np.std(ranks)
        min_rank = np.min(ranks)
        max_rank = np.max(ranks)
        top5_freq = (np.sum(ranks <= 5) / iterations) * 100
        bottom3_freq = (np.sum(ranks >= (n_alts - 2)) / iterations) * 100
        
        report_data.append({
            'Alternative': name,
            'Baseline Rank': int(baseline_ranks[j]),
            'Mean Rank': float(mean_rank),
            'Rank SD': float(rank_sd),
            'Min Rank': int(min_rank),
            'Max Rank': int(max_rank),
            'Top-5 Freq (%)': float(top5_freq),
            'Bottom-3 Freq (%)': float(bottom3_freq)
        })
    
    results_df = pd.DataFrame(report_data)
    # Sort by Mean Rank
    results_df = results_df.sort_values(by='Mean Rank')
    
    excel_path = 'AURA_MonteCarlo_Results.xlsx'
    try:
        results_df.to_excel(excel_path, index=False)
        print(f"Excel report saved to: {excel_path}")
    except PermissionError:
        print(f"WARNING: Could not overwrite {excel_path} (likely open in another program).")
        alt_path = 'AURA_MonteCarlo_Results_v2.xlsx'
        results_df.to_excel(alt_path, index=False)
        print(f"Excel report saved to alternate path: {alt_path}")

    detailed_rows = []
    for j, name in enumerate(alternatives):
        ranks = rank_matrix[:, j]
        detailed_rows.append({
            "State": name,
            "Baseline Rank": int(baseline_ranks[j]),
            "Average Simulated Rank": round(float(np.mean(ranks)), 2),
            "Rank SD": round(float(np.std(ranks)), 2),
            "Top-3 Frequency": float(np.mean(ranks <= min(3, n_alts))),
            "Top-5 Frequency": float(np.mean(ranks <= min(5, n_alts))),
            "Bottom-3 Frequency": float(np.mean(ranks >= max(1, n_alts - 2))),
        })
    pd.DataFrame(detailed_rows).sort_values("Baseline Rank").to_excel(
        "AURA_MonteCarlo_Results_Detailed.xlsx", index=False
    )

    summary_rows = []
    for column in criterion_columns:
        values = pd.to_numeric(df[column], errors="raise")
        label = column.split(" ", 1)[1] if column.startswith("C") and " " in column else column
        summary_rows.append({
            "Criterion": label,
            "Min": float(values.min()),
            "Max": float(values.max()),
            "Mean": float(values.mean()),
            "Median": float(values.median()),
            "Std. Dev.": float(values.std()),
        })
    pd.DataFrame(summary_rows).to_excel("Summary_Statistics.xlsx", index=False)
    
    # --- 2. Generate Figures ---
    print("Generating Figures...")
    
    # Figure 1: Boxplot of Ranks
    # Create a DataFrame for Seaborn
    plot_data = []
    for j in range(n_alts):
        name = alternatives[j].split(': ')[1] if ': ' in alternatives[j] else alternatives[j]
        for r in rank_matrix[:, j]:
            plot_data.append({'State': name, 'Rank': r, 'Baseline': baseline_ranks[j]})
            
    plot_df = pd.DataFrame(plot_data)
    
    # Sort states by their baseline rank for the x-axis
    sorted_states = df.sort_values(by='Rank')['Alternatives'].apply(lambda x: x.split(': ')[1] if ': ' in x else x).values
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x='State', y='Rank', hue='State', data=plot_df, order=sorted_states,
        palette='viridis', legend=False,
    )
    plt.title(f'Monte Carlo Simulation: Distribution of Ranks per State ({iterations:,} Iterations)', fontsize=16)
    plt.ylabel('Rank (1 is Best)', fontsize=14)
    plt.xlabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(1, n_alts + 1, 1))
    plt.gca().invert_yaxis() # Invert so Rank 1 is at the top
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('MCDA_Rank_Volatility_Boxplot.png', dpi=300)
    plt.close()
    print("Saved MCDA_Rank_Volatility_Boxplot.png")
    
    # Figure 2: Rank Acceptability Index (RAI) Stacked Bar Chart
    # Count frequencies of ranks 1-5, 6-12, 13-16
    rai_data = []
    top_limit = min(5, n_alts)
    bottom_start = max(top_limit + 1, n_alts - 2)
    for j in range(n_alts):
        ranks = rank_matrix[:, j]
        name = alternatives[j].split(': ')[1] if ': ' in alternatives[j] else alternatives[j]
        
        top5 = np.sum(ranks <= top_limit) / iterations * 100
        mid = np.sum((ranks > top_limit) & (ranks < bottom_start)) / iterations * 100
        bot3 = np.sum(ranks >= bottom_start) / iterations * 100
        
        rai_data.append({
            'State': name,
            f'Top (1-{top_limit})': top5,
            f'Middle ({top_limit + 1}-{bottom_start - 1})': mid,
            f'Bottom ({bottom_start}-{n_alts})': bot3
        })
        
    rai_df = pd.DataFrame(rai_data)
    rai_df = rai_df.set_index('State')
    # Reorder by Top 5 frequency
    rai_df = rai_df.sort_values(by=f'Top (1-{top_limit})', ascending=False)
    
    ax = rai_df.plot(kind='bar', stacked=True, figsize=(14, 8), color=['#2ca02c', '#b5b5b5', '#d62728'])
    plt.title('Rank Acceptability Index (RAI) - State Performance Frequencies', fontsize=16)
    plt.ylabel('Frequency (%)', fontsize=14)
    plt.xlabel('State', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Rank Tiers', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add percentage labels inside bars if > 5%
    for c in ax.containers:
        labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')

    plt.tight_layout()
    plt.savefig('MCDA_Rank_Acceptability_Bar.png', dpi=300)
    plt.close()
    print("Saved MCDA_Rank_Acceptability_Bar.png")

    # Figure 3: Heatmap of Rank Frequencies
    print("Generating Heatmap...")
    heatmap_data = np.zeros((n_alts, n_alts))
    
    # Calculate frequencies of each rank for each state
    for i in range(n_alts):
        ranks = rank_matrix[:, i]
        for r in range(1, n_alts + 1):
            # Count how many times state 'i' got rank 'r'
            heatmap_data[i, r-1] = np.sum(ranks == r) / iterations * 100
            
    # Create DataFrame for the heatmap
    states_names = [a.split(': ')[1] if ': ' in a else a for a in alternatives]
    heatmap_df = pd.DataFrame(heatmap_data, index=states_names, columns=[f'Rank {r}' for r in range(1, n_alts + 1)])
    
    # Sort states by baseline rank
    heatmap_df = heatmap_df.reindex(sorted_states)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Frequency (%)'}, linewidths=.5)
    plt.title(f'Monte Carlo Simulation: Heatmap of Rank Frequencies ({iterations:,} Iterations)', fontsize=16)
    plt.ylabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xlabel('Possible Ranks', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig('MCDA_Rank_Heatmap.png', dpi=300)
    plt.close()
    print("Saved MCDA_Rank_Heatmap.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reproducible AURA Monte Carlo exports.")
    parser.add_argument("--input", default="Full result AURA_new_weights.csv")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--criteria-types", type=int, nargs="+", default=list(DEFAULT_CRITERIA_TYPES))
    args = parser.parse_args()
    generate_exports(
        args.input,
        iterations=args.iterations,
        seed=args.seed,
        target=args.target,
        criteria_types=tuple(args.criteria_types),
    )
