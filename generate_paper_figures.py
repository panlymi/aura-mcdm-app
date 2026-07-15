import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "aura-mcdm-matplotlib"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from monte_carlo_aura import run_monte_carlo_aura
from monte_carlo_scenarios import run_targeted_perturbation
import argparse

DEFAULT_CRITERIA_TYPES = (1, 1, -1, -1, 0)


def generate_paper_figures(
    input_path="Full result AURA_new_weights.csv",
    *,
    iterations=10_000,
    seed=42,
    target=0.65,
    criteria_types=DEFAULT_CRITERIA_TYPES,
):
    print("Loading Baseline Data...")
    df = pd.read_csv(input_path)
    
    alternatives = df['Alternatives'].values
    state_names = [a.split(': ')[1] if ': ' in a else a for a in alternatives]
    criterion_columns = df.columns[1 : 1 + len(criteria_types)].tolist()
    raw_data = df[criterion_columns].values
    baseline_ranks = df['Rank'].values
    utility_scores = df['Utility Score'].values
    
    n_alts = len(alternatives)
    # --- FIGURE 1: Baseline Societal Utility Bar Chart ---
    print("Generating Figure 1: Baseline Utility Bar Chart...")
    
    # Sort data by Utility Score for the clean bar chart
    fig1_df = pd.DataFrame({'State': state_names, 'Utility': utility_scores, 'Rank': baseline_ranks})
    fig1_df = fig1_df.sort_values(by='Rank', ascending=True)
    
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(
        x='State', y='Utility', hue='State', data=fig1_df,
        palette='viridis', legend=False,
    )
    plt.title('Figure 1: Final AURA Societal Utility Scores (Baseline Control)', fontsize=16, fontweight='bold')
    plt.ylabel('AURA Utility Score (lower is better)', fontsize=14)
    plt.xlabel('State', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add rank annotations on top of the bars
    for i, p in enumerate(bars.patches):
        bars.annotate(f"Rank {int(fig1_df.iloc[i]['Rank'])}", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', xytext=(0, 10), 
                      textcoords='offset points', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('Figure_1_Baseline_Utility.png', dpi=300)
    plt.close()
    
    
    # --- Run Monte Carlo for Figures 2 & 4 ---
    print(f"\nRunning {iterations} Global Dirichlet Iterations for Figures 2 & 4...")
    rank_matrix_A, _ = run_monte_carlo_aura(
        raw_data,
        baseline_ranks,
        criteria_types,
        iterations=iterations,
        seed=seed,
        target_val=target,
    )
    
    # --- FIGURE 2: Global Volatility Box Plot (Scenario A) ---
    print("Generating Figure 2: Rank Volatility Boxplot...")
    plot_data_A = []
    for j in range(n_alts):
        for r in rank_matrix_A[:, j]:
            plot_data_A.append({'State': state_names[j], 'Rank': r})
            
    plot_df_A = pd.DataFrame(plot_data_A)
    sorted_states_A = df.sort_values(by='Rank')['Alternatives'].apply(lambda x: x.split(': ')[1] if ': ' in x else x).values
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x='State', y='Rank', hue='State', data=plot_df_A,
        order=sorted_states_A, palette='coolwarm', legend=False,
    )
    plt.title(f'Figure 2: Rank Volatility Across {iterations:,} Global Randomizations (Scenario A)', fontsize=16, fontweight='bold')
    plt.ylabel('Simulated Rank (1 is Best)', fontsize=14)
    plt.xlabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(1, n_alts + 1, 1))
    plt.gca().invert_yaxis()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figure_2_Global_Volatility_Boxplot.png', dpi=300)
    plt.close()
    
    
    # --- Run Perturbations for Figure 3 ---
    print("\nRunning Perturbation Scenarios for Figure 3...")
    rank_matrix_B, _ = run_targeted_perturbation(
        raw_data, baseline_ranks, criteria_types, scenario='B', iterations=iterations,
        seed=seed, target_val=target,
    )
    rank_matrix_C, _ = run_targeted_perturbation(
        raw_data, baseline_ranks, criteria_types, scenario='C', iterations=iterations,
        seed=seed + 1, target_val=target,
    )
    
    # --- FIGURE 3: Policy Perturbation Slopegraph (Bump Chart) ---
    print("Generating Figure 3: Perturbation Slopegraph...")
    
    # Calculate Mean Ranks across scenarios
    mean_ranks_B = np.mean(rank_matrix_B, axis=0)
    mean_ranks_C = np.mean(rank_matrix_C, axis=0)
    
    # Pick 5 narrative-heavy states to make the chart readable
    # Top 2, Bottom 2, and 1 Volatile state
    baseline_order = np.argsort(baseline_ranks)
    narrative_indices = list(dict.fromkeys([
        int(baseline_order[0]),
        int(baseline_order[min(1, n_alts - 1)]),
        int(baseline_order[n_alts // 2]),
        int(baseline_order[max(0, n_alts - 2)]),
        int(baseline_order[-1]),
    ]))
    
    plt.figure(figsize=(10, 8))
    
    for idx in narrative_indices:
        state = state_names[idx]
        y_points = [mean_ranks_B[idx], baseline_ranks[idx], mean_ranks_C[idx]]
        x_points = [1, 2, 3]
        
        plt.plot(x_points, y_points, marker='o', markersize=8, linewidth=3, label=state)
        
        # Add text labels on the left and right 
        plt.text(0.95, y_points[0], f"{state} ({y_points[0]:.1f})", ha='right', va='center', fontsize=10)
        plt.text(3.05, y_points[2], f"{state} ({y_points[2]:.1f})", ha='left', va='center', fontsize=10)

    plt.title('Figure 3: Mean Rank Shifts Under Extremist Policy Scenarios', fontsize=16, fontweight='bold')
    plt.xticks([1, 2, 3], ['Hyper-Capitalist\n(Max GDP / Min Poverty)', 'Balanced Baseline\n(Empirical Weights)', 'Extreme Welfare\n(Min GDP / Max Poverty)'], fontsize=12)
    plt.ylabel('Mean Rank (1 is Best)', fontsize=14)
    plt.yticks(np.arange(1, n_alts + 1, 1))
    plt.gca().invert_yaxis()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xlim(0.5, 3.5) # Add padding for the labels
    plt.legend(loc='lower left', title="Narrative States")
    plt.tight_layout()
    plt.savefig('Figure_3_Perturbation_Slopegraph.png', dpi=300)
    plt.close()
    
    
    # --- FIGURE 4: Rank Acceptability Index (RAI) Heatmap ---
    print("\nGenerating Figure 4: RAI Tier Heatmap...")
    
    # Dynamic tiers: Top 3, two middle bands, and Bottom 3.
    rai_tiers = np.zeros((n_alts, 4))
    bottom_start = max(4, n_alts - 2)
    first_middle_end = (3 + (bottom_start - 1) + 1) // 2
    
    for i in range(n_alts):
        ranks = rank_matrix_A[:, i]
        
        rai_tiers[i, 0] = np.sum(ranks <= 3) / iterations * 100
        rai_tiers[i, 1] = np.sum((ranks >= 4) & (ranks <= first_middle_end)) / iterations * 100
        rai_tiers[i, 2] = np.sum((ranks > first_middle_end) & (ranks < bottom_start)) / iterations * 100
        rai_tiers[i, 3] = np.sum(ranks >= bottom_start) / iterations * 100
        
    tier_labels = [
        'Top 3',
        f'Ranks 4-{first_middle_end}',
        f'Ranks {first_middle_end + 1}-{bottom_start - 1}',
        'Bottom 3',
    ]
    rai_df = pd.DataFrame(rai_tiers, index=state_names, columns=tier_labels)
    
    # Sort states by baseline rank
    rai_df = rai_df.reindex(sorted_states_A)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(rai_df, annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': 'Frequency (%)'}, linewidths=1)
    plt.title('Figure 4: Rank Acceptability Index (RAI) Stability Tiers', fontsize=16, fontweight='bold')
    plt.ylabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xlabel('Acceptability Tiers', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig('Figure_4_RAI_Heatmap.png', dpi=300)
    plt.close()

    print("\n--- All four research figures successfully generated! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reproducible AURA paper figures.")
    parser.add_argument("--input", default="Full result AURA_new_weights.csv")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--criteria-types", type=int, nargs="+", default=list(DEFAULT_CRITERIA_TYPES))
    args = parser.parse_args()
    generate_paper_figures(
        args.input,
        iterations=args.iterations,
        seed=args.seed,
        target=args.target,
        criteria_types=tuple(args.criteria_types),
    )
