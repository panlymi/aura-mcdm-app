import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from monte_carlo_aura import run_monte_carlo_aura
from monte_carlo_scenarios import run_targeted_perturbation

def generate_paper_figures():
    print("Loading Baseline Data...")
    csv_path = 'Full result AURA_new_weights.csv'
    df = pd.read_csv(csv_path)
    
    alternatives = df['Alternatives'].values
    state_names = [a.split(': ')[1] if ': ' in a else a for a in alternatives]
    raw_data = df.iloc[:, 1:6].values
    baseline_ranks = df['Rank'].values
    utility_scores = df['Utility Score'].values
    
    n_alts = len(alternatives)
    criteria_types = [1, 1, -1, -1, 0]
    iterations = 10000
    
    # --- FIGURE 1: Baseline Societal Utility Bar Chart ---
    print("Generating Figure 1: Baseline Utility Bar Chart...")
    
    # Sort data by Utility Score for the clean bar chart
    fig1_df = pd.DataFrame({'State': state_names, 'Utility': utility_scores, 'Rank': baseline_ranks})
    fig1_df = fig1_df.sort_values(by='Utility', ascending=False)
    
    plt.figure(figsize=(14, 8))
    bars = sns.barplot(x='State', y='Utility', data=fig1_df, palette='viridis')
    plt.title('Figure 1: Final AURA Societal Utility Scores (Baseline Control)', fontsize=16, fontweight='bold')
    plt.ylabel('Societal Utility Score (0 to 1)', fontsize=14)
    plt.xlabel('State', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylim(0, max(utility_scores) * 1.1)
    
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
    np.random.seed(42)
    rank_matrix_A, _ = run_monte_carlo_aura(raw_data, baseline_ranks, criteria_types, iterations=iterations)
    
    # --- FIGURE 2: Global Volatility Box Plot (Scenario A) ---
    print("Generating Figure 2: Rank Volatility Boxplot...")
    plot_data_A = []
    for j in range(n_alts):
        for r in rank_matrix_A[:, j]:
            plot_data_A.append({'State': state_names[j], 'Rank': r})
            
    plot_df_A = pd.DataFrame(plot_data_A)
    sorted_states_A = df.sort_values(by='Rank')['Alternatives'].apply(lambda x: x.split(': ')[1] if ': ' in x else x).values
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='State', y='Rank', data=plot_df_A, order=sorted_states_A, palette='coolwarm')
    plt.title('Figure 2: Rank Volatility Across 10,000 Global Randomizations (Scenario A)', fontsize=16, fontweight='bold')
    plt.ylabel('Simulated Rank (1 is Best)', fontsize=14)
    plt.xlabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(1, 17, 1))
    plt.gca().invert_yaxis()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figure_2_Global_Volatility_Boxplot.png', dpi=300)
    plt.close()
    
    
    # --- Run Perturbations for Figure 3 ---
    print("\nRunning Perturbation Scenarios for Figure 3...")
    rank_matrix_B, _ = run_targeted_perturbation(raw_data, baseline_ranks, criteria_types, scenario='B', iterations=iterations)
    rank_matrix_C, _ = run_targeted_perturbation(raw_data, baseline_ranks, criteria_types, scenario='C', iterations=iterations)
    
    # --- FIGURE 3: Policy Perturbation Slopegraph (Bump Chart) ---
    print("Generating Figure 3: Perturbation Slopegraph...")
    
    # Calculate Mean Ranks across scenarios
    mean_ranks_B = np.mean(rank_matrix_B, axis=0)
    mean_ranks_C = np.mean(rank_matrix_C, axis=0)
    
    # Pick 5 narrative-heavy states to make the chart readable
    # Top 2, Bottom 2, and 1 Volatile state
    narrative_indices = [
        np.where(baseline_ranks == 1)[0][0],  # #1 Baseline (Selangor)
        np.where(baseline_ranks == 2)[0][0],  # #2 Baseline (KL)
        np.where(baseline_ranks == 14)[0][0], # #14 Baseline (Kelantan)
        np.where(baseline_ranks == 15)[0][0], # #15 Baseline (Sabah)
        np.where(baseline_ranks == 5)[0][0]   # #5 Labuan (Highly Volatile)
    ]
    
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
    plt.yticks(np.arange(1, 17, 1))
    plt.gca().invert_yaxis()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.xlim(0.5, 3.5) # Add padding for the labels
    plt.legend(loc='lower left', title="Narrative States")
    plt.tight_layout()
    plt.savefig('Figure_3_Perturbation_Slopegraph.png', dpi=300)
    plt.close()
    
    
    # --- FIGURE 4: Rank Acceptability Index (RAI) Heatmap ---
    print("\nGenerating Figure 4: RAI Tier Heatmap...")
    
    # Tiers: Top 3, Ranks 4-8, Ranks 9-12, Bottom 3 (13-15)
    rai_tiers = np.zeros((n_alts, 4))
    
    for i in range(n_alts):
        ranks = rank_matrix_A[:, i]
        
        rai_tiers[i, 0] = np.sum(ranks <= 3) / iterations * 100
        rai_tiers[i, 1] = np.sum((ranks >= 4) & (ranks <= 8)) / iterations * 100
        rai_tiers[i, 2] = np.sum((ranks >= 9) & (ranks <= 12)) / iterations * 100
        rai_tiers[i, 3] = np.sum(ranks >= 13) / iterations * 100
        
    tier_labels = ['Top 3', 'Ranks 4-8', 'Ranks 9-12', 'Bottom 3']
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
    generate_paper_figures()
