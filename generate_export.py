import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from monte_carlo_aura import run_monte_carlo_aura

def generate_exports():
    print("Loading data...")
    csv_path = 'Full result AURA_new_weights.csv'
    df = pd.read_csv(csv_path)
    
    alternatives = df['Alternatives'].values
    raw_data = df.iloc[:, 1:6].values
    
    if df['Rank'].dtype == object:
        df['Rank'] = df['Rank'].str.replace('\r', '').astype(int)
    baseline_ranks = df['Rank'].values
    
    n_alts = len(alternatives)
    criteria_types = [1, 1, -1, -1, 0]
    
    print("Running 10,000 Monte Carlo simulations...")
    np.random.seed(42)
    iterations = 10000
    rank_matrix, correlations = run_monte_carlo_aura(raw_data, baseline_ranks, criteria_types, iterations=iterations)
    
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
    sns.boxplot(x='State', y='Rank', data=plot_df, order=sorted_states, palette='viridis')
    plt.title('Monte Carlo Simulation: Distribution of Ranks per State (10,000 Iterations)', fontsize=16)
    plt.ylabel('Rank (1 is Best)', fontsize=14)
    plt.xlabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(1, 17, 1))
    plt.gca().invert_yaxis() # Invert so Rank 1 is at the top
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('MCDA_Rank_Volatility_Boxplot.png', dpi=300)
    plt.close()
    print("Saved MCDA_Rank_Volatility_Boxplot.png")
    
    # Figure 2: Rank Acceptability Index (RAI) Stacked Bar Chart
    # Count frequencies of ranks 1-5, 6-12, 13-16
    rai_data = []
    for j in range(n_alts):
        ranks = rank_matrix[:, j]
        name = alternatives[j].split(': ')[1] if ': ' in alternatives[j] else alternatives[j]
        
        top5 = np.sum(ranks <= 5) / iterations * 100
        mid = np.sum((ranks > 5) & (ranks < 13)) / iterations * 100
        bot3 = np.sum(ranks >= 13) / iterations * 100
        
        rai_data.append({
            'State': name,
            'Top 5 (1-5)': top5,
            'Middle (6-12)': mid,
            'Bottom (13-15)': bot3
        })
        
    rai_df = pd.DataFrame(rai_data)
    rai_df = rai_df.set_index('State')
    # Reorder by Top 5 frequency
    rai_df = rai_df.sort_values(by='Top 5 (1-5)', ascending=False)
    
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
    plt.title('Monte Carlo Simulation: Heatmap of Rank Frequencies (10,000 Iterations)', fontsize=16)
    plt.ylabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xlabel('Possible Ranks', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig('MCDA_Rank_Heatmap.png', dpi=300)
    plt.close()
    print("Saved MCDA_Rank_Heatmap.png")


if __name__ == "__main__":
    generate_exports()
