import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import copy

def calculate_aura_baseline(matrix, weights, criteria_types, target_val=0.65):
    """
    Executes the Adaptive Utility Ranking Algorithm (AURA).
    Criteria Types: 1 (Benefit), -1 (Cost), 0 (Target)
    """
    n_alts, n_crits = matrix.shape
    norm_matrix = np.zeros((n_alts, n_crits))
    
    # 1. Dynamic Normalization
    for j in range(n_crits):
        col = matrix[:, j]
        max_val, min_val = np.max(col), np.min(col)
        
        # Handle division by zero if all values are identical
        if max_val == min_val:
            norm_matrix[:, j] = 1.0
            continue
            
        if criteria_types[j] == 1: # Benefit
            norm_matrix[:, j] = (col - min_val) / (max_val - min_val)
        elif criteria_types[j] == -1: # Cost
            norm_matrix[:, j] = (max_val - col) / (max_val - min_val)
        elif criteria_types[j] == 0: # Target
            if max_val == min_val:
                norm_matrix[:, j] = 1.0
            else:
                norm_matrix[:, j] = 1.0 - (np.abs(col - target_val) / (max_val - min_val))
            
    # 2. Apply Expert Weights
    weighted_matrix = norm_matrix * weights
    
    # 3. Tri-Benchmark Reference Points
    pis = np.max(weighted_matrix, axis=0)  # Positive Ideal Solution
    nis = np.min(weighted_matrix, axis=0)  # Negative Ideal Solution
    avg = np.mean(weighted_matrix, axis=0) # Average Solution
    
    # 4. Euclidean Distances
    d_pis = np.sqrt(np.sum((weighted_matrix - pis)**2, axis=1))
    d_nis = np.sqrt(np.sum((weighted_matrix - nis)**2, axis=1))
    d_avg = np.sqrt(np.sum((weighted_matrix - avg)**2, axis=1))
    
    # 5. Societal Utility Score Calculation (Exact AURA Formula)
    def correct(d):
        sigma = np.max(d) - np.min(d)
        return d + sigma * (d ** 2)
        
    d_plus_corrected = correct(d_pis)
    d_minus_corrected = correct(d_nis)
    d_avg_corrected = correct(d_avg)
    
    alpha = 0.5
    utility_scores = (alpha * (d_plus_corrected - d_minus_corrected) + (1.0 - alpha) * d_avg_corrected) / 2.0
    
    # Ranking (Lowest utility score gets highest rank 1)
    ranks = np.argsort(np.argsort(utility_scores)) + 1
    
    return utility_scores, ranks


def generate_constrained_weights(scenario, iterations=10000):
    """
    Generates an N x 5 matrix of weights based on specific policy perturbation scenarios.
    Ensures the sum of each weight vector is exactly 1.0.
    """
    weight_matrix = np.zeros((iterations, 5))
    
    for i in range(iterations):
        if scenario == 'B':  # Hyper-Capitalist: Maximize GDP, Minimize Poverty
            w_gdp = np.random.uniform(0.40, 0.70)      # w[0]
            w_poverty = np.random.uniform(0.05, 0.15)  # w[3]
            
        elif scenario == 'C':  # Extreme Welfare: Maximize Poverty, Minimize GDP
            w_gdp = np.random.uniform(0.05, 0.15)      # w[0]
            w_poverty = np.random.uniform(0.40, 0.70)  # w[3]
            
        else:
            raise ValueError("Scenario must be 'B' or 'C'")
            
        # Calculate the remaining weight to be distributed
        remainder = 1.0 - (w_gdp + w_poverty)
        
        # Distribute the remainder dynamically to Income [1], Gini [2], Target [4]
        # Using Dirichlet ensures a fair, randomized spread that strictly sums to 1
        rem_distribution = np.random.dirichlet([1, 1, 1]) * remainder
        
        # Assemble the final valid weight vector
        weight_matrix[i, 0] = w_gdp
        weight_matrix[i, 1] = rem_distribution[0] # Income
        weight_matrix[i, 2] = rem_distribution[1] # Gini
        weight_matrix[i, 3] = w_poverty
        weight_matrix[i, 4] = rem_distribution[2] # Target
        
    return weight_matrix

def run_targeted_perturbation(matrix, base_ranks, criteria_types, scenario='B', iterations=10000):
    """
    Executes the AURA Monte Carlo simulation using the constrained scenario weights.
    """
    n_alts = matrix.shape[0]
    
    # Generate the mathematically sound constrained weights
    simulated_weights = generate_constrained_weights(scenario, iterations)
    
    spearman_correlations = []
    rank_matrix = np.zeros((iterations, n_alts))
    
    print(f"\n--- Executing Scenario {scenario} ({iterations} Iterations) ---")
    
    for i in range(iterations):
        weights = simulated_weights[i]
        
        # Call the baseline AURA function from Phase 2
        _, sim_ranks = calculate_aura_baseline(matrix, weights, criteria_types)
        
        rank_matrix[i, :] = sim_ranks
        
        # Calculate Spearman's r_s
        rho, _ = spearmanr(base_ranks, sim_ranks)
        spearman_correlations.append(rho)
        
    avg_rho = np.mean(spearman_correlations)
    
    # Calculate volatility for a specific highly-industrialized state 
    # (Assuming State 0 is Selangor in your matrix)
    state_0_ranks = rank_matrix[:, 0]
    prob_rank_1 = np.sum(state_0_ranks == 1) / iterations * 100
    worst_rank = np.max(state_0_ranks)
    
    # Also find probability of ranking lower than 5 to show volatility
    prob_low_rank = np.sum(state_0_ranks > 5) / iterations * 100
    
    print(f"Average Spearman Correlation (r_s) against Baseline: {avg_rho:.4f}")
    print(f"State A10 (Selangor) Probability of Rank 1: {prob_rank_1:.2f}%")
    print(f"State A10 (Selangor) Worst Rank Achieved: {worst_rank:.0f}")
    print(f"State A10 (Selangor) Probability of Rank > 5: {prob_low_rank:.2f}%")
    
    return rank_matrix, spearman_correlations

if __name__ == "__main__":
    csv_path = 'Full result AURA_new_weights.csv'
    df = pd.read_csv(csv_path)
    
    alternatives = df['Alternatives'].values
    raw_data = df.iloc[:, 1:6].values
    
    if df['Rank'].dtype == object:
        df['Rank'] = df['Rank'].str.replace('\r', '').astype(int)
    baseline_ranks = df['Rank'].values
    
    n_alts = len(alternatives)
    criteria_types = [1, 1, -1, -1, 0]
    
    np.random.seed(42)
    # Scenario B
    ranks_B, corrs_B = run_targeted_perturbation(raw_data, baseline_ranks, criteria_types, scenario='B', iterations=10000)
    
    # Scenario C
    ranks_C, corrs_C = run_targeted_perturbation(raw_data, baseline_ranks, criteria_types, scenario='C', iterations=10000)
    
    # --- 1. Export Targeted Scenario Excel Report ---
    print("\nGenerating Scenario Comparison Excel Report...")
    report_data = []
    
    for j in range(n_alts):
        name = alternatives[j]
        base_r = int(baseline_ranks[j])
        
        # B Stats
        ranks_b = ranks_B[:, j]
        mean_b = np.mean(ranks_b)
        min_b = np.min(ranks_b)
        max_b = np.max(ranks_b)
        top5_b = (np.sum(ranks_b <= 5) / 10000) * 100
        
        # C Stats
        ranks_c = ranks_C[:, j]
        mean_c = np.mean(ranks_c)
        min_c = np.min(ranks_c)
        max_c = np.max(ranks_c)
        top5_c = (np.sum(ranks_c <= 5) / 10000) * 100
        
        report_data.append({
            'Alternative': name,
            'Baseline Rank': base_r,
            'Mean Rank (Capitalist)': float(mean_b),
            'Top-5 Freq % (Capitalist)': float(top5_b),
            'Min-Max (Capitalist)': f"{int(min_b)} - {int(max_b)}",
            'Mean Rank (Welfare)': float(mean_c),
            'Top-5 Freq % (Welfare)': float(top5_c),
            'Min-Max (Welfare)': f"{int(min_c)} - {int(max_c)}",
            'Absolute Mean Shift': abs(float(mean_b) - float(mean_c))
        })
    
    results_df = pd.DataFrame(report_data)
    results_df = results_df.sort_values(by='Baseline Rank')
    
    excel_path = 'AURA_Scenarios_BC_Results.xlsx'
    try:
        results_df.to_excel(excel_path, index=False)
        print(f"Scenario Excel report saved to: {excel_path}")
    except PermissionError:
        alt_path = 'AURA_Scenarios_BC_Results_v2.xlsx'
        results_df.to_excel(alt_path, index=False)
        print(f"Scenario Excel report saved to alternate path: {alt_path}")
        
    # --- 2. Generate Comparative Figure ---
    print("\nGenerating Comparative Scenario Figure...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a DataFrame for Seaborn
    plot_data = []
    for j in range(n_alts):
        name = alternatives[j].split(': ')[1] if ': ' in alternatives[j] else alternatives[j]
        # Boxplot B
        for r in ranks_B[:, j]:
            plot_data.append({'State': name, 'Rank': r, 'Scenario': 'Hyper-Capitalist'})
        # Boxplot C
        for r in ranks_C[:, j]:
            plot_data.append({'State': name, 'Rank': r, 'Scenario': 'Extreme Welfare'})
            
    plot_df = pd.DataFrame(plot_data)
    
    # Sort states by their baseline rank for the x-axis
    sorted_states = df.sort_values(by='Rank')['Alternatives'].apply(lambda x: x.split(': ')[1] if ': ' in x else x).values
    
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='State', y='Rank', hue='Scenario', data=plot_df, order=sorted_states, palette=['#1f77b4', '#ff7f0e'])
    plt.title('AURA Scenario Testing: Hyper-Capitalist vs Extreme Welfare (10,000 Iterations)', fontsize=16)
    plt.ylabel('Rank (1 is Best)', fontsize=14)
    plt.xlabel('State (Ordered by Baseline Rank)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(1, 17, 1))
    plt.gca().invert_yaxis() # Invert so Rank 1 is at the top
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Policy Scenario', loc='upper right')
    plt.tight_layout()
    plt.savefig('MCDA_Scenario_Comparison_Boxplot.png', dpi=300)
    plt.close()
    print("Saved MCDA_Scenario_Comparison_Boxplot.png")
