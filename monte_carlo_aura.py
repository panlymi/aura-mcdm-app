import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os

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

def run_monte_carlo_aura(matrix, base_ranks, criteria_types, iterations=10000):
    """
    Executes Scenario A: Global Randomization using Dirichlet distribution.
    """
    n_alts = matrix.shape[0]
    
    # Generate weight vectors summing to 1 (Dirichlet)
    simulated_weights = np.random.dirichlet([1, 1, 1, 1, 1], size=iterations)
    
    spearman_correlations = []
    rank_matrix = np.zeros((iterations, n_alts))
    
    for i in range(iterations):
        weights = simulated_weights[i]
        _, sim_ranks = calculate_aura_baseline(matrix, weights, criteria_types)
        
        rank_matrix[i, :] = sim_ranks
        
        # Calculate Spearman's r_s against the baseline
        rho, _ = spearmanr(base_ranks, sim_ranks)
        spearman_correlations.append(rho)
        
    avg_rho = np.mean(spearman_correlations)
    
    # Rank Acceptability Index (RAI) for State 0 (e.g., Selangor)
    state_0_ranks = rank_matrix[:, 0]
    prob_rank_1 = np.sum(state_0_ranks == 1) / iterations * 100
    
    print(f"--- Monte Carlo Results ({iterations:,} Iterations) ---")
    print(f"Average Spearman Correlation (r_s): {avg_rho:.4f}")
    print(f"Probability of Selangor (index=0) retaining Rank 1: {prob_rank_1:.2f}%")
    
    return rank_matrix, spearman_correlations

if __name__ == "__main__":
    csv_path = r"C:\Users\fadzl\Desktop\MCDM\Full result AURA.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        exit(1)
        
    # Read the data
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} alternatives.")
    
    # Raw data columns: C1 to C5
    # From CSV: C1 Real GDP CAGR (%), C2 Median Household Income (RM), C3 Gini Coefficient, C4 Absolute Poverty Rate (%), C5 Expenditure / Income Ratio
    raw_data = df.iloc[:, 1:6].values
    
    # Baseline ranks from the CSV file
    baseline_ranks = df['Rank'].values
    
    # Criteria Types: 1 (Benefit), -1 (Cost), 0 (Target)
    # GDP(+), Income(+), Gini(-), Poverty(-), Target(0)
    criteria_types = [1, 1, -1, -1, 0]
    
    # Execute Phase 3
    np.random.seed(42)
    sim_ranks, correlations = run_monte_carlo_aura(raw_data, baseline_ranks, criteria_types, iterations=10000)
    
