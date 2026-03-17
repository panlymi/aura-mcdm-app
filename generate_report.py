import pandas as pd
import numpy as np
import json
from monte_carlo_aura import run_monte_carlo_aura

def generate_report():
    csv_path = 'Full result AURA.csv'
    df = pd.read_csv(csv_path)
    
    # Extract alternative names
    alternatives = df['Alternatives'].values
    
    # Extract raw data
    raw_data = df.iloc[:, 1:6].values
    
    # Extract baseline ranks and clean them
    if df['Rank'].dtype == object:
        df['Rank'] = df['Rank'].str.replace('\r', '').astype(int)
    baseline_ranks = df['Rank'].values
    
    n_alts = len(alternatives)
    criteria_types = [1, 1, -1, -1, 0]
    
    np.random.seed(42)
    iterations = 10000
    rank_matrix, correlations = run_monte_carlo_aura(raw_data, baseline_ranks, criteria_types, iterations=iterations)
    
    avg_spearman = np.mean(correlations)
    
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
            'Baseline_Rank': int(baseline_ranks[j]),
            'Mean_Rank': float(mean_rank),
            'Rank_SD': float(rank_sd),
            'Min_Rank': int(min_rank),
            'Max_Rank': int(max_rank),
            'Top_5_Freq_Pct': float(top5_freq),
            'Bottom_3_Freq_Pct': float(bottom3_freq)
        })
        
    result = {
        'average_spearman': float(avg_spearman),
        'iterations': iterations,
        'metrics': report_data
    }
    
    with open('report.json', 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    generate_report()
