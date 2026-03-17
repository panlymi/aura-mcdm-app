import pandas as pd
import numpy as np

def generate_new_baseline():
    """Calculates AURA baseline to replace the loaded Full result AURA ranks"""
    csv_path = 'Full result AURA.csv'
    df = pd.read_csv(csv_path)
    
    alternatives = df['Alternatives'].values
    raw_data = df.iloc[:, 1:6].values
    
    criteria_types = [1, 1, -1, -1, 0]
    expert_weights = np.array([0.20, 0.25, 0.20, 0.25, 0.10])
    target_val = 0.65
    
    n_alts, n_crits = raw_data.shape
    norm_matrix = np.zeros((n_alts, n_crits))
    
    for j in range(n_crits):
        col = raw_data[:, j]
        max_val, min_val = np.max(col), np.min(col)
        if max_val == min_val:
            norm_matrix[:, j] = 1.0
            continue
        if criteria_types[j] == 1:
            norm_matrix[:, j] = (col - min_val) / (max_val - min_val)
        elif criteria_types[j] == -1:
            norm_matrix[:, j] = (max_val - col) / (max_val - min_val)
        elif criteria_types[j] == 0:
            if max_val == min_val:
                norm_matrix[:, j] = 1.0
            else:
                norm_matrix[:, j] = 1.0 - (np.abs(col - target_val) / (max_val - min_val))
                
    weighted_matrix = norm_matrix * expert_weights
    
    pis = np.max(weighted_matrix, axis=0)
    nis = np.min(weighted_matrix, axis=0)
    avg = np.mean(weighted_matrix, axis=0)
    
    d_pis = np.sqrt(np.sum((weighted_matrix - pis)**2, axis=1))
    d_nis = np.sqrt(np.sum((weighted_matrix - nis)**2, axis=1))
    d_avg = np.sqrt(np.sum((weighted_matrix - avg)**2, axis=1))
    
    def correct(d):
        sigma = np.max(d) - np.min(d)
        return d + sigma * (d ** 2)
        
    d_plus_corrected = correct(d_pis)
    d_minus_corrected = correct(d_nis)
    d_avg_corrected = correct(d_avg)
    
    alpha = 0.5
    utility_scores = (alpha * (d_plus_corrected - d_minus_corrected) + (1.0 - alpha) * d_avg_corrected) / 2.0
    
    ranks = np.argsort(np.argsort(utility_scores)) + 1
    
    df['D+ (PIS)'] = d_plus_corrected
    df['D- (NIS)'] = d_minus_corrected
    df['D_avg (AS)'] = d_avg_corrected
    df['Rank'] = ranks
    df['Utility Score'] = utility_scores
    
    df.to_csv('Full result AURA_new_weights.csv', index=False)
    print("New baseline generated and saved to Full result AURA_new_weights.csv")
    return df

if __name__ == "__main__":
    generate_new_baseline()
