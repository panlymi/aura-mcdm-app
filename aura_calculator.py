import pandas as pd
import numpy as np

def calculate_aura(data: pd.DataFrame, weights: dict, directions: dict, alpha: float, p: int = 2):
    """
    Computes the Adaptive Utility Ranking Algorithm (AURA).
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        alpha (float): The balance parameter (0 to 1).
        p (int): Distance metric parameter (1 for Manhattan, 2 for Euclidean).
        
    Returns:
        pd.DataFrame: A dataframe containing the rankings, scores, and distance metrics.
    """
    df = data.copy()
    
    # 1. Normalization
    normalized_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        max_val = df[col].max()
        min_val = df[col].min()
        
        # Handle edge case where all values are the same
        if max_val == min_val:
            normalized_df[col] = 1.0
            continue
            
        if directions.get(col, 'maximize') == 'maximize':
            normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = (max_val - df[col]) / (max_val - min_val)

    # 2. Weighted Matrix
    weighted_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        w = weights.get(col, 1.0)
        weighted_df[col] = normalized_df[col] * w

    # 3. Determine Ideal Solutions (PIS, NIS, AS)
    pis = weighted_df.max()
    nis = weighted_df.min()
    as_sol = weighted_df.mean()

    # Calculate raw distances (using Minkowski distance)
    # Adding epsilon to zero values to avoid perfectly zero distance which breaks things sometimes
    if p == 1:
        d_pos_raw = (abs(weighted_df - pis)).sum(axis=1)
        d_neg_raw = (abs(weighted_df - nis)).sum(axis=1)
        d_avg_raw = (abs(weighted_df - as_sol)).sum(axis=1)
    else:
        # Default to Euclidean (p=2)
        d_pos_raw = np.power((np.power(abs(weighted_df - pis), p)).sum(axis=1), 1/p)
        d_neg_raw = np.power((np.power(abs(weighted_df - nis), p)).sum(axis=1), 1/p)
        d_avg_raw = np.power((np.power(abs(weighted_df - as_sol), p)).sum(axis=1), 1/p)

    # Apply Correction Factor (Penalty Term)
    def correct(d):
        sigma = np.max(d) - np.min(d)
        return d + sigma * (d ** 2)

    distances = pd.DataFrame(index=df.index)
    distances['D_plus'] = correct(d_pos_raw)
    distances['D_minus'] = correct(d_neg_raw)
    distances['D_avg'] = correct(d_avg_raw)

    # 5. Final Utility Score (Exact AURA Formula)
    distances['Utility_Score'] = (alpha * (distances['D_plus'] - distances['D_minus']) + (1 - alpha) * distances['D_avg']) / 2

    # 6. Ranking (Lowest score gets highest rank)
    distances['Rank'] = distances['Utility_Score'].rank(ascending=True, method='min').astype(int)
    
    # Merge results back
    results = df.copy()
    results['D+ (PIS)'] = distances['D_plus']
    results['D- (NIS)'] = distances['D_minus']
    results['D_avg (AS)'] = distances['D_avg']
    results['Utility Score'] = distances['Utility_Score']
    results['Rank'] = distances['Rank']
    
    return results.sort_values(by='Rank')
