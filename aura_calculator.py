import pandas as pd
import numpy as np
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def calculate_aura(data: pd.DataFrame, weights: dict, directions: dict, alpha: float, p: int = 2, return_steps: bool = False):
    """
    Computes the Adaptive Utility Ranking Algorithm (AURA).
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        alpha (float): The balance parameter (0 to 1).
        p (int): Distance metric parameter (1 for Manhattan, 2 for Euclidean).
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings, scores, and distance metrics. 
                               If return_steps=True, returns (results_df, steps_dict).
    """
    if not (0.0 <= alpha <= 1.0):
        # Enforce bounds
        alpha = max(0.0, min(1.0, alpha))
        
    df = data.copy()
    steps = {}
    
    # 1. Normalization
    normalized_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        max_val = df[col].max()
        min_val = df[col].min()
        
        # Handle edge case where all values are the same
        if max_val == min_val:
            normalized_df[col] = 1.0
            continue
            
        direction_info = directions.get(col, 'maximize')
        
        if isinstance(direction_info, dict) and direction_info.get("type") == "target":
            r_j = direction_info.get("value", max_val)
        elif direction_info == 'maximize':
            r_j = max_val
        else: # minimize
            r_j = min_val
            
        normalized_df[col] = 1.0 - (abs(df[col] - r_j) / max((max_val - min_val), 1e-9))

    steps['Step 1: Normalized Decision Matrix'] = normalized_df.copy()

    # 2. Weighted Matrix
    weighted_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        w = weights.get(col, 1.0)
        weighted_df[col] = normalized_df[col] * w

    steps['Step 2: Weighted Normalized Matrix'] = weighted_df.copy()

    # 3. Determine Ideal Solutions (PIS, NIS, AS)
    pis = weighted_df.max()
    nis = weighted_df.min()
    as_sol = weighted_df.mean()
    
    steps['Step 3: Ideal Solutions'] = {
        'PIS (Positive Ideal Solution)': pis,
        'NIS (Negative Ideal Solution)': nis,
        'AS (Average Solution)': as_sol
    }

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

    steps['Step 4a: Raw Distances'] = pd.DataFrame({
        'd+ (Raw dist to PIS)': d_pos_raw,
        'd- (Raw dist to NIS)': d_neg_raw,
        'd_avg (Raw dist to AS)': d_avg_raw
    })

    # Apply Correction Factor (Penalty Term)
    def correct(d):
        sigma = np.max(d) - np.min(d)
        return d + sigma * (d ** 2), sigma

    d_plus_corrected, sigma_plus = correct(d_pos_raw)
    d_minus_corrected, sigma_minus = correct(d_neg_raw)
    d_avg_corrected, sigma_avg = correct(d_avg_raw)

    distances = pd.DataFrame(index=df.index)
    distances['D_plus'] = d_plus_corrected
    distances['D_minus'] = d_minus_corrected
    distances['D_avg'] = d_avg_corrected

    steps['Step 4b: Corrected Distances'] = distances.copy()
    steps['Step 4b: Correction Factors'] = {
        'Sigma+': sigma_plus,
        'Sigma-': sigma_minus,
        'Sigma_avg': sigma_avg
    }

    # 5. Final Utility Score (Exact AURA Formula)
    distances['Utility_Score'] = (alpha * (distances['D_plus'] - distances['D_minus']) + (1.0 - alpha) * distances['D_avg']) / 2.0

    steps['Step 5: Final Utility Score'] = distances[['Utility_Score']].copy()

    # 6. Ranking (Lowest score gets highest rank)
    distances['Rank'] = distances['Utility_Score'].rank(ascending=True, method='min').astype(int)
    
    # Merge results back
    results = df.copy()
    results['D+ (PIS)'] = distances['D_plus']
    results['D- (NIS)'] = distances['D_minus']
    results['D_avg (AS)'] = distances['D_avg']
    results['Utility Score'] = distances['Utility_Score']
    results['Rank'] = distances['Rank']
    
    # Sort by rank, then naturally by alternative name (index)
    results['sort_index'] = results.index.map(lambda x: tuple(natural_sort_key(x)))
    results = results.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    steps['Step 6: Final Result and Ranking'] = results.copy()
    
    if return_steps:
        return results, steps
    return results
