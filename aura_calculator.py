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

    # 4. Calculate Distances (using Minkowski distance)
    distances = pd.DataFrame(index=df.index)
    
    if p == 1:
        distances['D_plus'] = (abs(weighted_df - pis)).sum(axis=1)
        distances['D_minus'] = (abs(weighted_df - nis)).sum(axis=1)
        distances['D_avg'] = (abs(weighted_df - as_sol)).sum(axis=1)
    else:
        # Default to Euclidean (p=2) or generic Minkowski if extended later
        distances['D_plus'] = np.power((np.power(abs(weighted_df - pis), p)).sum(axis=1), 1/p)
        distances['D_minus'] = np.power((np.power(abs(weighted_df - nis), p)).sum(axis=1), 1/p)
        distances['D_avg'] = np.power((np.power(abs(weighted_df - as_sol), p)).sum(axis=1), 1/p)

    # Calculate Relative Closeness (Standard TOPSIS base)
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-9
    rc = distances['D_minus'] / (distances['D_plus'] + distances['D_minus'] + epsilon)
    
    # Calculate Relative Distance to Average (normalized)
    max_d_avg = distances['D_avg'].max()
    if max_d_avg == 0:
        rel_d_avg = 0
    else:
        rel_d_avg = distances['D_avg'] / max_d_avg

    # 5. Final Utility Score (Adaptive Formula)
    # Balances standard relative closeness with the normalized distance to the average solution
    # The AURA technique often customizes standard ranking metrics based on alpha
    distances['Utility_Score'] = alpha * rc + (1 - alpha) * rel_d_avg

    # 6. Ranking
    distances['Rank'] = distances['Utility_Score'].rank(ascending=False, method='min').astype(int)
    
    # Merge results back
    results = df.copy()
    results['D+ (PIS)'] = distances['D_plus']
    results['D- (NIS)'] = distances['D_minus']
    results['D_avg (AS)'] = distances['D_avg']
    results['Utility Score'] = distances['Utility_Score']
    results['Rank'] = distances['Rank']
    
    return results.sort_values(by='Rank')
