import pandas as pd
import numpy as np

def calculate_syai(data: pd.DataFrame, weights: dict, directions: dict, beta: float = 0.5):
    """
    Computes the Simplified Yielded Aggregation Index (SYAI) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix.
        weights (dict): A dictionary of criteria weights (default sum to 1).
        directions (dict): 'maximize' or 'minimize' for each criterion.
        beta (float): Tunable parameter for closeness score. Default is 0.5.
        
    Returns:
        pd.DataFrame: A dataframe containing SYAI calculation steps and final ranking.
    """
    # Create copies
    df = data.copy()
    columns = df.columns
    
    # Validation: Ensure weights sum to 1.0 (handled in app.py typically, but good practice here)
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        w_norm = {k: v / total_weight for k, v in weights.items()}
    else:
        w_norm = weights.copy()
        
    C = 0.01  # Fixed constant to prevent zero outputs
    
    # 1. Normalize the decision matrix
    norm_df = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        col_data = df[col]
        max_val = col_data.max()
        min_val = col_data.min()
        r_val = max_val - min_val
        
        # Guard against zero range (all values identical)
        if r_val == 0:
            norm_df[col] = 1.0  # If all items are identical, they all perfectly match the ideal
            continue
            
        if directions.get(col, 'maximize') == 'maximize':
            # Benefit criterion -> ideal point is the maximum
            x_star = max_val
        else:
            # Cost criterion -> ideal point is the minimum
            x_star = min_val
            
        # N_ij = C + (1 - C) * (1 - (|x_ij - x^*| / R))
        norm_col = C + (1 - C) * (1 - (np.abs(col_data - x_star) / r_val))
        norm_df[col] = norm_col
        
    # 2. Calculate Weighted Normalized Matrix
    weighted_df = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        weighted_df[col] = norm_df[col] * w_norm[col]
        
    # 3. Determine Yielded-Ideal (A+) and Anti-Ideal (A-) Solutions
    A_plus = weighted_df.max()
    A_minus = weighted_df.min()
    
    # 4. Compute Distances and Closeness Score
    D_plus = pd.Series(index=df.index, dtype=float)
    D_minus = pd.Series(index=df.index, dtype=float)
    
    for idx in df.index:
        row = weighted_df.loc[idx]
        # D+ = sum(|v_ij - A+_j|)
        D_plus[idx] = np.sum(np.abs(row - A_plus))
        # D- = sum(|v_ij - A-_j|)
        D_minus[idx] = np.sum(np.abs(row - A_minus))
        
    # Closeness Score (D_i) = ((1 - beta) * D-) / (beta * D+ + (1 - beta) * D-)
    # With a guard for divide by zero if both D+ and D- are exactly 0 (occurs if 1 row)
    D_score = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        numerator = (1 - beta) * D_minus[idx]
        denominator = (beta * D_plus[idx]) + numerator
        if denominator == 0:
            D_score[idx] = 1.0 # If distance to both ideal and anti-ideal is 0, it is the only alternative
        else:
            D_score[idx] = numerator / denominator
            
    # 5. Rank the alternatives
    rank = D_score.rank(ascending=False, method='min').astype(int)
    
    # Format results
    results = df.copy()
    results['D+ (Dist to Ideal)'] = D_plus
    results['D- (Dist to Anti-Ideal)'] = D_minus
    results['Closeness Score (D_i)'] = D_score
    results['Rank'] = rank
    
    return results.sort_values(by='Rank')
