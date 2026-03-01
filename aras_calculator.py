import pandas as pd
import numpy as np

def calculate_aras(data: pd.DataFrame, weights: dict, directions: dict):
    """
    Computes the Additive Ratio Assessment (ARAS) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        
    Returns:
        pd.DataFrame: A dataframe containing the rankings, scores, and utility degrees.
    """
    df = data.copy()
    columns = df.columns
    
    # 1. Determine the optimal alternative (reference row x_0)
    x_0 = pd.Series(index=columns, dtype=float)
    for col in columns:
        if directions.get(col, 'maximize') == 'maximize':
            x_0[col] = df[col].max()
        else:
            x_0[col] = df[col].min()
            
    # Append the optimal alternative to the dataframe
    df_with_opt = pd.concat([pd.DataFrame([x_0], index=['Optimal_0']), df])
    
    # 2. Normalize the decision matrix
    normalized_df = pd.DataFrame(index=df_with_opt.index, columns=columns)
    for col in columns:
        if directions.get(col, 'maximize') == 'maximize':
            # Benefit criteria: x_ij / sum(x_ij)
            col_sum = df_with_opt[col].sum()
            if col_sum != 0:
                normalized_df[col] = df_with_opt[col] / col_sum
            else:
                normalized_df[col] = 0.0
        else:
            # Cost criteria: (1 / x_ij) / sum(1 / x_ij)
            # Handle division by zero
            reciprocal = 1.0 / df_with_opt[col].replace(0, np.nan)
            reciprocal_sum = reciprocal.sum()
            if reciprocal_sum != 0 and pd.notna(reciprocal_sum):
                normalized_df[col] = reciprocal / reciprocal_sum
            else:
                # Fallback if there are zeros or NAs
                normalized_df[col] = 0.0
                
    # 3. Calculate the weighted normalized matrix
    weighted_df = pd.DataFrame(index=df_with_opt.index, columns=columns)
    for col in columns:
        w = weights.get(col, 1.0)
        weighted_df[col] = normalized_df[col] * w
        
    # 4. Calculate the Optimality Function (S_i)
    # S_i is the sum of weighted normalized values for each alternative
    # S_0 is the optimality function for the optimal alternative
    S = weighted_df.sum(axis=1)
    S_0 = S['Optimal_0']
    
    # 5. Calculate the Utility Degree (K_i)
    # K_i = S_i / S_0
    # K_i ranges from 0 to 1. Higher is better.
    K = pd.Series(index=df.index, dtype=float)
    if S_0 != 0:
        for idx in df.index:
            K[idx] = S[idx] / S_0
    else:
        K[:] = 0.0
        
    # 6. Ranking (Highest K_i gets highest rank, so Rank 1)
    # Note: Unlike AURA where lowest score is best, in ARAS highest Utility Degree is best
    rank = K.rank(ascending=False, method='min').astype(int)
    
    # Format the results
    results = df.copy()
    results['S (Optimality)'] = S[df.index]
    results['K (Utility Degree)'] = K
    results['Rank'] = rank
    
    return results.sort_values(by='Rank')
