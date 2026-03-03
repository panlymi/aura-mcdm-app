import pandas as pd
import numpy as np

def calculate_aras(data: pd.DataFrame, weights: dict, directions: dict, return_steps: bool = False):
    """
    Computes the Additive Ratio Assessment (ARAS) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings, scores, and utility degrees, or a tuple containing that and a dictionary of calculation steps.
    """
    df = data.copy()
    columns = df.columns
    
    steps_dict = {}
    if return_steps:
        steps_dict['Step 1: Original Decision Matrix'] = df.copy()
    
    # 1. Determine the optimal alternative (reference row x_0)
    x_0 = pd.Series(index=columns, dtype=float)
    for col in columns:
        if directions.get(col, 'maximize') == 'maximize':
            x_0[col] = df[col].max()
        else:
            x_0[col] = df[col].min()
            
    # Append the optimal alternative to the dataframe
    df_with_opt = pd.concat([pd.DataFrame([x_0], index=['Optimal_0']), df])
    
    if return_steps:
        steps_dict['Step 1b: Decision Matrix with Optimal Alternative ($x_0$)'] = df_with_opt.copy()
    
    # 2. Normalize the decision matrix
    normalized_df = pd.DataFrame(index=df_with_opt.index, columns=columns)
    for col in columns:
        if directions.get(col, 'maximize') == 'maximize':
            # Benefit criteria: x_ij / sum(x_ij)
            col_sum = df_with_opt[col].sum()
            if abs(col_sum) > 1e-9:
                normalized_df[col] = df_with_opt[col] / col_sum
            else:
                normalized_df[col] = 0.0
        else:
            # Cost criteria: (1 / x_ij) / sum(1 / x_ij)
            # Handle division by zero
            # using epsilon to prevent dividing by tiny numbers safely
            reciprocal = 1.0 / (df_with_opt[col].replace(0, np.nan) + 1e-9).fillna(np.inf)
            reciprocal_sum = reciprocal.replace(np.inf, np.nan).sum() 
            if abs(reciprocal_sum) > 1e-9 and pd.notna(reciprocal_sum):
                normalized_df[col] = reciprocal / reciprocal_sum
            else:
                # Fallback if there are zeros or NAs
                normalized_df[col] = 0.0
                
    if return_steps:
        steps_dict['Step 2: Normalized Decision Matrix ($\overline{x}_{ij}$)'] = normalized_df.copy()
                
    # 3. Calculate the weighted normalized matrix
    weighted_df = pd.DataFrame(index=df_with_opt.index, columns=columns)
    for col in columns:
        w = weights.get(col, 1.0)
        weighted_df[col] = normalized_df[col] * w
        
    if return_steps:
        steps_dict['Step 3: Weighted Normalized Matrix ($\hat{x}_{ij}$)'] = weighted_df.copy()
        
    # 4. Calculate the Optimality Function (S_i)
    # S_i is the sum of weighted normalized values for each alternative
    # S_0 is the optimality function for the optimal alternative
    S = weighted_df.sum(axis=1)
    S_0 = S['Optimal_0']
    
    if return_steps:
        s_df = pd.DataFrame(S, columns=['Optimality Function ($S_i$)'])
        steps_dict['Step 4: Optimality Function ($S_i$)'] = s_df.copy()
    
    # 5. Calculate the Utility Degree (K_i)
    # K_i = S_i / S_0
    # K_i ranges from 0 to 1. Higher is better.
    K = pd.Series(index=df.index, dtype=float)
    if abs(S_0) > 1e-9:
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
    
    results = results.sort_values(by='Rank')
    
    if return_steps:
        steps_dict['Step 5: Final Result and Ranking'] = results.copy()
        return results, steps_dict
    
    return results
