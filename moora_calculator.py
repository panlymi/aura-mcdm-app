import pandas as pd
import numpy as np

def calculate_moora(data: pd.DataFrame, weights: dict, directions: dict, return_steps: bool = False):
    """
    Computes the Multi-Objective Optimization on the basis of Ratio Analysis (MOORA) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings and scores, or a tuple containing that and a dictionary of calculation steps.
    """
    df = data.copy()
    columns = df.columns
    
    steps_dict = {}
    if return_steps:
        steps_dict['Step 1: Original Decision Matrix'] = df.copy()
        
    # 1. Ratio Normalization
    # x*_ij = x_ij / sqrt(sum(x_ij^2))
    normalized_df = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        col_sum_sq = np.sqrt((df[col]**2).sum())
        if col_sum_sq != 0:
            normalized_df[col] = df[col] / col_sum_sq
        else:
            normalized_df[col] = 0.0
            
    if return_steps:
        steps_dict['Step 2: Ratio Normalized Matrix ($x^*_{ij}$)'] = normalized_df.copy()
            
    # 2. Weighted Normalized Matrix
    # v_ij = w_j * x*_ij
    weighted_df = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        w = weights.get(col, 1.0)
        weighted_df[col] = normalized_df[col] * w
        
    if return_steps:
        steps_dict['Step 3: Weighted Normalized Matrix ($v_{ij}$)'] = weighted_df.copy()
        
    # 3. Normalized Assessment Value (y_i)
    # y_i = sum(maximize criteria) - sum(minimize criteria)
    y_values = pd.Series(0.0, index=df.index)
    
    max_cols = [col for col in columns if directions.get(col, 'maximize') == 'maximize']
    min_cols = [col for col in columns if directions.get(col, 'maximize') == 'minimize']
    
    sum_max = weighted_df[max_cols].sum(axis=1) if max_cols else pd.Series(0.0, index=df.index)
    sum_min = weighted_df[min_cols].sum(axis=1) if min_cols else pd.Series(0.0, index=df.index)
    
    y_values = sum_max - sum_min
    
    if return_steps:
        y_df = pd.DataFrame({
            'Sum (Maximize)': sum_max,
            'Sum (Minimize)': sum_min,
            'Assessment Value ($y_i$)': y_values
        })
        steps_dict['Step 4: Normalized Assessment Value ($y_i$)'] = y_df.copy()
        
    # 4. Ranking (Highest y_i gets highest rank, so Rank 1)
    rank = y_values.rank(ascending=False, method='min').astype(int)
    
    # Format the results
    results = df.copy()
    results['y_i (Assessment Value)'] = y_values
    results['Rank'] = rank
    
    results = results.sort_values(by='Rank')
    
    if return_steps:
        steps_dict['Step 5: Final Result and Ranking'] = results.copy()
        return results, steps_dict
    
    return results
