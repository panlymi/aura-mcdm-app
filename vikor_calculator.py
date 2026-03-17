import pandas as pd
import numpy as np
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def calculate_vikor(data: pd.DataFrame, weights: dict, directions: dict, v_param: float = 0.5, return_steps: bool = False):
    """
    Computes the VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize', 'minimize', or 'target' for each criterion.
        v_param (float): Weight of the strategy of 'the majority of criteria' (or 'the maximum group utility').
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings and scores, or a tuple containing that and a dictionary of calculation steps.
    """
    df = data.copy()
    columns = df.columns
    
    steps_dict = {}
    if return_steps:
        steps_dict['Step 1: Original Decision Matrix'] = df.copy()

    # 1. Determine the best (f*) and worst (f-) values
    f_star = pd.Series(index=columns, dtype=float)
    f_minus = pd.Series(index=columns, dtype=float)
    
    # For target type, we will internally transform to distance from target and make it minimize
    transformed_df = df.copy()
    internal_dirs = {}
    
    for col in columns:
        direction = directions.get(col, 'maximize')
        
        if isinstance(direction, dict):
            dir_type = direction.get('type', 'maximize')
            target_val = direction.get('value', 0.0)
        else:
            dir_type = direction
            target_val = 0.0
            
        if dir_type == 'maximize':
            f_star[col] = df[col].max()
            f_minus[col] = df[col].min()
            internal_dirs[col] = 'maximize'
        elif dir_type == 'target':
            # Transform to absolute distance
            transformed_df[col] = (df[col] - target_val).abs()
            f_star[col] = transformed_df[col].min() # Best is 0
            f_minus[col] = transformed_df[col].max() # Worst is max distance
            internal_dirs[col] = 'minimize'
        else: # minimize
            f_star[col] = df[col].min()
            f_minus[col] = df[col].max()
            internal_dirs[col] = 'minimize'
            
    if return_steps:
        steps_dict['Step 2: Best (f*) and Worst (f-) Values'] = pd.DataFrame({
            'Best (f*)': f_star,
            'Worst (f-)': f_minus
        }).T.copy()

    # 2. Compute S_i and R_i
    # normalized distance: (f* - x) / (f* - f-) for max
    # for min: (x - f*) / (f- - f*) = (f* - x) / (f* - f-)   Wait.
    # Actually, standard formula for beneficial (maximize): (f* - x) / (f* - f-)
    # Standard formula for non-beneficial (minimize): (x - f*) / (f- - f*)
    
    norm_dist = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for col in columns:
        denom = f_star[col] - f_minus[col]
        # To avoid division by zero
        if abs(denom) < 1e-9:
            norm_dist[col] = 0.0
        else:
            if internal_dirs[col] == 'maximize':
                norm_dist[col] = (f_star[col] - transformed_df[col]) / denom
            else: # minimize
                norm_dist[col] = (transformed_df[col] - f_star[col]) / (f_minus[col] - f_star[col])
                
    weighted_norm_dist = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for col in columns:
        w = weights.get(col, 1.0)
        weighted_norm_dist[col] = norm_dist[col] * w
        
    if return_steps:
        steps_dict['Step 3: Weighted Normalized Distance Matrix'] = weighted_norm_dist.copy()
        
    s_i = weighted_norm_dist.sum(axis=1) # Utility measure
    r_i = weighted_norm_dist.max(axis=1) # Regret measure
    
    if return_steps:
        steps_dict['Step 4: Utility (S_i) and Regret (R_i) Measures'] = pd.DataFrame({
            'S_i': s_i,
            'R_i': r_i
        }).copy()

    # 3. Compute Q_i
    s_star, s_minus = s_i.min(), s_i.max()
    r_star, r_minus = r_i.min(), r_i.max()
    
    q_i = pd.Series(index=df.index, dtype=float)
    
    for idx in df.index:
        # Avoid division by zero
        s_term = 0.0
        r_term = 0.0
        
        if abs(s_minus - s_star) > 1e-9:
            s_term = (s_i[idx] - s_star) / (s_minus - s_star)
            
        if abs(r_minus - r_star) > 1e-9:
            r_term = (r_i[idx] - r_star) / (r_minus - r_star)
            
        q_i[idx] = v_param * s_term + (1 - v_param) * r_term
        
    # 4. Final Ranking (Ascending order, smaller Q is better)
    rank = q_i.rank(ascending=True, method='min').astype(int)
    
    # Format the results
    results = df.copy()
    results['S_i (Utility)'] = s_i
    results['R_i (Regret)'] = r_i
    results['Q_i (VIKOR Index)'] = q_i
    results['Rank'] = rank
    
    # Sort by rank, then naturally by alternative name (index)
    results['sort_index'] = results.index.map(lambda x: tuple(natural_sort_key(x)))
    results = results.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    
    if return_steps:
        # Add S*, S-, R*, R- context
        steps_dict['Step 5: VIKOR Index (Q_i) Parameters'] = {
            'S* (Min S_i)': s_star,
            'S- (Max S_i)': s_minus,
            'R* (Min R_i)': r_star,
            'R- (Max R_i)': r_minus,
            'v (Weight of strategy)': v_param
        }
        steps_dict['Step 6: Final Result and Ranking'] = results.copy()
        return results, steps_dict
    
    return results
