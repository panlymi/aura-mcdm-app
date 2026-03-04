import pandas as pd
import numpy as np
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def calculate_topsis(data: pd.DataFrame, weights: dict, directions: dict, return_steps: bool = False):
    """
    Computes the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) MCDM method.
    
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

    # 1. Vector Normalization
    # r_ij = x_ij / sqrt(sum(x_ij^2))
    normalized_df = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        col_sum_sq = np.sqrt((df[col]**2).sum())
        if abs(col_sum_sq) > 1e-9:
            normalized_df[col] = df[col] / col_sum_sq
        else:
            normalized_df[col] = 0.0
            
    if return_steps:
        steps_dict['Step 2: Normalized Decision Matrix ($r_{ij}$)'] = normalized_df.copy()

    # 2. Weighted Normalized Matrix
    # v_ij = w_j * r_ij
    weighted_df = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        w = weights.get(col, 1.0)
        weighted_df[col] = normalized_df[col] * w
        
    if return_steps:
        steps_dict['Step 3: Weighted Normalized Matrix ($v_{ij}$)'] = weighted_df.copy()

    # 3. Ideal (PIS) and Anti-Ideal (NIS) Solutions
    pis = pd.Series(index=columns, dtype=float)
    nis = pd.Series(index=columns, dtype=float)
    
    for col in columns:
        direction = directions.get(col, 'maximize')
        if direction == 'maximize' or direction == 'target':
            pis[col] = weighted_df[col].max()
            nis[col] = weighted_df[col].min()
        else: # minimize
            pis[col] = weighted_df[col].min()
            nis[col] = weighted_df[col].max()
            
    if return_steps:
        steps_dict['Step 4: Ideal and Anti-Ideal Solutions'] = pd.DataFrame({
            'PIS (A+)': pis,
            'NIS (A-)': nis
        }).T.copy()
        
    # 4. Separation Measures (Euclidean Distance)
    # D+_i = sqrt(sum((v_ij - PIS_j)^2))
    # D-_i = sqrt(sum((v_ij - NIS_j)^2))
    
    d_plus = pd.Series(0.0, index=df.index)
    d_minus = pd.Series(0.0, index=df.index)
    
    for idx in df.index:
        d_plus[idx] = np.sqrt(((weighted_df.loc[idx] - pis) ** 2).sum())
        d_minus[idx] = np.sqrt(((weighted_df.loc[idx] - nis) ** 2).sum())
        
    if return_steps:
        steps_dict['Step 5: Separation Measures'] = pd.DataFrame({
            'D+ (Distance to PIS)': d_plus,
            'D- (Distance to NIS)': d_minus
        }).copy()

    # 5. Relative Closeness to Ideal Solution
    # C_i = D-_i / (D+_i + D-_i)
    
    c_i = pd.Series(0.0, index=df.index)
    for idx in df.index:
        den = d_plus[idx] + d_minus[idx]
        if den > 1e-9:
            c_i[idx] = d_minus[idx] / den
        else:
            c_i[idx] = 0.0
            
    if return_steps:
        closeness_df = pd.DataFrame({
            'D+ (Ideal)': d_plus,
            'D- (Anti-Ideal)': d_minus,
            'Relative Closeness (C_i)': c_i
        })
        steps_dict['Step 6: Relative Closeness'] = closeness_df.copy()

    # 6. Final Ranking
    rank = c_i.rank(ascending=False, method='min').astype(int)
    
    # Format the results
    results = df.copy()
    results['D+ (Ideal)'] = d_plus
    results['D- (Anti-Ideal)'] = d_minus
    results['Relative Closeness (C_i)'] = c_i
    results['Rank'] = rank
    
    # Sort by rank, then naturally by alternative name (index)
    results['sort_index'] = results.index.map(lambda x: tuple(natural_sort_key(x)))
    results = results.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    
    if return_steps:
        steps_dict['Step 7: Final Result and Ranking'] = results.copy()
        return results, steps_dict
    
    return results
