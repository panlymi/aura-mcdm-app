import pandas as pd
import numpy as np

from mcdm.criteria import CriterionType, validate_method_capabilities
from mcdm.ranking import natural_sort_key, rank_scores
from mcdm.validation import validate_crisp_matrix, validate_method_matrix, validate_weights

def calculate_saw(data: pd.DataFrame, weights: dict, directions: dict, return_steps: bool = False):
    """
    Computes the Simple Additive Weighting (SAW) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings and scores, or a tuple containing that and a dictionary of calculation steps.
    """
    df = validate_crisp_matrix(data)
    columns = df.columns
    preferences = validate_method_capabilities("SAW", columns, directions)
    validate_method_matrix("SAW", df, directions)
    normalized_weights = validate_weights(weights, columns, normalize=True)
    
    steps_dict = {}
    if return_steps:
        steps_dict['Step 1: Original Decision Matrix'] = df.copy()

    # 1. Normalization
    # Maximize: r_ij = x_ij / x_j^max
    # Minimize: r_ij = x_j^min / x_ij
    normalized_df = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for col in columns:
        col_max = df[col].max()
        col_min = df[col].min()

        if preferences[col].kind is CriterionType.BENEFIT:
            if abs(col_max) > 1e-9:
                normalized_df[col] = df[col] / col_max
            else:
                normalized_df[col] = 0.0
        else: # minimize
            for idx in df.index:
                val = df.loc[idx, col]
                if abs(val) > 1e-9:
                    normalized_df.loc[idx, col] = col_min / val
                else:
                    normalized_df.loc[idx, col] = 0.0
            
    if return_steps:
        steps_dict['Step 2: Normalized Decision Matrix'] = normalized_df.copy()

    # 2. Weighted Normalized Matrix
    weighted_df = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for col in columns:
        w = normalized_weights[col]
        weighted_df[col] = normalized_df[col] * w
        
    if return_steps:
        steps_dict['Step 3: Weighted Normalized Matrix'] = weighted_df.copy()

    # 3. Final Score
    v_i = weighted_df.sum(axis=1)
    
    # 4. Final Ranking
    rank = rank_scores(v_i, ascending=False)
    
    # Format the results
    results = df.copy()
    results['V_i (SAW Score)'] = v_i
    results['Rank'] = rank
    
    # Sort by rank, then naturally by alternative name (index)
    results['sort_index'] = results.index.map(lambda x: tuple(natural_sort_key(x)))
    results = results.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    
    if return_steps:
        steps_dict['Step 4: Final Result and Ranking'] = results.copy()
        return results, steps_dict
    
    return results
