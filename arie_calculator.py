import pandas as pd
import numpy as np

from mcdm.criteria import CriterionType, validate_method_capabilities
from mcdm.ranking import natural_sort_key, rank_scores
from mcdm.validation import validate_crisp_matrix, validate_method_matrix, validate_weights

def calculate_arie(matrix, weights, directions, gamma=1.0, kappa=0.5, return_steps=False):
    """
    Calculates the Adaptive Ranking with Ideal Evaluation (ARIE) method.
    
    Args:
        matrix: pandas DataFrame containing the decision matrix (alternatives x criteria)
        weights: dict of weights for each criterion
        directions: dict of directions ('maximize', 'minimize', or {'type': 'target', 'value': T})
        gamma: sensitivity parameter for similarity (default 1.0)
        kappa: balancing parameter for relative closeness (default 0.5)
        return_steps: boolean, if True returns a dictionary of step-by-step DataFrames
        
    Returns:
        results_df (if return_steps=False)
        (results_df, steps_dict) (if return_steps=True)
    """
    if gamma <= 0:
        raise ValueError("gamma must be greater than zero.")
    if not 0 <= kappa <= 1:
        raise ValueError("kappa must be between 0 and 1.")

    df = validate_crisp_matrix(matrix)
    criteria = df.columns
    preferences = validate_method_capabilities("ARIE", criteria, directions)
    validate_method_matrix("ARIE", df, directions)
    norm_weights = validate_weights(weights, criteria, normalize=True)
    EPSILON = 1e-9  # to prevent division by zero
    
    # Step 1: Construct the Decision Matrix
    steps = {}
    if return_steps:
        steps['Step 1: Decision Matrix'] = df.copy()

    # Step 2: Normalize the Decision Matrix
    r_matrix = pd.DataFrame(index=df.index, columns=criteria)
    for c in criteria:
        x_j = df[c]
        x_max = x_j.max()
        x_min = x_j.min()
        
        preference = preferences[c]
        if preference.kind is CriterionType.TARGET:
            target_val = float(preference.target_value)
            # target-type (Goal) criterion
            max_diff = max(abs(x_max - target_val), abs(x_min - target_val))
            if abs(max_diff) < 1e-9:
                r_matrix[c] = 1.0
            else:
                r_matrix[c] = 1 - (abs(x_j - target_val) / max_diff)
        else:
            if preference.kind is CriterionType.BENEFIT:
                if abs(x_max) < 1e-9:
                    r_matrix[c] = 0.0
                else:
                    r_matrix[c] = x_j / x_max
            else:
                r_matrix[c] = x_min / (x_j + EPSILON) # prevent div by zero if x_ij is 0
    
    if return_steps:
        steps['Step 2: Normalized Decision Matrix'] = r_matrix.copy()
        
    # Step 3: Calculate Weighted Normalized Decision Matrix
    v_matrix = pd.DataFrame(index=df.index, columns=criteria)
    for c in criteria:
        v_matrix[c] = r_matrix[c] * norm_weights[c]
        
    if return_steps:
        steps['Step 3: Weighted Normalized Matrix'] = v_matrix.copy()
        
    # Step 4: Compute Similarity to Ideal and Anti-Ideal Solutions
    v_max = v_matrix.max()
    v_min = v_matrix.min()
    
    ideal_dict = {'Ideal Solution (v_max)': v_max, 'Anti-Ideal Solution (v_min)': v_min}
    if return_steps:
        steps['Step 4a: Ideal and Anti-Ideal Solutions'] = pd.DataFrame(ideal_dict).T
        
    sim_best_parts = pd.DataFrame(index=df.index, columns=criteria)
    sim_worst_parts = pd.DataFrame(index=df.index, columns=criteria)
    
    for c in criteria:
        denom_best = v_max[c] + EPSILON
        sim_best_parts[c] = (v_matrix[c] / denom_best) ** gamma
        
        denom_worst = v_matrix[c] + EPSILON
        sim_worst_parts[c] = (v_min[c] / denom_worst) ** gamma
        
    sim_best = sim_best_parts.sum(axis=1)
    sim_worst = sim_worst_parts.sum(axis=1)
    
    if return_steps:
        steps['Step 4b: Similarity Computations'] = pd.DataFrame({
            'Sim_best': sim_best,
            'Sim_worst': sim_worst
        })

    # Step 5: Compute Relative Closeness and Ranking
    # RC_i = (kappa * Sim_best) / (kappa * Sim_best + (1 - kappa) * Sim_worst)
    rc_scores = (kappa * sim_best) / (kappa * sim_best + (1 - kappa) * sim_worst + EPSILON)
    
    results_df = pd.DataFrame({
        'Sim_best': sim_best,
        'Sim_worst': sim_worst,
        'Relative Closeness (RC_i)': rc_scores
    }, index=df.index)
    
    # Rank descendingly
    results_df['Rank'] = rank_scores(results_df['Relative Closeness (RC_i)'], ascending=False)
    
    # Sort by rank, then naturally by alternative name (index)
    results_df['sort_index'] = results_df.index.map(lambda x: tuple(natural_sort_key(x)))
    results_df = results_df.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    
    if return_steps:
        steps['Step 5: Final Result and Ranking'] = results_df.copy()
        return results_df, steps
        
    return results_df
