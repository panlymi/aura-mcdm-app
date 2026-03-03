import numpy as np
import pandas as pd

def calculate_entropy_weights(df, directions):
    """
    Calculates weights using the Entropy Weight Method (EWM).
    
    df: pandas DataFrame containing the decision matrix (numeric only).
    directions: dictionary mapping criterion name to direction ('maximize', 'minimize', or {'type': 'target', 'value': T}).
    
    Returns:
    weights_dict: Dictionary mapping criterion name to its calculated EWM weight.
    steps_dict: Detailed dictionaries and DataFrames for UI verification steps.
    """
    m, n = df.shape
    if m <= 1:
        # Cannot calculate entropy properly with <=1 alternative
        return {col: 1.0/n for col in df.columns}, {}
        
    k = 1.0 / np.log(m)
    
    # Step 2: Normalize
    norm_df = df.copy().astype(float)
    for col in df.columns:
        col_data = df[col]
        min_val = col_data.min()
        max_val = col_data.max()
        
        dir_info = directions.get(col, "maximize")
        if isinstance(dir_info, dict) and dir_info.get("type") == "target":
            target = dir_info.get("value", 0.0)
            diffs = np.abs(col_data - target)
            max_diff = diffs.max()
            if max_diff == 0:
                norm_df[col] = 1.0
            else:
                norm_df[col] = 1.0 - (diffs / max_diff)
        elif dir_info == "minimize":
            if max_val == min_val:
                norm_df[col] = 1.0
            else:
                norm_df[col] = (max_val - col_data) / (max_val - min_val)
        else: # maximize
            if max_val == min_val:
                norm_df[col] = 1.0
            else:
                norm_df[col] = (col_data - min_val) / (max_val - min_val)
                
    # Step 3: Proportion / Probability
    sum_y = norm_df.sum(axis=0)
    p_df = norm_df.copy()
    for col in norm_df.columns:
        if sum_y[col] == 0:
            p_df[col] = 1.0 / m
        else:
            p_df[col] = norm_df[col] / sum_y[col]
            
    # Step 4: Calculate Information Entropy
    e_dict = {}
    for col in p_df.columns:
        p = p_df[col].values
        # Only take p > 0 to avoid ln(0) making NaN
        p_nonzero = p[p > 0]
        if len(p_nonzero) == 0:
            e_j = 1.0
        else:
            e_j = -k * np.sum(p_nonzero * np.log(p_nonzero))
        e_dict[col] = e_j
        
    # Step 5: Degree of diversification
    d_dict = {col: 1.0 - e_dict[col] for col in e_dict}
    
    # Step 6: Final Entropy weights
    sum_d = sum(d_dict.values())
    w_dict = {}
    if sum_d == 0:
        w_dict = {col: 1.0 / n for col in d_dict}
    else:
        w_dict = {col: d_dict[col] / sum_d for col in d_dict}
        
    steps = {
        "Step 2: Normalized Data": norm_df,
        "Step 3: Proportion / Probability": p_df,
        "Step 4: Information Entropy (e_j)": e_dict,
        "Step 5: Degree of Diversification (d_j)": d_dict,
        "Step 6: Final Entropy Weights": w_dict
    }

    return w_dict, steps
