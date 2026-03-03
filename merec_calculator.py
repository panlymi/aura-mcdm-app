import numpy as np
import pandas as pd

def calculate_merec_weights(df, directions):
    """
    Calculates weights using the MEREC (Method based on the Removal Effects of Criteria).
    
    df: pandas DataFrame containing the decision matrix (numeric only).
        Assumes strictly positive elements (x_ij > 0).
    directions: dictionary mapping criterion name to direction ('maximize', 'minimize', or 'target' dict).
    
    Returns:
    weights_dict: Dictionary mapping criterion name to its calculated MEREC weight.
    steps: Dictionary with intermediate steps for UI display.
    """
    n, m = df.shape # n = alternatives, m = criteria
    if n <= 1 or m <= 1:
        return {col: 1.0/m for col in df.columns}, {}
        
    epsilon = 1e-6
    x_df = df.copy()
    # Ensure strictly positive
    x_df[x_df <= 0] = epsilon
    
    # Step 2: Normalize decision matrix (N)
    norm_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        col_data = x_df[col]
        min_val = col_data.min()
        max_val = col_data.max()
        
        dir_info = directions.get(col, "maximize")
        is_target = isinstance(dir_info, dict) and dir_info.get("type") == "target"
        
        if is_target:
            target = dir_info.get("value", 0.0)
            diffs = np.abs(col_data - target)
            diffs[diffs <= 0] = epsilon
            max_diff = diffs.max()
            norm_df[col] = diffs / max_diff if max_diff > 0 else 1.0
        else:
            if dir_info == "minimize":
                norm_df[col] = col_data / max_val if max_val > 0 else col_data
            else: # maximize
                norm_df[col] = min_val / col_data
                
    # Step 3: Calculate overall performance (S_i)
    # Handle possible exactly 1.0 resulting in ln(1) = 0
    # The formula uses absolute value of the log
    ln_N_abs = np.abs(np.log(norm_df.astype(float)))
    
    # S_i = ln( 1 + (1/m * sum_j(|ln(N_ij)|)) )
    sum_ln_N = ln_N_abs.sum(axis=1)
    S_i = np.log(1 + (1 / m) * sum_ln_N)
    
    # Step 4: Calculate performance without each criterion (S'_ij)
    S_prime = pd.DataFrame(index=df.index, columns=df.columns)
    for j in df.columns:
        # Sum of all EXCEPT criterion j
        sum_ln_N_excl_j = sum_ln_N - ln_N_abs[j]
        S_prime[j] = np.log(1 + (1 / m) * sum_ln_N_excl_j)
        
    # Step 5: Compute summation of absolute deviations (E_j)
    E_j = pd.Series(index=df.columns, dtype=float)
    for j in df.columns:
        E_j[j] = np.sum(np.abs(S_prime[j] - S_i))
        
    # Step 6: Determine final weights (w_j)
    sum_E = E_j.sum()
    if sum_E == 0:
        w_dict = {col: 1.0 / m for col in df.columns}
    else:
        w_dict = (E_j / sum_E).to_dict()
        
    steps = {
        "Step 2: Normalized Decision Matrix (N)": norm_df,
        "Step 3: Logarithmic Penalty (|ln(N_ij)|)": ln_N_abs,
        "Step 3: Overall Performance (S_i)": pd.DataFrame(S_i, columns=["S_i"]),
        "Step 4: Performance Without Criterion (S'_ij)": S_prime,
        "Step 5: Removal Effects (E_j)": E_j.to_frame(name="E_j"),
        "Step 6: Final Weights (w_j)": pd.DataFrame.from_dict(w_dict, orient='index', columns=['Weight'])
    }
    
    return w_dict, steps
