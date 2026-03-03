import pandas as pd
import numpy as np

def calculate_fuzzy_aras(data: pd.DataFrame, weights: dict, directions: dict, return_steps: bool = False):
    """
    Computes the Fuzzy Additive Ratio Assessment (Fuzzy ARAS) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix with Triangular Fuzzy Numbers (l, m, u).
        weights (dict): Crisp weights (will be treated as (w, w, w)) or fuzzy weights.
        directions (dict): 'maximize' or 'minimize' for each criterion.
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: Contains fuzzy and defuzzified scores, and rankings.
                               If return_steps=True, returns (results_df, steps_dict).
    """
    # Create copies
    df = data.copy()
    columns = df.columns
    steps = {}
    
    # Ensure weights are TFNs, if crisp just repeat (w, w, w)
    tfn_weights = {}
    for col, w in weights.items():
        if isinstance(w, tuple) and len(w) == 3:
            tfn_weights[col] = w
        else:
            tfn_weights[col] = (float(w), float(w), float(w))
            
    steps['Step 0: Fuzzy Weights'] = pd.DataFrame([tfn_weights], index=['Weights']).T

    # Helper to extract l, m, u vectors for a specific column
    def get_lmu(series):
        l = np.array([x[0] for x in series])
        m = np.array([x[1] for x in series])
        u = np.array([x[2] for x in series])
        return l, m, u
        
    # 1. Determine the optimal alternative TFN x_0
    x_0 = {}
    for col in columns:
        l, m, u = get_lmu(df[col])
        if directions.get(col, 'maximize') == 'maximize':
            x_0[col] = (np.max(l), np.max(m), np.max(u))
        else:
            x_0[col] = (np.min(l), np.min(m), np.min(u))
            
    # Include x_0 in calculations
    df_opt = pd.DataFrame([x_0], index=['Optimal_0'])
    df_combined = pd.concat([df_opt, df])

    steps['Step 1: Decision Matrix with Optimal TFN ($x_0$)'] = df_combined.copy()
    
    # 2. Normalize the fuzzy decision matrix
    norm_df = pd.DataFrame(index=df_combined.index, columns=columns)
    for col in columns:
        l, m, u = get_lmu(df_combined[col])
        
        if directions.get(col, 'maximize') == 'maximize':
            # Benefit criteria
            sum_u = np.sum(u)
            sum_m = np.sum(m)
            sum_l = np.sum(l)
            # x_ij / sum(x_kj) -> (l/sum_u, m/sum_m, u/sum_l)
            norm_col = [(l[i]/sum_u if abs(sum_u) > 1e-9 else 0.0, 
                         m[i]/sum_m if abs(sum_m) > 1e-9 else 0.0, 
                         u[i]/sum_l if abs(sum_l) > 1e-9 else 0.0) for i in range(len(l))]
            norm_df[col] = norm_col
        else:
            # Cost criteria
            # 1. Take reciprocal (1/u, 1/m, 1/l)
            # 2. Sum them up
            # 3. Divide reciprocal by sum
            inv_l = 1.0 / np.where(u == 0, 1e-9, u)
            inv_m = 1.0 / np.where(m == 0, 1e-9, m)
            inv_u = 1.0 / np.where(l == 0, 1e-9, l)
            
            sum_inv_u = np.sum(inv_u)
            sum_inv_m = np.sum(inv_m)
            sum_inv_l = np.sum(inv_l)
            
            norm_col = [(inv_l[i]/sum_inv_u if abs(sum_inv_u) > 1e-9 else 0.0,
                         inv_m[i]/sum_inv_m if abs(sum_inv_m) > 1e-9 else 0.0,
                         inv_u[i]/sum_inv_l if abs(sum_inv_l) > 1e-9 else 0.0) for i in range(len(inv_l))]
            norm_df[col] = norm_col

    steps['Step 2: Normalized Fuzzy Decision Matrix'] = norm_df.copy()
        
    # 3. Weighted normalized matrix & 4. Optimality Function (S_i)
    # S_i = sum(normalized_ij * weight_j)
    fuzzy_S = pd.Series(index=df_combined.index, dtype=object)
    weighted_df = pd.DataFrame(index=df_combined.index, columns=columns)
    
    for idx in df_combined.index:
        sum_l, sum_m, sum_u = 0.0, 0.0, 0.0
        for col in columns:
            r_l, r_m, r_u = norm_df.at[idx, col]
            w_l, w_m, w_u = tfn_weights[col]
            # Multiplication of TFNs: (a,b,c) * (x,y,z) roughly = (ax, by, cz) if all positive
            val_l = r_l * w_l
            val_m = r_m * w_m
            val_u = r_u * w_u
            weighted_df.at[idx, col] = (val_l, val_m, val_u)
            sum_l += val_l
            sum_m += val_m
            sum_u += val_u
        fuzzy_S[idx] = (sum_l, sum_m, sum_u)

    steps['Step 3: Weighted Normalized Fuzzy Decision Matrix'] = weighted_df.copy()
    steps['Step 4: Fuzzy Optimality Function ($S_i$)'] = fuzzy_S.to_frame(name='Fuzzy $S_i$')
        
    # 5. Defuzzification (Center of Area)
    # defuzzified = (l + m + u) / 3
    def defuzzify(tfn):
        return (tfn[0] + tfn[1] + tfn[2]) / 3.0
        
    S_crisp = fuzzy_S.apply(defuzzify)
    S_0_crisp = S_crisp['Optimal_0']

    steps['Step 5: Defuzzified Crisp $S_i$'] = S_crisp.to_frame(name='Crisp $S_i$')
    steps['Step 5: Optimal $S_0$ (Crisp)'] = S_0_crisp
    
    # 6. Calculate Utility Degree K_i and Rank
    K = pd.Series(index=df.index, dtype=float)
    if abs(S_0_crisp) > 1e-9:
        for idx in df.index:
            K[idx] = S_crisp[idx] / S_0_crisp
    else:
        K[:] = 0.0
        
    steps['Step 6: Utility Degree ($K_i$)'] = K.to_frame(name='Utility Degree $K_i$')

    rank = K.rank(ascending=False, method='min').astype(int)
    
    # Format results
    results = df.copy()
    results['S_i (Fuzzy)'] = fuzzy_S[df.index].apply(lambda x: f"({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})")
    results['S_i (Crisp)'] = S_crisp[df.index]
    results['K_i (Utility Degree)'] = K
    results['Rank'] = rank
    
    results_sorted = results.sort_values(by='Rank')
    steps['Step 7: Final Result and Ranking'] = results_sorted.copy()
    
    if return_steps:
        return results_sorted, steps
    return results_sorted
