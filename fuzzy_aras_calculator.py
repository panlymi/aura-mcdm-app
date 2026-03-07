import pandas as pd
import numpy as np
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

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
    
    # Ensure weights are fuzzy numbers, if crisp just repeat
    tfn_weights = {}
    for col, w in weights.items():
        if isinstance(w, tuple) and len(w) in [3, 4]:
            tfn_weights[col] = w
        else:
            # Check length of the first item in the column to see if it's TrFN
            first_val = df[col].iloc[0] if len(df) > 0 else (0,0,0)
            if isinstance(first_val, tuple) and len(first_val) == 4:
                tfn_weights[col] = (float(w), float(w), float(w), float(w))
            else:
                tfn_weights[col] = (float(w), float(w), float(w))
            
    steps['Step 0: Fuzzy Weights'] = pd.DataFrame([tfn_weights], index=['Weights']).T

    # Helper to extract fuzzy number vectors for a specific column
    def get_fuzzy_vectors(series):
        is_trfn = len(series.iloc[0]) == 4 if len(series) > 0 and isinstance(series.iloc[0], tuple) else False
        if is_trfn:
            a = np.array([x[0] for x in series])
            b = np.array([x[1] for x in series])
            c = np.array([x[2] for x in series])
            d = np.array([x[3] for x in series])
            return a, b, c, d
        else:
            l = np.array([x[0] for x in series])
            m = np.array([x[1] for x in series])
            u = np.array([x[2] for x in series])
            return l, m, u
    # 1. Determine the optimal alternative TFN/TrFN x_0
    x_0 = {}
    for col in columns:
        vecs = get_fuzzy_vectors(df[col])
        if len(vecs) == 4:
            a, b, c, d = vecs
            if directions.get(col, 'maximize') == 'maximize':
                x_0[col] = (np.max(a), np.max(b), np.max(c), np.max(d))
            else:
                x_0[col] = (np.min(a), np.min(b), np.min(c), np.min(d))
        else:
            l, m, u = vecs
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
        vecs = get_fuzzy_vectors(df_combined[col])
        if len(vecs) == 4:
            a, b, c, d = vecs
            if directions.get(col, 'maximize') == 'maximize':
                sum_a, sum_b, sum_c, sum_d = np.sum(a), np.sum(b), np.sum(c), np.sum(d)
                norm_col = [(a[i]/sum_d if abs(sum_d) > 1e-9 else 0.0,
                             b[i]/sum_c if abs(sum_c) > 1e-9 else 0.0,
                             c[i]/sum_b if abs(sum_b) > 1e-9 else 0.0,
                             d[i]/sum_a if abs(sum_a) > 1e-9 else 0.0) for i in range(len(a))]
                norm_df[col] = norm_col
            else:
                inv_a = 1.0 / np.where(d == 0, 1e-9, d)
                inv_b = 1.0 / np.where(c == 0, 1e-9, c)
                inv_c = 1.0 / np.where(b == 0, 1e-9, b)
                inv_d = 1.0 / np.where(a == 0, 1e-9, a)
                sum_inv_a, sum_inv_b, sum_inv_c, sum_inv_d = np.sum(inv_a), np.sum(inv_b), np.sum(inv_c), np.sum(inv_d)
                norm_col = [(inv_a[i]/sum_inv_d if abs(sum_inv_d) > 1e-9 else 0.0,
                             inv_b[i]/sum_inv_c if abs(sum_inv_c) > 1e-9 else 0.0,
                             inv_c[i]/sum_inv_b if abs(sum_inv_b) > 1e-9 else 0.0,
                             inv_d[i]/sum_inv_a if abs(sum_inv_a) > 1e-9 else 0.0) for i in range(len(inv_a))]
                norm_df[col] = norm_col
        else:
            l, m, u = vecs
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
        is_trfn = False
        sum_vals = []
        for col in columns:
            r = norm_df.at[idx, col]
            w = tfn_weights[col]
            if len(r) == 4:
                is_trfn = True
                val = (r[0] * w[0], r[1] * w[1], r[2] * w[2], r[3] * w[3])
            else:
                val = (r[0] * w[0], r[1] * w[1], r[2] * w[2])
            weighted_df.at[idx, col] = val
            
            if not sum_vals:
                sum_vals = list(val)
            else:
                for j in range(len(val)):
                    sum_vals[j] += val[j]
        fuzzy_S[idx] = tuple(sum_vals)

    steps['Step 3: Weighted Normalized Fuzzy Decision Matrix'] = weighted_df.copy()
    steps['Step 4: Fuzzy Optimality Function ($S_i$)'] = fuzzy_S.to_frame(name='Fuzzy $S_i$')
        
    # 5. Defuzzification (Center of Area)
    # defuzzified = mean of values
    def defuzzify(tfn):
        return sum(tfn) / len(tfn)
        
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
    
    def format_fuzzy(x):
        if len(x) == 4:
            return f"({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}, {x[3]:.3f})"
        return f"({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})"
        
    results['S_i (Fuzzy)'] = fuzzy_S[df.index].apply(format_fuzzy)
    results['S_i (Crisp)'] = S_crisp[df.index]
    results['K_i (Utility Degree)'] = K
    results['Rank'] = rank
    
    # Sort by rank, then naturally by alternative name (index)
    results['sort_index'] = results.index.map(lambda x: tuple(natural_sort_key(x)))
    results = results.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    steps['Step 7: Final Result and Ranking'] = results.copy()
    
    if return_steps:
        return results, steps
    return results
