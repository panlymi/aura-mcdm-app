import pandas as pd
import numpy as np

from mcdm.criteria import CriterionType, validate_method_capabilities
from mcdm.ranking import natural_sort_key, rank_scores
from mcdm.validation import validate_crisp_matrix, validate_method_matrix, validate_weights

def calculate_saw(
    data: pd.DataFrame,
    weights: dict,
    directions: dict,
    normalization: str = "ratio_to_max",
    return_steps: bool = False,
):
    """
    Computes the Simple Additive Weighting (SAW) MCDM method.
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        normalization (str): Normalization scheme to use:
            - 'ratio_to_max' (default): Max for benefit (x/max), Min/x for cost.
            - 'min_max': Linear range (x - min)/(max - min) for benefit, (max - x)/(max - min) for cost.
            - 'sum': Proportional shares (x / sum(x)) for benefit, (1/x)/sum(1/x) for cost.
            - 'vector': Euclidean normalization (x / sqrt(sum(x^2))) for benefit, 1 - (x/norm) for cost.
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings and scores, or a tuple containing that and a dictionary of calculation steps.
    """
    df = validate_crisp_matrix(data)
    columns = df.columns
    preferences = validate_method_capabilities("SAW", columns, directions)
    validate_method_matrix("SAW", df, directions)
    normalized_weights = validate_weights(weights, columns, normalize=True)

    norm_key = str(normalization).strip().lower()
    if norm_key in ("ratio_to_max", "max", "canonical"):
        norm_method = "ratio_to_max"
        norm_title = "Ratio-to-Max (Canonical)"
    elif norm_key in ("min_max", "minmax", "range", "0_1", "0-1"):
        norm_method = "min_max"
        norm_title = "Min–Max (Range / 0–1)"
    elif norm_key in ("sum", "proportion", "linear_sum", "sum_ratio"):
        norm_method = "sum"
        norm_title = "Sum / Linear Proportion"
    elif norm_key in ("vector", "euclidean", "norm"):
        norm_method = "vector"
        norm_title = "Vector (Euclidean)"
    else:
        raise ValueError(
            f"Unknown SAW normalization method '{normalization}'. "
            "Supported options: 'ratio_to_max', 'min_max', 'sum', 'vector'."
        )
    
    steps_dict = {}
    if return_steps:
        steps_dict['Step 1: Original Decision Matrix'] = df.copy()

    # 1. Normalization
    normalized_df = pd.DataFrame(index=df.index, columns=columns, dtype=float)
    for col in columns:
        col_max = float(df[col].max())
        col_min = float(df[col].min())
        is_benefit = preferences[col].kind is CriterionType.BENEFIT

        if norm_method == "ratio_to_max":
            if is_benefit:
                normalized_df[col] = df[col] / col_max if abs(col_max) > 1e-9 else 0.0
            else:
                for idx in df.index:
                    val = float(df.loc[idx, col])
                    normalized_df.loc[idx, col] = col_min / val if abs(val) > 1e-9 else 0.0

        elif norm_method == "min_max":
            range_span = col_max - col_min
            if abs(range_span) > 1e-9:
                if is_benefit:
                    normalized_df[col] = (df[col] - col_min) / range_span
                else:
                    normalized_df[col] = (col_max - df[col]) / range_span
            else:
                normalized_df[col] = 1.0

        elif norm_method == "sum":
            if is_benefit:
                col_sum = float(df[col].sum())
                normalized_df[col] = df[col] / col_sum if abs(col_sum) > 1e-9 else 0.0
            else:
                reciprocals = pd.Series(
                    [1.0 / float(v) if abs(float(v)) > 1e-9 else 0.0 for v in df[col]],
                    index=df.index,
                )
                recip_sum = float(reciprocals.sum())
                normalized_df[col] = reciprocals / recip_sum if abs(recip_sum) > 1e-9 else 0.0

        elif norm_method == "vector":
            col_norm = float(np.sqrt((df[col] ** 2).sum()))
            if abs(col_norm) > 1e-9:
                if is_benefit:
                    normalized_df[col] = df[col] / col_norm
                else:
                    normalized_df[col] = 1.0 - (df[col] / col_norm)
            else:
                normalized_df[col] = 0.0
            
    if return_steps:
        steps_dict[f'Step 2: Normalized Decision Matrix ({norm_title})'] = normalized_df.copy()

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
