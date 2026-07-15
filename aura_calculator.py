import pandas as pd
import numpy as np

from mcdm.criteria import CriterionType, validate_method_capabilities
from mcdm.ranking import natural_sort_key, rank_scores
from mcdm.validation import validate_crisp_matrix, validate_weights


def prepare_aura_matrix(data: pd.DataFrame, directions: dict) -> pd.DataFrame:
    """Validate and normalize an AURA decision matrix once for repeated scoring."""
    df = validate_crisp_matrix(data)
    preferences = validate_method_capabilities("AURA", df.columns, directions)
    normalized_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val == min_val:
            normalized_df[col] = 1.0
            continue

        preference = preferences[col]
        if preference.kind is CriterionType.TARGET:
            reference = float(preference.target_value)
        elif preference.kind is CriterionType.BENEFIT:
            reference = max_val
        else:
            reference = min_val
        normalized_df[col] = 1.0 - (
            np.abs(df[col] - reference) / max(max_val - min_val, 1e-9)
        )
    return normalized_df


def calculate_aura_score_arrays(
    normalized_matrix: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float = 0.5,
    p: int = 2,
) -> dict[str, np.ndarray | float]:
    """Canonical NumPy scoring kernel shared by the UI and research simulations."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if p not in {1, 2}:
        raise ValueError("p must be 1 (Manhattan) or 2 (Euclidean).")
    matrix = np.asarray(normalized_matrix, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or weight_array.shape != (matrix.shape[1],):
        raise ValueError("Weight count must match the normalized matrix columns.")
    if not np.isfinite(matrix).all() or not np.isfinite(weight_array).all():
        raise ValueError("AURA inputs must be finite.")
    if (weight_array < 0).any() or weight_array.sum() <= 0:
        raise ValueError("AURA weights must be non-negative with a positive sum.")
    weight_array = weight_array / weight_array.sum()

    weighted = matrix * weight_array
    pis = weighted.max(axis=0)
    nis = weighted.min(axis=0)
    average = weighted.mean(axis=0)

    def distance(reference: np.ndarray) -> np.ndarray:
        deviations = np.abs(weighted - reference)
        if p == 1:
            return deviations.sum(axis=1)
        return np.power(np.power(deviations, p).sum(axis=1), 1 / p)

    d_plus_raw = distance(pis)
    d_minus_raw = distance(nis)
    d_average_raw = distance(average)

    def correct(values: np.ndarray) -> tuple[np.ndarray, float]:
        sigma = float(values.max() - values.min())
        return values + sigma * np.square(values), sigma

    d_plus, sigma_plus = correct(d_plus_raw)
    d_minus, sigma_minus = correct(d_minus_raw)
    d_average, sigma_average = correct(d_average_raw)
    utility = (
        alpha * (d_plus - d_minus) + (1.0 - alpha) * d_average
    ) / 2.0

    return {
        "weighted": weighted,
        "pis": pis,
        "nis": nis,
        "average": average,
        "d_plus_raw": d_plus_raw,
        "d_minus_raw": d_minus_raw,
        "d_average_raw": d_average_raw,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "d_average": d_average,
        "sigma_plus": sigma_plus,
        "sigma_minus": sigma_minus,
        "sigma_average": sigma_average,
        "utility": utility,
    }


def calculate_aura(data: pd.DataFrame, weights: dict, directions: dict, alpha: float, p: int = 2, return_steps: bool = False):
    """
    Computes the Adaptive Utility Ranking Algorithm (AURA).
    
    Args:
        data (pd.DataFrame): The decision matrix (alternatives as index/rows, criteria as columns).
        weights (dict): A dictionary of weights for each criterion.
        directions (dict): A dictionary specifying 'maximize' or 'minimize' for each criterion.
        alpha (float): The balance parameter (0 to 1).
        p (int): Distance metric parameter (1 for Manhattan, 2 for Euclidean).
        return_steps (bool): Whether to return a dictionary of intermediate calculation steps.
        
    Returns:
        pd.DataFrame or tuple: A dataframe containing the rankings, scores, and distance metrics. 
                               If return_steps=True, returns (results_df, steps_dict).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    if p not in {1, 2}:
        raise ValueError("p must be 1 (Manhattan) or 2 (Euclidean).")
        
    df = validate_crisp_matrix(data)
    normalized_weights = validate_weights(weights, df.columns, normalize=True)
    steps = {}

    # 1. Normalization
    normalized_df = prepare_aura_matrix(df, directions)

    steps['Step 1: Normalized Decision Matrix'] = normalized_df.copy()

    # 2. Weighted Matrix
    kernel = calculate_aura_score_arrays(
        normalized_df.to_numpy(dtype=float),
        np.array([normalized_weights[col] for col in df.columns], dtype=float),
        alpha=alpha,
        p=p,
    )
    weighted_df = pd.DataFrame(kernel["weighted"], index=df.index, columns=df.columns)

    steps['Step 2: Weighted Normalized Matrix'] = weighted_df.copy()

    # 3. Determine Ideal Solutions (PIS, NIS, AS)
    pis = pd.Series(kernel["pis"], index=df.columns)
    nis = pd.Series(kernel["nis"], index=df.columns)
    as_sol = pd.Series(kernel["average"], index=df.columns)
    
    steps['Step 3: Ideal Solutions'] = {
        'PIS (Positive Ideal Solution)': pis,
        'NIS (Negative Ideal Solution)': nis,
        'AS (Average Solution)': as_sol
    }

    # Calculate raw distances (using Minkowski distance)
    # Adding epsilon to zero values to avoid perfectly zero distance which breaks things sometimes
    d_pos_raw = pd.Series(kernel["d_plus_raw"], index=df.index)
    d_neg_raw = pd.Series(kernel["d_minus_raw"], index=df.index)
    d_avg_raw = pd.Series(kernel["d_average_raw"], index=df.index)

    steps['Step 4a: Raw Distances'] = pd.DataFrame({
        'd+ (Raw dist to PIS)': d_pos_raw,
        'd- (Raw dist to NIS)': d_neg_raw,
        'd_avg (Raw dist to AS)': d_avg_raw
    })

    d_plus_corrected = pd.Series(kernel["d_plus"], index=df.index)
    d_minus_corrected = pd.Series(kernel["d_minus"], index=df.index)
    d_avg_corrected = pd.Series(kernel["d_average"], index=df.index)
    sigma_plus = kernel["sigma_plus"]
    sigma_minus = kernel["sigma_minus"]
    sigma_avg = kernel["sigma_average"]

    distances = pd.DataFrame(index=df.index)
    distances['D_plus'] = d_plus_corrected
    distances['D_minus'] = d_minus_corrected
    distances['D_avg'] = d_avg_corrected

    steps['Step 4b: Corrected Distances'] = distances.copy()
    steps['Step 4b: Correction Factors'] = {
        'Sigma+': sigma_plus,
        'Sigma-': sigma_minus,
        'Sigma_avg': sigma_avg
    }

    # 5. Final Utility Score (Exact AURA Formula)
    distances['Utility_Score'] = pd.Series(kernel["utility"], index=df.index)

    steps['Step 5: Final Utility Score'] = distances[['Utility_Score']].copy()

    # 6. Ranking (Lowest score gets highest rank)
    distances['Rank'] = rank_scores(distances['Utility_Score'], ascending=True)
    
    # Merge results back
    results = df.copy()
    results['D+ (PIS)'] = distances['D_plus']
    results['D- (NIS)'] = distances['D_minus']
    results['D_avg (AS)'] = distances['D_avg']
    results['Utility Score'] = distances['Utility_Score']
    results['Rank'] = distances['Rank']
    
    # Sort by rank, then naturally by alternative name (index)
    results['sort_index'] = results.index.map(lambda x: tuple(natural_sort_key(x)))
    results = results.sort_values(by=['Rank', 'sort_index']).drop(columns=['sort_index'])
    steps['Step 6: Final Result and Ranking'] = results.copy()
    
    if return_steps:
        return results, steps
    return results
