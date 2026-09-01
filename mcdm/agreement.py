"""Agreement and rank-consistency metrics for multi-method MCDM comparison."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def calculate_spearman_rho(rank_a: pd.Series | Sequence[float], rank_b: pd.Series | Sequence[float]) -> float:
    """Calculate Spearman's rank correlation coefficient (rho) using pure NumPy.
    
    Equivalent to Pearson correlation computed on ranking scores, fully tie-safe.
    """
    x = np.asarray(rank_a, dtype=float)
    y = np.asarray(rank_b, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return 1.0 if len(x) == len(y) and np.allclose(x, y) else 0.0

    x_dev = x - np.mean(x)
    y_dev = y - np.mean(y)
    denom = np.sqrt(np.sum(x_dev**2) * np.sum(y_dev**2))
    if denom <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.clip(np.sum(x_dev * y_dev) / denom, -1.0, 1.0))


def calculate_kendall_tau(rank_a: pd.Series | Sequence[float], rank_b: pd.Series | Sequence[float]) -> float:
    """Calculate Kendall's tau-b rank correlation coefficient using pure NumPy (tie-adjusted)."""
    x = np.asarray(rank_a, dtype=float)
    y = np.asarray(rank_b, dtype=float)
    n = len(x)
    if n != len(y) or n < 2:
        return 1.0 if n == len(y) and np.allclose(x, y) else 0.0

    # Vectorized pairwise differences on upper triangle
    i_upper, j_upper = np.triu_indices(n, k=1)
    sign_x = np.sign(x[i_upper] - x[j_upper])
    sign_y = np.sign(y[i_upper] - y[j_upper])

    concordant = np.sum(sign_x * sign_y > 0)
    discordant = np.sum(sign_x * sign_y < 0)
    n0 = len(i_upper)
    ties_x = np.sum(sign_x == 0)
    ties_y = np.sum(sign_y == 0)

    denom = np.sqrt(float((n0 - ties_x) * (n0 - ties_y)))
    if denom <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.clip((concordant - discordant) / denom, -1.0, 1.0))


def calculate_mard(rank_a: pd.Series | Sequence[float], rank_b: pd.Series | Sequence[float]) -> float:
    """Calculate Mean Absolute Rank Difference (MARD) between two rankings."""
    arr_a = np.asarray(rank_a, dtype=float)
    arr_b = np.asarray(rank_b, dtype=float)
    if len(arr_a) == 0:
        return 0.0
    return float(np.mean(np.abs(arr_a - arr_b)))


def calculate_top_k_jaccard(
    rank_a: pd.Series | Sequence[float],
    rank_b: pd.Series | Sequence[float],
    k: int,
) -> float:
    """Calculate Jaccard similarity coefficient between top-k subsets of two rankings."""
    if k <= 0 or len(rank_a) == 0:
        return 1.0

    # Handle pandas Series with alternative index or array-like
    if isinstance(rank_a, pd.Series) and isinstance(rank_b, pd.Series):
        top_a = set(rank_a.nsmallest(k).index)
        top_b = set(rank_b.nsmallest(k).index)
    else:
        s_a = pd.Series(rank_a)
        s_b = pd.Series(rank_b)
        top_a = set(s_a.nsmallest(k).index)
        top_b = set(s_b.nsmallest(k).index)

    union = top_a.union(top_b)
    if not union:
        return 1.0
    intersection = top_a.intersection(top_b)
    return float(len(intersection) / len(union))


def get_default_jaccard_cutoffs(num_alternatives: int) -> tuple[int, int]:
    """Return appropriate (k1, k2) cutoffs for top-k Jaccard similarity based on problem size."""
    if num_alternatives >= 20:
        return 10, 20
    if num_alternatives >= 10:
        return 5, 10
    if num_alternatives >= 5:
        return 3, 5
    return 1, max(1, min(2, num_alternatives))


def calculate_agreement_table(
    rankings_df: pd.DataFrame,
    benchmark_method: str | None = None,
    k1: int | None = None,
    k2: int | None = None,
) -> pd.DataFrame:
    """Generate the Agreement Table of all MCDM methods compared with a benchmark method.
    
    Columns returned:
      Method | Spearman ρ | Kendall τ-b | MARD | Top-k1 Jaccard | Top-k2 Jaccard
    """
    if rankings_df.empty:
        return pd.DataFrame()

    methods = list(rankings_df.columns)
    if benchmark_method is None or benchmark_method not in methods:
        benchmark_method = methods[0]

    num_alts = len(rankings_df)
    default_k1, default_k2 = get_default_jaccard_cutoffs(num_alts)
    k1 = k1 if k1 is not None and k1 > 0 else default_k1
    k2 = k2 if k2 is not None and k2 > 0 else default_k2

    benchmark_ranks = rankings_df[benchmark_method]
    records = []

    for method in methods:
        method_ranks = rankings_df[method]
        rho = calculate_spearman_rho(method_ranks, benchmark_ranks)
        tau = calculate_kendall_tau(method_ranks, benchmark_ranks)
        mard = calculate_mard(method_ranks, benchmark_ranks)
        jaccard_1 = calculate_top_k_jaccard(method_ranks, benchmark_ranks, k1)
        jaccard_2 = calculate_top_k_jaccard(method_ranks, benchmark_ranks, k2)

        records.append({
            "Method": method,
            "Spearman ρ": rho,
            "Kendall τ-b": tau,
            "MARD": mard,
            f"Top-{k1} Jaccard": jaccard_1,
            f"Top-{k2} Jaccard": jaccard_2,
        })

    result_df = pd.DataFrame(records)
    result_df.set_index("Method", inplace=True)
    return result_df


def calculate_pairwise_spearman_matrix(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate full pairwise Spearman rank correlation matrix using pure NumPy."""
    if rankings_df.empty:
        return pd.DataFrame()
    methods = list(rankings_df.columns)
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
            else:
                matrix.loc[m1, m2] = calculate_spearman_rho(rankings_df[m1], rankings_df[m2])
    return matrix


def calculate_pairwise_kendall_matrix(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate full pairwise Kendall tau-b correlation matrix using pure NumPy."""
    if rankings_df.empty:
        return pd.DataFrame()
    methods = list(rankings_df.columns)
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
            else:
                matrix.loc[m1, m2] = calculate_kendall_tau(rankings_df[m1], rankings_df[m2])
    return matrix


__all__ = [
    "calculate_spearman_rho",
    "calculate_kendall_tau",
    "calculate_mard",
    "calculate_top_k_jaccard",
    "get_default_jaccard_cutoffs",
    "calculate_agreement_table",
    "calculate_pairwise_spearman_matrix",
    "calculate_pairwise_kendall_matrix",
]
