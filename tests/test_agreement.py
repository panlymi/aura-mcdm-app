from __future__ import annotations

import numpy as np
import pandas as pd

from mcdm.agreement import (
    calculate_agreement_table,
    calculate_kendall_tau,
    calculate_mard,
    calculate_pairwise_kendall_matrix,
    calculate_pairwise_spearman_matrix,
    calculate_spearman_rho,
    calculate_top_k_jaccard,
    get_default_jaccard_cutoffs,
)


def test_spearman_and_kendall_identical_rankings():
    r1 = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
    r2 = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])

    assert calculate_spearman_rho(r1, r2) == 1.0
    assert calculate_kendall_tau(r1, r2) == 1.0
    assert calculate_mard(r1, r2) == 0.0
    assert calculate_top_k_jaccard(r1, r2, 3) == 1.0


def test_spearman_and_kendall_reversed_rankings():
    r1 = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
    r2 = pd.Series([5, 4, 3, 2, 1], index=["A", "B", "C", "D", "E"])

    assert np.isclose(calculate_spearman_rho(r1, r2), -1.0)
    assert np.isclose(calculate_kendall_tau(r1, r2), -1.0)
    # MARD for [1,2,3,4,5] vs [5,4,3,2,1]: |1-5| + |2-4| + |3-3| + |4-2| + |5-1| = 4+2+0+2+4 = 12 / 5 = 2.4
    assert np.isclose(calculate_mard(r1, r2), 2.4)
    # Top-2 of r1: {A, B}. Top-2 of r2: {D, E}. Intersection = 0, Union = 4 -> Jaccard = 0.0
    assert calculate_top_k_jaccard(r1, r2, 2) == 0.0


def test_top_k_jaccard_partial_overlap():
    r1 = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
    r2 = pd.Series([2, 1, 4, 3, 5], index=["A", "B", "C", "D", "E"])

    # Top-3 of r1: {A, B, C}. Top-3 of r2: {A, B, D}.
    # Intersection: {A, B} (2 items). Union: {A, B, C, D} (4 items). Jaccard = 2/4 = 0.5
    assert calculate_top_k_jaccard(r1, r2, 3) == 0.5


def test_get_default_jaccard_cutoffs():
    assert get_default_jaccard_cutoffs(25) == (10, 20)
    assert get_default_jaccard_cutoffs(15) == (5, 10)
    assert get_default_jaccard_cutoffs(8) == (3, 5)
    assert get_default_jaccard_cutoffs(3) == (1, 2)


def test_calculate_agreement_table():
    df = pd.DataFrame(
        {
            "AURA": [1, 2, 3, 4, 5, 6],
            "SAW": [1, 2, 4, 3, 5, 6],
            "TOPSIS": [2, 1, 3, 4, 6, 5],
        },
        index=["Alt1", "Alt2", "Alt3", "Alt4", "Alt5", "Alt6"],
    )

    table = calculate_agreement_table(df, benchmark_method="AURA", k1=3, k2=5)
    assert list(table.index) == ["AURA", "SAW", "TOPSIS"]
    assert "Spearman ρ" in table.columns
    assert "Kendall τ-b" in table.columns
    assert "MARD" in table.columns
    assert "Top-3 Jaccard" in table.columns
    assert "Top-5 Jaccard" in table.columns

    # Benchmark row with itself
    assert table.loc["AURA", "Spearman ρ"] == 1.0
    assert table.loc["AURA", "Kendall τ-b"] == 1.0
    assert table.loc["AURA", "MARD"] == 0.0
    assert table.loc["AURA", "Top-3 Jaccard"] == 1.0
    assert table.loc["AURA", "Top-5 Jaccard"] == 1.0

    # SAW row
    assert table.loc["SAW", "MARD"] == (0 + 0 + 1 + 1 + 0 + 0) / 6.0


def test_pairwise_matrices():
    df = pd.DataFrame(
        {
            "AURA": [1, 2, 3],
            "SAW": [1, 2, 3],
            "TOPSIS": [3, 2, 1],
        },
        index=["A", "B", "C"],
    )

    spearman_matrix = calculate_pairwise_spearman_matrix(df)
    kendall_matrix = calculate_pairwise_kendall_matrix(df)

    assert spearman_matrix.shape == (3, 3)
    assert kendall_matrix.shape == (3, 3)
    assert spearman_matrix.loc["AURA", "SAW"] == 1.0
    assert np.isclose(spearman_matrix.loc["AURA", "TOPSIS"], -1.0)
