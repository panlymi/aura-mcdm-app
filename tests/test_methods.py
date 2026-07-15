from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mcdm.analysis import calculate_method, compare_methods
from mcdm.criteria import UnsupportedCriterionError
from mcdm.presentation import RESULT_PRESENTATION
from mcdm.validation import MCDMValidationError


CRISP_METHODS = ["AURA", "ARAS", "SYAI", "ARIE", "MOORA", "TOPSIS", "SAW", "VIKOR"]


@pytest.fixture
def dominance_problem():
    matrix = pd.DataFrame(
        {"Benefit": [10.0, 5.0, 1.0], "Cost": [1.0, 5.0, 10.0]},
        index=["Best", "Middle", "Worst"],
    )
    return matrix, {"Benefit": 0.5, "Cost": 0.5}, {"Benefit": "maximize", "Cost": "minimize"}


@pytest.mark.parametrize("method", CRISP_METHODS)
def test_dominant_alternative_is_ranked_first(method, dominance_problem):
    matrix, weights, directions = dominance_problem
    result = calculate_method(method, matrix, weights, directions)
    assert result.loc["Best", "Rank"] == 1


@pytest.mark.parametrize("method", ["AURA", "ARIE", "SYAI"])
def test_native_target_methods_select_exact_target(method):
    matrix = pd.DataFrame({"Target": [0.0, 5.0, 10.0]}, index=["Low", "Exact", "High"])
    result = calculate_method(
        method,
        matrix,
        {"Target": 1.0},
        {"Target": {"type": "target", "value": 5.0}},
    )
    assert result.loc["Exact", "Rank"] == 1


@pytest.mark.parametrize("method", ["ARAS", "MOORA", "TOPSIS", "SAW", "VIKOR"])
def test_non_target_methods_reject_target_criteria(method):
    matrix = pd.DataFrame({"Target": [0.0, 5.0, 10.0]}, index=["Low", "Exact", "High"])
    with pytest.raises(UnsupportedCriterionError):
        calculate_method(
            method,
            matrix,
            {"Target": 1.0},
            {"Target": {"type": "target", "value": 5.0}},
        )


def test_comparison_excludes_incompatible_methods():
    matrix = pd.DataFrame({"Target": [0.0, 5.0, 10.0]}, index=["Low", "Exact", "High"])
    comparison, excluded = compare_methods(
        ["AURA", "ARIE", "MOORA", "TOPSIS"],
        matrix,
        {"Target": 1.0},
        {"Target": {"type": "target", "value": 5.0}},
    )
    assert list(comparison.columns) == ["AURA", "ARIE"]
    assert set(excluded) == {"MOORA", "TOPSIS"}


def test_aura_is_invariant_to_weight_scale(dominance_problem):
    matrix, weights, directions = dominance_problem
    scaled = {criterion: value * 100 for criterion, value in weights.items()}
    first = calculate_method("AURA", matrix, weights, directions).sort_index()
    second = calculate_method("AURA", matrix, scaled, directions).sort_index()
    pd.testing.assert_series_equal(first["Rank"], second["Rank"])
    np.testing.assert_allclose(first["Utility Score"], second["Utility Score"])


@pytest.mark.parametrize("method", CRISP_METHODS)
def test_ties_are_preserved(method):
    matrix = pd.DataFrame(
        {"Benefit": [4.0, 4.0, 4.0], "Cost": [2.0, 2.0, 2.0]},
        index=["A1", "A2", "A3"],
    )
    result = calculate_method(
        method,
        matrix,
        {"Benefit": 0.5, "Cost": 0.5},
        {"Benefit": "maximize", "Cost": "minimize"},
    )
    assert result["Rank"].tolist() == [1, 1, 1]


@pytest.mark.parametrize("method", ["ARAS", "ARIE", "SAW"])
def test_reciprocal_cost_methods_reject_nonpositive_values(method):
    matrix = pd.DataFrame({"Cost": [0.0, 2.0, 4.0]}, index=["A1", "A2", "A3"])
    with pytest.raises(MCDMValidationError, match="strictly positive"):
        calculate_method(method, matrix, {"Cost": 1.0}, {"Cost": "minimize"})


GOLDEN = {
    "AURA": ([1, 2, 3], [-0.067051326651, -0.047445403786, 0.266096130765]),
    "ARAS": ([1, 2, 3], [0.943869368713, 0.720513194343, 0.366671526462]),
    "SYAI": ([1, 2, 3], [0.833333333333, 0.75, 0.0]),
    "ARIE": ([1, 2, 3], [0.659999999936, 0.594510616219, 0.287128713793]),
    "MOORA": ([1, 2, 3], [0.363135831752, 0.263625815345, -0.091918975894]),
    "TOPSIS": ([1, 2, 3], [0.872961656104, 0.677941433046, 0.0]),
    "SAW": ([1, 2, 3], [0.9375, 0.736111111111, 0.377083333333]),
    "VIKOR": ([2, 1, 3], [0.0625, 0.05, 1.0]),
}


@pytest.mark.parametrize("method", CRISP_METHODS)
def test_golden_benchmark(method):
    matrix = pd.DataFrame(
        {
            "Benefit": [9.0, 7.0, 3.0],
            "Cost": [2.0, 4.0, 8.0],
            "Benefit2": [6.0, 8.0, 5.0],
        },
        index=["A1", "A2", "A3"],
    )
    weights = {"Benefit": 0.4, "Cost": 0.35, "Benefit2": 0.25}
    directions = {"Benefit": "maximize", "Cost": "minimize", "Benefit2": "maximize"}
    result = calculate_method(method, matrix, weights, directions).sort_index()
    expected_ranks, expected_scores = GOLDEN[method]
    metadata = RESULT_PRESENTATION[method]
    assert result["Rank"].tolist() == expected_ranks
    np.testing.assert_allclose(
        result[metadata.score_column].to_numpy(), expected_scores, rtol=1e-9, atol=1e-9
    )
