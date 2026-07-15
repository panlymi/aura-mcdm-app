from __future__ import annotations

import pandas as pd
import pytest

from entropy_calculator import calculate_entropy_weights
from fuzzy_aras_calculator import calculate_fuzzy_aras
from mcdm.validation import MCDMValidationError
from merec_calculator import calculate_merec_weights


@pytest.fixture
def crisp_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Quality": [4.0, 7.0, 9.0],
            "Cost": [8.0, 5.0, 6.0],
        },
        index=["A1", "A2", "A3"],
    )


def test_fuzzy_aras_direction_aliases_match_canonical_results():
    matrix = pd.DataFrame(
        {
            "Quality": [(2.0, 3.0, 4.0), (4.0, 5.0, 6.0), (3.0, 4.0, 5.0)],
            "Cost": [(6.0, 7.0, 8.0), (4.0, 5.0, 6.0), (5.0, 6.0, 7.0)],
        },
        index=["A1", "A2", "A3"],
    )
    weights = {"Quality": 0.6, "Cost": 0.4}
    canonical = calculate_fuzzy_aras(
        matrix, weights, {"Quality": "maximize", "Cost": "minimize"}
    )

    for directions in (
        {"Quality": "benefit", "Cost": "cost"},
        {"Quality": "max", "Cost": "min"},
        {"Quality": "maximise", "Cost": "minimise"},
    ):
        aliased = calculate_fuzzy_aras(matrix, weights, directions)
        pd.testing.assert_series_equal(
            aliased["K_i (Utility Degree)"], canonical["K_i (Utility Degree)"]
        )
        pd.testing.assert_series_equal(aliased["Rank"], canonical["Rank"])


@pytest.mark.parametrize(
    "values",
    [
        [(-1.0, 0.0, 1.0), (1.0, 2.0, 3.0)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    ],
)
def test_fuzzy_aras_rejects_invalid_benefit_ratio_domain(values):
    matrix = pd.DataFrame({"Quality": values}, index=["A1", "A2"])

    with pytest.raises(MCDMValidationError, match="non-negative.*at least one positive"):
        calculate_fuzzy_aras(matrix, {"Quality": 1.0}, {"Quality": "benefit"})


def test_entropy_direction_aliases_match_canonical_results(crisp_matrix):
    canonical_weights, canonical_steps = calculate_entropy_weights(
        crisp_matrix,
        {"Quality": "maximize", "Cost": "minimize"},
        method="standard",
    )

    for directions in (
        {"Quality": "benefit", "Cost": "cost"},
        {"Quality": "max", "Cost": "min"},
        {"Quality": "maximise", "Cost": "minimise"},
    ):
        weights, steps = calculate_entropy_weights(crisp_matrix, directions, method="standard")
        assert weights == pytest.approx(canonical_weights)
        pd.testing.assert_frame_equal(
            steps["Step 2: Normalized Data"], canonical_steps["Step 2: Normalized Data"]
        )


@pytest.mark.parametrize("method", ["simple", "standard", "shifted"])
def test_entropy_rejects_target_criteria_and_requires_manual_weights(crisp_matrix, method):
    with pytest.raises(MCDMValidationError, match="target criteria.*manual weights"):
        calculate_entropy_weights(
            crisp_matrix,
            {"Quality": {"type": "target", "value": 7.0}, "Cost": "cost"},
            method=method,
        )


def test_merec_direction_aliases_match_canonical_results(crisp_matrix):
    canonical_weights, canonical_steps = calculate_merec_weights(
        crisp_matrix, {"Quality": "maximize", "Cost": "minimize"}
    )

    for directions in (
        {"Quality": "benefit", "Cost": "cost"},
        {"Quality": "max", "Cost": "min"},
        {"Quality": "maximise", "Cost": "minimise"},
    ):
        weights, steps = calculate_merec_weights(crisp_matrix, directions)
        assert weights == pytest.approx(canonical_weights)
        pd.testing.assert_frame_equal(
            steps["Step 2: Normalized Decision Matrix (N)"],
            canonical_steps["Step 2: Normalized Decision Matrix (N)"],
        )


def test_merec_rejects_target_criteria_and_requires_manual_weights(crisp_matrix):
    with pytest.raises(MCDMValidationError, match="target criteria.*manual weights"):
        calculate_merec_weights(
            crisp_matrix,
            {"Quality": {"type": "target", "value": 7.0}, "Cost": "cost"},
        )
