from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mcdm.validation import MCDMValidationError
from waspas_calculator import calculate_waspas


MATRIX = pd.DataFrame(
    {
        "Benefit": [9.0, 7.0, 3.0],
        "Cost": [2.0, 4.0, 8.0],
        "Benefit2": [6.0, 8.0, 5.0],
    },
    index=["A1", "A2", "A3"],
)
WEIGHTS = {"Benefit": 0.4, "Cost": 0.35, "Benefit2": 0.25}
DIRECTIONS = {"Benefit": "maximize", "Cost": "minimize", "Benefit2": "maximize"}


def test_waspas_matches_hand_calculated_wsm_wpm_and_aggregate_scores():
    result, steps = calculate_waspas(
        MATRIX, WEIGHTS, DIRECTIONS, lambda_value=0.5, return_steps=True
    )

    expected_wsm = np.array([0.9375, 0.7361111111111112, 0.3770833333333333])
    expected_wpm = np.array([0.9306048591020996, 0.7095478913576515, 0.352695976622208])
    expected = 0.5 * expected_wsm + 0.5 * expected_wpm

    ordered = result.reindex(MATRIX.index)
    np.testing.assert_allclose(ordered["Q_i (WSM)"], expected_wsm)
    np.testing.assert_allclose(ordered["Q_i (WPM)"], expected_wpm)
    np.testing.assert_allclose(ordered["Q_i (WASPAS Score)"], expected)
    assert ordered["Rank"].tolist() == [1, 2, 3]
    assert list(steps) == [
        "Step 1: Original Decision Matrix",
        "Step 2: Normalized Decision Matrix",
        "Step 3: Weighted Sum Components",
        "Step 4: Weighted Product Components",
        "Step 5: Aggregated Scores",
        "Step 6: Final Result and Ranking",
    ]


def test_lambda_endpoints_equal_the_component_models():
    wpm = calculate_waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_value=0.0)
    wsm = calculate_waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_value=1.0)

    np.testing.assert_allclose(wpm["Q_i (WASPAS Score)"], wpm["Q_i (WPM)"])
    np.testing.assert_allclose(wsm["Q_i (WASPAS Score)"], wsm["Q_i (WSM)"])


@pytest.mark.parametrize("lambda_value", [-0.01, 1.01, np.nan, "not-a-number"])
def test_invalid_lambda_is_rejected(lambda_value):
    with pytest.raises(MCDMValidationError, match="lambda"):
        calculate_waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_value=lambda_value)


def test_zero_weight_is_neutral_in_the_product_component():
    result = calculate_waspas(
        MATRIX,
        {"Benefit": 1.0, "Cost": 0.0, "Benefit2": 0.0},
        DIRECTIONS,
    ).reindex(MATRIX.index)

    expected = MATRIX["Benefit"] / MATRIX["Benefit"].max()
    np.testing.assert_allclose(result["Q_i (WPM)"], expected)
