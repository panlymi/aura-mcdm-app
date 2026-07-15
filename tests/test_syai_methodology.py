from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mcdm.validation import MCDMValidationError
from syai_calculator import calculate_syai


@pytest.fixture
def published_syai_example():
    """Numerical example from Rahman et al. (2025), Tables 1-7.

    Source: https://www.ejpam.com/index.php/ejpam/article/view/6560/2443
    """
    matrix = pd.DataFrame(
        {
            "Cost": [200.0, 250.0, 300.0],
            "Quality": [8.0, 7.0, 9.0],
            "Delivery Time": [4.0, 5.0, 6.0],
            "Temperature": [30.0, 60.0, 85.0],
        },
        index=["A1", "A2", "A3"],
    )
    weights = {criterion: 0.25 for criterion in matrix.columns}
    directions = {
        "Cost": "minimize",
        "Quality": "maximize",
        "Delivery Time": "minimize",
        "Temperature": {"type": "target", "value": 60.0},
    }
    return matrix, weights, directions


def test_published_worked_example_reproduces_intermediate_and_final_tables(
    published_syai_example,
):
    matrix, weights, directions = published_syai_example

    result, steps = calculate_syai(
        matrix,
        weights,
        directions,
        beta=0.5,
        return_steps=True,
    )

    expected_normalized = np.array(
        [
            [1.0, 0.505, 1.0, 0.46],
            [0.505, 0.01, 0.505, 1.0],
            [0.01, 1.0, 0.01, 0.55],
        ]
    )
    np.testing.assert_allclose(
        steps["Step 1: Normalized Decision Matrix"].to_numpy(dtype=float),
        expected_normalized,
        atol=1e-12,
    )

    expected_distances = np.array(
        [
            [0.25875, 0.61875],
            [0.49500, 0.38250],
            [0.60750, 0.27000],
        ]
    )
    np.testing.assert_allclose(
        result.loc[["A1", "A2", "A3"], ["D+ (Dist to Ideal)", "D- (Dist to Anti-Ideal)"]],
        expected_distances,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.loc[["A1", "A2", "A3"], "Closeness Score (D_i)"],
        [0.705128, 0.435897, 0.307692],
        atol=5e-7,
    )
    assert result.loc[["A1", "A2", "A3"], "Rank"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_beta_endpoints_are_rejected(beta, published_syai_example):
    matrix, weights, directions = published_syai_example

    with pytest.raises(MCDMValidationError, match=r"0 < beta < 1"):
        calculate_syai(matrix, weights, directions, beta=beta)


@pytest.mark.parametrize("beta", [0.05, 0.5, 0.95])
def test_supported_beta_values_remain_valid(beta, published_syai_example):
    matrix, weights, directions = published_syai_example

    result = calculate_syai(matrix, weights, directions, beta=beta)

    assert np.isfinite(result["Closeness Score (D_i)"]).all()
    assert result["Closeness Score (D_i)"].between(0.0, 1.0).all()
