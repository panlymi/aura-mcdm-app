from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aras_calculator import calculate_aras
from arie_calculator import calculate_arie
from mcdm.validation import MCDMValidationError
from saw_calculator import calculate_saw


RATIO_METHODS = [calculate_aras, calculate_arie, calculate_saw]


@pytest.mark.parametrize("calculator", RATIO_METHODS)
def test_ratio_normalization_rejects_negative_benefit_values(calculator):
    matrix = pd.DataFrame(
        {"Benefit": [-1.0, 2.0, 3.0]},
        index=["A1", "A2", "A3"],
    )

    with pytest.raises(
        MCDMValidationError,
        match="non-negative values with at least one positive value",
    ):
        calculator(matrix, {"Benefit": 1.0}, {"Benefit": "maximize"})


@pytest.mark.parametrize("calculator", RATIO_METHODS)
def test_ratio_normalization_rejects_all_zero_benefit_column(calculator):
    matrix = pd.DataFrame(
        {"Benefit": [0.0, 0.0, 0.0]},
        index=["A1", "A2", "A3"],
    )

    with pytest.raises(
        MCDMValidationError,
        match="non-negative values with at least one positive value",
    ):
        calculator(matrix, {"Benefit": 1.0}, {"Benefit": "benefit"})


@pytest.mark.parametrize("calculator", RATIO_METHODS)
def test_ratio_normalization_accepts_zero_when_column_has_positive_value(calculator):
    matrix = pd.DataFrame(
        {
            "Benefit": [0.0, 5.0, 10.0],
            "Cost": [3.0, 2.0, 1.0],
        },
        index=["A1", "A2", "A3"],
    )

    result = calculator(
        matrix,
        {"Benefit": 0.5, "Cost": 0.5},
        {"Benefit": "benefit", "Cost": "cost"},
    )

    assert result.loc["A3", "Rank"] == 1
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()
