from __future__ import annotations

import pandas as pd
import pytest

from entropy_calculator import calculate_entropy_weights
from mcdm.validation import MCDMValidationError, validate_crisp_matrix, validate_weights
from merec_calculator import calculate_merec_weights


def test_numeric_validation_reports_bad_cell():
    matrix = pd.DataFrame({"Cost": ["1", "not-a-number"]}, index=["A1", "A2"])
    with pytest.raises(MCDMValidationError, match="A2"):
        validate_crisp_matrix(matrix)


def test_valid_thousands_separator_is_supported():
    matrix = pd.DataFrame({"Income": ["1,000", "2,500.50"]}, index=["A1", "A2"])
    validated = validate_crisp_matrix(matrix)
    assert validated["Income"].tolist() == [1000.0, 2500.5]


def test_ambiguous_comma_number_is_rejected():
    matrix = pd.DataFrame({"Value": ["1,2", "3,4"]}, index=["A1", "A2"])
    with pytest.raises(MCDMValidationError, match="thousands separators"):
        validate_crisp_matrix(matrix)


def test_weights_match_criteria_and_are_normalized():
    weights = validate_weights({"C1": 2, "C2": 3}, ["C1", "C2"])
    assert weights == {"C1": 0.4, "C2": 0.6}
    with pytest.raises(MCDMValidationError, match="missing"):
        validate_weights({"C1": 1}, ["C1", "C2"])


def test_simple_entropy_rejects_negative_probabilities():
    matrix = pd.DataFrame({"C1": [-1.0, 2.0, 3.0], "C2": [1.0, 2.0, 3.0]})
    with pytest.raises(MCDMValidationError, match="non-negative"):
        calculate_entropy_weights(
            matrix, {"C1": "maximize", "C2": "maximize"}, method="simple"
        )


def test_merec_rejects_nonpositive_values_instead_of_clamping():
    matrix = pd.DataFrame({"C1": [0.0, 2.0], "C2": [1.0, 3.0]})
    with pytest.raises(MCDMValidationError, match="strictly positive"):
        calculate_merec_weights(
            matrix, {"C1": "maximize", "C2": "maximize"}
        )
