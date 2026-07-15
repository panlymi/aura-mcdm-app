from __future__ import annotations

import pandas as pd
import pytest

from fuzzy_aras_calculator import calculate_fuzzy_aras
from fuzzy_parser import parse_fuzzy_matrix, parse_fuzzy_weights, parse_tfn_string
from mcdm.validation import MCDMValidationError


@pytest.mark.parametrize("arity", [3, 4])
def test_fuzzy_aras_supports_consistent_tfn_and_trfn(arity):
    if arity == 3:
        values = [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)]
        weight = (0.2, 0.3, 0.4)
    else:
        values = [(1.0, 2.0, 3.0, 4.0), (2.0, 3.0, 4.0, 5.0)]
        weight = (0.2, 0.3, 0.4, 0.5)
    matrix = pd.DataFrame({"Quality": values}, index=["A1", "A2"])
    result = calculate_fuzzy_aras(
        matrix, {"Quality": weight}, {"Quality": "maximize"}
    )
    assert result.loc["A2", "Rank"] == 1


def test_trfn_matrix_rejects_tfn_weight():
    matrix = pd.DataFrame(
        {"Quality": [(1.0, 2.0, 3.0, 4.0), (2.0, 3.0, 4.0, 5.0)]},
        index=["A1", "A2"],
    )
    with pytest.raises(MCDMValidationError, match="must contain 4 values"):
        calculate_fuzzy_aras(
            matrix,
            {"Quality": (0.2, 0.3, 0.4)},
            {"Quality": "maximize"},
        )


def test_mixed_fuzzy_arities_are_rejected():
    matrix = pd.DataFrame(
        {"Quality": [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0, 5.0)]},
        index=["A1", "A2"],
    )
    with pytest.raises(MCDMValidationError, match="must contain 3 values"):
        calculate_fuzzy_aras(
            matrix,
            {"Quality": (0.2, 0.3, 0.4)},
            {"Quality": "maximize"},
        )


def test_fuzzy_parser_rejects_unordered_number():
    assert parse_tfn_string("3, 2, 4") is None


def test_fuzzy_parser_is_framework_independent():
    raw = pd.DataFrame({"Quality": ["Good", "Very Good"]}, index=["A1", "A2"])
    parsed = parse_fuzzy_matrix(raw, "Linguistic Terms")
    weights = parse_fuzzy_weights({"Quality": "Good"}, "Linguistic Terms")
    assert parsed.at["A1", "Quality"] == (0.6, 0.8, 1.0)
    assert weights["Quality"] == (0.6, 0.8, 1.0)


def test_fuzzy_cost_requires_positive_lower_bound():
    matrix = pd.DataFrame(
        {"Cost": [(0.0, 0.2, 0.4), (0.3, 0.5, 0.7)]}, index=["A1", "A2"]
    )
    with pytest.raises(MCDMValidationError, match="positive lower bounds"):
        calculate_fuzzy_aras(matrix, {"Cost": 1.0}, {"Cost": "minimize"})
