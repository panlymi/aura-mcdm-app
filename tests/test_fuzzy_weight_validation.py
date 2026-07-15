from __future__ import annotations

import math

import pytest

from mcdm.validation import MCDMValidationError, validate_fuzzy_weights


def test_fuzzy_weight_keys_must_match_criteria_exactly():
    with pytest.raises(
        MCDMValidationError,
        match=r"missing: Cost; unexpected: Other",
    ):
        validate_fuzzy_weights(
            {
                "Quality": (0.2, 0.3, 0.4),
                "Other": (0.1, 0.2, 0.3),
            },
            ["Quality", "Cost"],
            arity=3,
        )


def test_numeric_string_is_accepted_as_a_crisp_fuzzy_weight():
    result = validate_fuzzy_weights({"Quality": "0.5"}, ["Quality"], arity=3)

    assert result == {"Quality": (0.5, 0.5, 0.5)}


@pytest.mark.parametrize("invalid_weight", ["not-a-number", math.nan, math.inf, -1.0])
def test_malformed_or_invalid_scalar_weight_has_user_facing_error(invalid_weight):
    with pytest.raises(MCDMValidationError, match="(?i)weight"):
        validate_fuzzy_weights(
            {"Quality": invalid_weight},
            ["Quality"],
            arity=3,
        )


@pytest.mark.parametrize(
    "weights",
    [
        {"Quality": 0.0, "Cost": 0.0},
        {"Quality": (0.0, 0.0, 0.0), "Cost": (0.0, 0.0, 0.0)},
        {"Quality": 0.0, "Cost": (0.0, 0.0, 0.0)},
    ],
)
def test_fuzzy_weights_reject_zero_combined_total(weights):
    with pytest.raises(MCDMValidationError, match="greater than zero"):
        validate_fuzzy_weights(weights, ["Quality", "Cost"], arity=3)


def test_positive_fuzzy_weight_keeps_zero_weight_criteria_valid():
    result = validate_fuzzy_weights(
        {
            "Quality": (0.0, 0.2, 0.4),
            "Cost": 0.0,
        },
        ["Quality", "Cost"],
        arity=3,
    )

    assert result["Quality"] == (0.0, 0.2, 0.4)
    assert result["Cost"] == (0.0, 0.0, 0.0)
