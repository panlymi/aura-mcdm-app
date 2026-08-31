"""Weighted Aggregated Sum Product Assessment (WASPAS)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from mcdm.criteria import CriterionType, validate_method_capabilities
from mcdm.ranking import natural_sort_key, rank_scores
from mcdm.validation import (
    MCDMValidationError,
    validate_crisp_matrix,
    validate_method_matrix,
    validate_weights,
)


def calculate_waspas(
    data: pd.DataFrame,
    weights: Mapping[str, Any],
    directions: Mapping[str, Any],
    lambda_value: float = 0.5,
    return_steps: bool = False,
):
    """Calculate WASPAS scores and competition ranks.

    WASPAS combines the Weighted Sum Model (WSM) and Weighted Product Model
    (WPM): ``Q_i = lambda * Q_i^(1) + (1-lambda) * Q_i^(2)``.  Higher scores
    are preferred.  The published method natively supports benefit and cost
    criteria through ratio normalization.
    """

    try:
        lambda_value = float(lambda_value)
    except (TypeError, ValueError) as exc:
        raise MCDMValidationError(
            "lambda must be numeric and between 0 and 1."
        ) from exc
    if not np.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise MCDMValidationError("lambda must be between 0 and 1.")

    frame = validate_crisp_matrix(data)
    columns = frame.columns
    preferences = validate_method_capabilities("WASPAS", columns, directions)
    validate_method_matrix("WASPAS", frame, directions)
    normalized_weights = validate_weights(weights, columns, normalize=True)

    steps: dict[str, pd.DataFrame] = {}
    if return_steps:
        steps["Step 1: Original Decision Matrix"] = frame.copy()

    normalized = pd.DataFrame(index=frame.index, columns=columns, dtype=float)
    for criterion in columns:
        values = frame[criterion]
        if preferences[criterion].kind is CriterionType.BENEFIT:
            normalized[criterion] = values / values.max()
        else:
            normalized[criterion] = values.min() / values

    weighted_sum_components = normalized.mul(
        pd.Series(normalized_weights), axis="columns"
    )
    weighted_product_components = pd.DataFrame(
        {
            criterion: np.power(normalized[criterion], normalized_weights[criterion])
            for criterion in columns
        },
        index=frame.index,
    )

    q_wsm = weighted_sum_components.sum(axis=1)
    q_wpm = weighted_product_components.prod(axis=1)
    q_waspas = lambda_value * q_wsm + (1.0 - lambda_value) * q_wpm
    ranks = rank_scores(q_waspas, ascending=False)

    score_table = pd.DataFrame(
        {
            "Q_i (WSM)": q_wsm,
            "Q_i (WPM)": q_wpm,
            "Q_i (WASPAS Score)": q_waspas,
        },
        index=frame.index,
    )
    results = pd.concat([frame, score_table], axis=1)
    results["Rank"] = ranks
    results["sort_index"] = results.index.map(
        lambda value: tuple(natural_sort_key(value))
    )
    results = results.sort_values(["Rank", "sort_index"]).drop(
        columns="sort_index"
    )

    if return_steps:
        steps["Step 2: Normalized Decision Matrix"] = normalized.copy()
        steps["Step 3: Weighted Sum Components"] = weighted_sum_components.copy()
        steps["Step 4: Weighted Product Components"] = (
            weighted_product_components.copy()
        )
        steps["Step 5: Aggregated Scores"] = score_table.copy()
        steps["Step 6: Final Result and Ranking"] = results.copy()
        return results, steps

    return results


__all__ = ["calculate_waspas"]
