"""Framework-neutral calculation dispatch and comparative analysis."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from aras_calculator import calculate_aras
from arie_calculator import calculate_arie
from aura_calculator import calculate_aura
from fuzzy_aras_calculator import calculate_fuzzy_aras
from moora_calculator import calculate_moora
from saw_calculator import calculate_saw
from syai_calculator import calculate_syai
from topsis_calculator import calculate_topsis
from vikor_calculator import calculate_vikor

from .criteria import compatible_methods, validate_method_capabilities
from .presentation import RESULT_PRESENTATION


def calculate_method(
    method: str,
    matrix: pd.DataFrame,
    weights: Mapping[str, Any],
    directions: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    return_steps: bool = False,
):
    method_key = method.strip().upper()
    parameters = dict(parameters or {})
    validate_method_capabilities(method_key, matrix.columns, directions)

    if method_key == "AURA":
        return calculate_aura(matrix, weights, directions, parameters.get("alpha", 0.5), parameters.get("p", 2), return_steps=return_steps)
    if method_key == "ARAS":
        return calculate_aras(matrix, weights, directions, return_steps=return_steps)
    if method_key == "FUZZY ARAS":
        return calculate_fuzzy_aras(matrix, weights, directions, return_steps=return_steps)
    if method_key == "SYAI":
        return calculate_syai(matrix, weights, directions, parameters.get("beta", 0.5), return_steps=return_steps)
    if method_key == "ARIE":
        return calculate_arie(matrix, weights, directions, parameters.get("gamma", 1.0), parameters.get("kappa", 0.5), return_steps=return_steps)
    if method_key == "MOORA":
        return calculate_moora(matrix, weights, directions, return_steps=return_steps)
    if method_key == "TOPSIS":
        return calculate_topsis(matrix, weights, directions, return_steps=return_steps)
    if method_key == "SAW":
        return calculate_saw(matrix, weights, directions, return_steps=return_steps)
    if method_key == "VIKOR":
        return calculate_vikor(matrix, weights, directions, parameters.get("v", 0.5), return_steps=return_steps)
    raise ValueError(f"Unknown MCDM method: {method}")


def compare_methods(
    methods: Sequence[str],
    matrix: pd.DataFrame,
    weights: Mapping[str, Any],
    directions: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    compatible, excluded = compatible_methods(methods, matrix.columns, directions)
    rankings: dict[str, pd.Series] = {}
    for method in compatible:
        result = calculate_method(
            method, matrix, weights, directions, parameters=parameters, return_steps=False
        )
        metadata = RESULT_PRESENTATION[method.strip().upper()]
        rankings[method] = result[metadata.score_column].rank(
            ascending=metadata.score_ascending, method="min"
        ).astype(int)
    return pd.DataFrame(rankings), excluded
