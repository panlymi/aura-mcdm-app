"""Strict validation for crisp and fuzzy MCDM inputs."""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .criteria import CriterionType, normalize_directions


class MCDMValidationError(ValueError):
    """User-facing validation error for invalid decision inputs."""


_THOUSANDS_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def _coerce_numeric(value: Any) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("blank value")
    if "," in text:
        if not _THOUSANDS_NUMBER.fullmatch(text):
            raise ValueError("commas must be valid thousands separators")
        text = text.replace(",", "")
    return float(text)


def validate_crisp_matrix(data: pd.DataFrame, *, minimum_alternatives: int = 1) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise MCDMValidationError("The decision matrix must be a pandas DataFrame.")
    if data.shape[0] < minimum_alternatives:
        raise MCDMValidationError(
            f"At least {minimum_alternatives} alternative(s) are required."
        )
    if data.shape[1] == 0:
        raise MCDMValidationError("The decision matrix must contain at least one criterion.")
    if not data.index.is_unique:
        raise MCDMValidationError("Alternative names must be unique.")
    if not data.columns.is_unique:
        raise MCDMValidationError("Criterion names must be unique.")

    converted = pd.DataFrame(index=data.index)
    errors: list[str] = []
    for column in data.columns:
        values: list[float] = []
        for alternative, value in data[column].items():
            try:
                numeric = _coerce_numeric(value)
                if not math.isfinite(numeric):
                    raise ValueError("value must be finite")
                values.append(numeric)
            except (TypeError, ValueError) as exc:
                if len(errors) < 5:
                    errors.append(f"{alternative!r} / {column!r}: {value!r} ({exc})")
        if not errors or len(values) == len(data.index):
            converted[column] = values

    if errors:
        raise MCDMValidationError(
            "Every criterion cell must contain a finite number. Invalid cells: " + "; ".join(errors)
        )
    converted.columns = data.columns
    return converted.astype(float)


def validate_weights(
    weights: Mapping[str, Any],
    criteria: Iterable[str],
    *,
    normalize: bool = True,
) -> dict[str, float]:
    criteria_list = list(criteria)
    missing = [criterion for criterion in criteria_list if criterion not in weights]
    extra = [criterion for criterion in weights if criterion not in criteria_list]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(map(str, missing))}")
        if extra:
            details.append(f"unexpected: {', '.join(map(str, extra))}")
        raise MCDMValidationError("Weights must match criteria exactly (" + "; ".join(details) + ").")

    parsed: dict[str, float] = {}
    for criterion in criteria_list:
        try:
            value = float(weights[criterion])
        except (TypeError, ValueError) as exc:
            raise MCDMValidationError(f"Weight for {criterion!r} is not numeric.") from exc
        if not math.isfinite(value) or value < 0:
            raise MCDMValidationError(f"Weight for {criterion!r} must be finite and non-negative.")
        parsed[criterion] = value

    total = sum(parsed.values())
    if total <= 0:
        raise MCDMValidationError("At least one criterion weight must be greater than zero.")
    if normalize:
        return {criterion: value / total for criterion, value in parsed.items()}
    return parsed


def validate_method_matrix(
    method: str, data: pd.DataFrame, directions: Mapping[str, Any]
) -> None:
    method_key = method.strip().upper()
    preferences = normalize_directions(data.columns, directions)
    ratio_normalized_methods = {"ARAS", "ARIE", "SAW"}
    if method_key in ratio_normalized_methods:
        invalid_benefits = [
            criterion
            for criterion, preference in preferences.items()
            if preference.kind is CriterionType.BENEFIT
            and ((data[criterion] < 0).any() or not (data[criterion] > 0).any())
        ]
        if invalid_benefits:
            raise MCDMValidationError(
                f"{method_key} ratio benefit normalization requires non-negative values "
                "with at least one positive value for: " + ", ".join(invalid_benefits)
            )

        invalid_costs = [
            criterion
            for criterion, preference in preferences.items()
            if preference.kind is CriterionType.COST and (data[criterion] <= 0).any()
        ]
        if invalid_costs:
            raise MCDMValidationError(
                f"{method_key} reciprocal cost normalization requires strictly positive values for: "
                + ", ".join(invalid_costs)
            )


def validate_entropy_input(data: pd.DataFrame, *, method: str) -> None:
    if method == "simple":
        bad = [column for column in data.columns if (data[column] < 0).any() or data[column].sum() <= 0]
        if bad:
            raise MCDMValidationError(
                "Simple-proportion entropy requires non-negative values and a positive column sum for: "
                + ", ".join(bad)
            )


def validate_merec_input(data: pd.DataFrame) -> None:
    bad = [column for column in data.columns if (data[column] <= 0).any()]
    if bad:
        raise MCDMValidationError(
            "MEREC requires strictly positive decision values for: " + ", ".join(bad)
        )


def validate_fuzzy_number(value: Any, *, expected_arity: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise MCDMValidationError(f"Expected a fuzzy tuple, received {value!r}.")
    if len(value) not in {3, 4}:
        raise MCDMValidationError("Fuzzy numbers must contain exactly three or four values.")
    if expected_arity is not None and len(value) != expected_arity:
        raise MCDMValidationError(
            f"All fuzzy numbers and fuzzy weights must contain {expected_arity} values."
        )
    try:
        parsed = tuple(float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise MCDMValidationError(f"Invalid fuzzy number: {value!r}.") from exc
    if not all(math.isfinite(part) for part in parsed):
        raise MCDMValidationError("Fuzzy-number components must be finite.")
    if any(left > right for left, right in zip(parsed, parsed[1:])):
        raise MCDMValidationError(f"Fuzzy-number components must be non-decreasing: {value!r}.")
    return parsed


def validate_fuzzy_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if data.empty or data.shape[1] == 0:
        raise MCDMValidationError("The fuzzy decision matrix cannot be empty.")
    if not data.index.is_unique or not data.columns.is_unique:
        raise MCDMValidationError("Alternative and criterion names must be unique.")

    first = data.iloc[0, 0]
    arity = len(first) if isinstance(first, (tuple, list)) else None
    if arity not in {3, 4}:
        raise MCDMValidationError("The fuzzy matrix must contain TFNs or TrFNs.")

    validated = pd.DataFrame(index=data.index, columns=data.columns, dtype=object)
    errors: list[str] = []
    for column in data.columns:
        for alternative, value in data[column].items():
            try:
                validated.at[alternative, column] = validate_fuzzy_number(
                    value, expected_arity=arity
                )
            except MCDMValidationError as exc:
                if len(errors) < 5:
                    errors.append(f"{alternative!r} / {column!r}: {exc}")
    if errors:
        raise MCDMValidationError("Invalid fuzzy matrix: " + "; ".join(errors))
    return validated, arity


def validate_fuzzy_weights(
    weights: Mapping[str, Any], criteria: Iterable[str], *, arity: int
) -> dict[str, tuple[float, ...]]:
    criteria_list = list(criteria)
    missing = [criterion for criterion in criteria_list if criterion not in weights]
    extra = [criterion for criterion in weights if criterion not in criteria_list]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(map(str, missing))}")
        if extra:
            details.append(f"unexpected: {', '.join(map(str, extra))}")
        raise MCDMValidationError(
            "Fuzzy weights must match criteria exactly (" + "; ".join(details) + ")."
        )

    validated: dict[str, tuple[float, ...]] = {}
    for criterion in criteria_list:
        weight = weights[criterion]
        if not isinstance(weight, (tuple, list)):
            try:
                value = _coerce_numeric(weight)
            except (TypeError, ValueError) as exc:
                raise MCDMValidationError(
                    f"Weight for {criterion!r} must be a numeric scalar or an ordered "
                    f"{arity}-value fuzzy number."
                ) from exc
            if not math.isfinite(value) or value < 0:
                raise MCDMValidationError(
                    f"Crisp weight for {criterion!r} must be finite and non-negative."
                )
            validated[criterion] = (value,) * arity
        else:
            parsed = validate_fuzzy_number(weight, expected_arity=arity)
            if any(part < 0 for part in parsed):
                raise MCDMValidationError(f"Fuzzy weight for {criterion!r} cannot be negative.")
            validated[criterion] = parsed

    if not any(part > 0 for weight in validated.values() for part in weight):
        raise MCDMValidationError("At least one fuzzy criterion weight must be greater than zero.")
    return validated
