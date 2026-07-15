"""Canonical criterion preferences and per-method capability validation.

The preference representation is shared, but each MCDM method retains its
published normalization formula.  In particular, target criteria are native to
AURA, ARIE, and SYAI in this application; they are not silently coerced for
classical methods that only support benefit and cost criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping


class CriterionType(str, Enum):
    BENEFIT = "maximize"
    COST = "minimize"
    TARGET = "target"


@dataclass(frozen=True)
class CriterionPreference:
    kind: CriterionType
    target_value: float | None = None

    def __post_init__(self) -> None:
        if self.kind is CriterionType.TARGET:
            if self.target_value is None or not math.isfinite(float(self.target_value)):
                raise ValueError("A target criterion requires a finite target value.")
        elif self.target_value is not None:
            raise ValueError("Only target criteria may define target_value.")

    def to_legacy(self) -> str | dict[str, float | str]:
        if self.kind is CriterionType.TARGET:
            return {"type": "target", "value": float(self.target_value)}
        return self.kind.value


class UnsupportedCriterionError(ValueError):
    """Raised when a method is asked to use an unsupported criterion type."""


METHOD_CAPABILITIES: dict[str, frozenset[CriterionType]] = {
    "AURA": frozenset({CriterionType.BENEFIT, CriterionType.COST, CriterionType.TARGET}),
    "ARIE": frozenset({CriterionType.BENEFIT, CriterionType.COST, CriterionType.TARGET}),
    "SYAI": frozenset({CriterionType.BENEFIT, CriterionType.COST, CriterionType.TARGET}),
    "ARAS": frozenset({CriterionType.BENEFIT, CriterionType.COST}),
    "FUZZY ARAS": frozenset({CriterionType.BENEFIT, CriterionType.COST}),
    "MOORA": frozenset({CriterionType.BENEFIT, CriterionType.COST}),
    "TOPSIS": frozenset({CriterionType.BENEFIT, CriterionType.COST}),
    "SAW": frozenset({CriterionType.BENEFIT, CriterionType.COST}),
    "VIKOR": frozenset({CriterionType.BENEFIT, CriterionType.COST}),
}


def parse_preference(value: Any) -> CriterionPreference:
    if isinstance(value, CriterionPreference):
        return value

    if isinstance(value, Mapping):
        raw_kind = str(value.get("type", "")).strip().lower()
        if raw_kind != "target":
            raise ValueError(f"Unsupported criterion mapping: {value!r}")
        target = value.get("value", value.get("target_value"))
        return CriterionPreference(CriterionType.TARGET, float(target) if target is not None else None)

    raw = str(value).strip().lower()
    aliases = {
        "maximize": CriterionType.BENEFIT,
        "maximise": CriterionType.BENEFIT,
        "benefit": CriterionType.BENEFIT,
        "max": CriterionType.BENEFIT,
        "minimize": CriterionType.COST,
        "minimise": CriterionType.COST,
        "cost": CriterionType.COST,
        "min": CriterionType.COST,
    }
    if raw == "target":
        raise ValueError("A target criterion must include its target value.")
    if raw not in aliases:
        raise ValueError(f"Unsupported criterion direction: {value!r}")
    return CriterionPreference(aliases[raw])


def normalize_directions(
    criteria: Iterable[str], directions: Mapping[str, Any]
) -> dict[str, CriterionPreference]:
    normalized: dict[str, CriterionPreference] = {}
    missing = [criterion for criterion in criteria if criterion not in directions]
    if missing:
        raise ValueError(f"Missing directions for criteria: {', '.join(map(str, missing))}")
    for criterion in criteria:
        normalized[str(criterion)] = parse_preference(directions[criterion])
    return normalized


def as_legacy_directions(
    criteria: Iterable[str], directions: Mapping[str, Any]
) -> dict[str, str | dict[str, float | str]]:
    return {
        criterion: preference.to_legacy()
        for criterion, preference in normalize_directions(criteria, directions).items()
    }


def validate_method_capabilities(
    method: str, criteria: Iterable[str], directions: Mapping[str, Any]
) -> dict[str, CriterionPreference]:
    method_key = method.strip().upper()
    if method_key not in METHOD_CAPABILITIES:
        raise ValueError(f"Unknown MCDM method: {method}")

    normalized = normalize_directions(criteria, directions)
    supported = METHOD_CAPABILITIES[method_key]
    unsupported = [name for name, preference in normalized.items() if preference.kind not in supported]
    if unsupported:
        kinds = sorted({normalized[name].kind.value for name in unsupported})
        raise UnsupportedCriterionError(
            f"{method_key} does not natively support {', '.join(kinds)} criteria: "
            f"{', '.join(unsupported)}. Exclude the method or explicitly transform the data "
            "and label the result as an adapted method."
        )
    return normalized


def compatible_methods(
    methods: Iterable[str], criteria: Iterable[str], directions: Mapping[str, Any]
) -> tuple[list[str], dict[str, str]]:
    supported: list[str] = []
    excluded: dict[str, str] = {}
    for method in methods:
        try:
            validate_method_capabilities(method, criteria, directions)
            supported.append(method)
        except UnsupportedCriterionError as exc:
            excluded[method] = str(exc)
    return supported, excluded
