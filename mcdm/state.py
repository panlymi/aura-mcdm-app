"""Stable calculation fingerprints for Streamlit result invalidation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import pandas as pd

from .criteria import normalize_directions


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, ".17g")
    return value


def calculation_fingerprint(
    *,
    method: str,
    matrix: pd.DataFrame,
    weights: Mapping[str, Any],
    directions: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
) -> str:
    normalized_directions = normalize_directions(matrix.columns, directions)
    payload = {
        "method": method.strip().upper(),
        "matrix": {
            "index": [_canonical(value) for value in matrix.index.tolist()],
            "columns": [str(value) for value in matrix.columns.tolist()],
            "data": [[_canonical(value) for value in row] for row in matrix.to_numpy(dtype=object).tolist()],
        },
        "weights": _canonical(weights),
        "directions": {
            criterion: preference.to_legacy()
            for criterion, preference in normalized_directions.items()
        },
        "parameters": _canonical(parameters or {}),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
