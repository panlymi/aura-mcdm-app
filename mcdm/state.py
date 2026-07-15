"""Stable fingerprints and derived-state helpers for the Streamlit app."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import MutableMapping
from datetime import date, datetime, time
from typing import Any, Mapping

import pandas as pd

from .criteria import normalize_directions


DERIVED_STATE_DEFAULTS: dict[str, Any] = {
    "calculated": False,
    "results_df": None,
    "steps_dict": None,
    "force_calculate": False,
    "ewm_steps": None,
    "merec_steps": None,
    "calculation_fingerprint": None,
    "sensitivity_result": None,
    "sensitivity_fingerprint": None,
    "comparison_result": None,
    "comparison_fingerprint": None,
    "monte_carlo_result": None,
    "monte_carlo_fingerprint": None,
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if value is pd.NaT:
        return "NaT"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if hasattr(value, "tolist") and not hasattr(value, "item"):
        return _canonical(value.tolist())
    if hasattr(value, "item"):
        try:
            item = value.item()
        except ValueError:
            return _canonical(value.tolist())
        if item is not value:
            return _canonical(item)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return format(value, ".17g")
    return value


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _canonical(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
            "data": [
                [_canonical(value) for value in row]
                for row in matrix.to_numpy(dtype=object).tolist()
            ],
        },
        "weights": _canonical(weights),
        "directions": {
            criterion: preference.to_legacy()
            for criterion, preference in normalized_directions.items()
        },
        "parameters": _canonical(parameters or {}),
    }
    return _payload_fingerprint(payload)


def analysis_fingerprint(
    *,
    baseline_fingerprint: str,
    analysis_name: str,
    controls: Mapping[str, Any],
) -> str:
    """Fingerprint analysis controls against the calculation that produced the baseline."""

    baseline = str(baseline_fingerprint).strip()
    analysis = str(analysis_name).strip().casefold()
    if not baseline:
        raise ValueError("A baseline calculation fingerprint is required.")
    if not analysis:
        raise ValueError("An analysis name is required.")
    return _payload_fingerprint(
        {
            "baseline_fingerprint": baseline,
            "analysis": analysis,
            "controls": controls,
        }
    )


def reset_derived_state(state: MutableMapping[str, Any]) -> None:
    """Reset calculation and analysis outputs while preserving unrelated UI controls."""

    for key, default in DERIVED_STATE_DEFAULTS.items():
        state[key] = default
