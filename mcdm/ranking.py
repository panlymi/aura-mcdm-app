"""Ranking and stable natural-order helpers."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def natural_sort_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(text) if text.isdigit() else text.casefold()
        for text in re.split(r"(\d+)", str(value))
    )


def rank_scores(scores: pd.Series, *, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Cannot rank non-finite scores.")
    return numeric.rank(ascending=ascending, method="min").astype(int)


def rank_array(scores: Iterable[float], *, ascending: bool = True) -> np.ndarray:
    series = pd.Series(np.asarray(list(scores), dtype=float))
    return rank_scores(series, ascending=ascending).to_numpy(dtype=int)
