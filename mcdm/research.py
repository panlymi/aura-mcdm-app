"""Reproducible AURA research and Monte Carlo utilities.

All simulations use the same normalization and scoring kernel as the Streamlit
application.  This prevents the research scripts from drifting away from the
interactive calculator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aura_calculator import calculate_aura, calculate_aura_score_arrays, prepare_aura_matrix
from .criteria import CriterionType, normalize_directions
from .ranking import rank_array
from .validation import MCDMValidationError, validate_crisp_matrix, validate_weights


def directions_from_types(
    columns: Sequence[str],
    criteria_types: Sequence[int],
    target_values: float | Sequence[float] | Mapping[str, float] = 0.65,
) -> dict[str, str | dict[str, float | str]]:
    if len(columns) != len(criteria_types):
        raise MCDMValidationError("criteria_types length must match the matrix columns.")

    if isinstance(target_values, Mapping):
        targets = {str(key): float(value) for key, value in target_values.items()}
    elif np.isscalar(target_values):
        targets = {str(column): float(target_values) for column in columns}
    else:
        if len(target_values) != len(columns):
            raise MCDMValidationError("target_values length must match the matrix columns.")
        targets = {str(column): float(value) for column, value in zip(columns, target_values)}

    directions: dict[str, str | dict[str, float | str]] = {}
    for column, criterion_type in zip(columns, criteria_types):
        if criterion_type == 1:
            directions[str(column)] = "maximize"
        elif criterion_type == -1:
            directions[str(column)] = "minimize"
        elif criterion_type == 0:
            directions[str(column)] = {"type": "target", "value": targets[str(column)]}
        else:
            raise MCDMValidationError(
                f"Criterion type for {column!r} must be 1, -1, or 0."
            )
    return directions


def types_and_targets_from_directions(
    columns: Sequence[str], directions: Mapping[str, Any]
) -> tuple[list[int], dict[str, float]]:
    """Convert application preferences into the compact research representation."""

    normalized = normalize_directions(columns, directions)
    type_codes: list[int] = []
    targets: dict[str, float] = {}
    code_by_kind = {
        CriterionType.BENEFIT: 1,
        CriterionType.COST: -1,
        CriterionType.TARGET: 0,
    }
    for column in columns:
        preference = normalized[str(column)]
        type_codes.append(code_by_kind[preference.kind])
        if preference.kind is CriterionType.TARGET:
            targets[str(column)] = float(preference.target_value)
    return type_codes, targets


def _matrix_frame(matrix: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return validate_crisp_matrix(matrix)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise MCDMValidationError("The research matrix must be two-dimensional.")
    return validate_crisp_matrix(
        pd.DataFrame(
            array,
            index=[f"A{i + 1}" for i in range(array.shape[0])],
            columns=[f"C{j + 1}" for j in range(array.shape[1])],
        )
    )


def calculate_aura_baseline(
    matrix: np.ndarray | pd.DataFrame,
    weights: Sequence[float],
    criteria_types: Sequence[int],
    target_val: float | Sequence[float] | Mapping[str, float] = 0.65,
    *,
    alpha: float = 0.5,
    p: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _matrix_frame(matrix)
    directions = directions_from_types(frame.columns.tolist(), criteria_types, target_val)
    weight_dict = validate_weights(
        dict(zip(frame.columns, weights)), frame.columns, normalize=True
    )
    result = calculate_aura(frame, weight_dict, directions, alpha, p)
    result = result.reindex(frame.index)
    return (
        result["Utility Score"].to_numpy(dtype=float),
        result["Rank"].to_numpy(dtype=int),
    )


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    right_rank = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 1.0 if np.array_equal(left_rank, right_rank) else float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def simulate_aura_weights(
    matrix: np.ndarray | pd.DataFrame,
    simulated_weights: np.ndarray,
    criteria_types: Sequence[int],
    *,
    target_val: float | Sequence[float] | Mapping[str, float] = 0.65,
    alpha: float = 0.5,
    p: int = 2,
    baseline_ranks: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _matrix_frame(matrix)
    directions = directions_from_types(frame.columns.tolist(), criteria_types, target_val)
    normalized = prepare_aura_matrix(frame, directions).to_numpy(dtype=float)
    weight_matrix = np.asarray(simulated_weights, dtype=float)
    if weight_matrix.ndim != 2 or weight_matrix.shape[1] != frame.shape[1]:
        raise MCDMValidationError("Each simulated weight vector must match the criteria count.")

    rank_matrix = np.empty((weight_matrix.shape[0], frame.shape[0]), dtype=int)
    correlations = np.full(weight_matrix.shape[0], np.nan, dtype=float)
    for index, weights in enumerate(weight_matrix):
        kernel = calculate_aura_score_arrays(normalized, weights, alpha=alpha, p=p)
        ranks = rank_array(kernel["utility"], ascending=True)
        rank_matrix[index] = ranks
        if baseline_ranks is not None:
            correlations[index] = spearman_rank_correlation(baseline_ranks, ranks)
    return rank_matrix, correlations


def generate_dirichlet_weights(
    criteria_count: int,
    iterations: int,
    *,
    seed: int = 42,
    center_weights: Sequence[float] | None = None,
    concentration: float = 50.0,
) -> np.ndarray:
    """Generate reproducible global or locally centred criterion weights.

    With no centre, ``Dirichlet(1, ..., 1)`` samples the complete weight simplex
    uniformly.  With a centre, the normalized current weights define the mean
    and ``concentration`` controls how tightly samples cluster around it.
    """

    if criteria_count <= 0:
        raise MCDMValidationError("criteria_count must be greater than zero.")
    if iterations <= 0:
        raise MCDMValidationError("iterations must be greater than zero.")

    if center_weights is None:
        alpha_vector = np.ones(criteria_count, dtype=float)
    else:
        center = np.asarray(center_weights, dtype=float)
        if center.shape != (criteria_count,):
            raise MCDMValidationError(
                "The local Monte Carlo centre must match the criteria count."
            )
        if not np.isfinite(center).all() or np.any(center <= 0):
            raise MCDMValidationError(
                "Local Monte Carlo sampling requires strictly positive current weights."
            )
        if not np.isfinite(concentration) or concentration <= 0:
            raise MCDMValidationError("Dirichlet concentration must be greater than zero.")
        center = center / center.sum()
        alpha_vector = center * float(concentration)

    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha_vector, size=iterations)


def run_monte_carlo_aura(
    matrix: np.ndarray | pd.DataFrame,
    base_ranks: Sequence[int],
    criteria_types: Sequence[int],
    iterations: int = 10_000,
    *,
    seed: int = 42,
    target_val: float | Sequence[float] | Mapping[str, float] = 0.65,
    alpha: float = 0.5,
    p: int = 2,
    center_weights: Sequence[float] | None = None,
    concentration: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _matrix_frame(matrix)
    weights = generate_dirichlet_weights(
        frame.shape[1],
        iterations,
        seed=seed,
        center_weights=center_weights,
        concentration=concentration,
    )
    return simulate_aura_weights(
        frame,
        weights,
        criteria_types,
        target_val=target_val,
        alpha=alpha,
        p=p,
        baseline_ranks=base_ranks,
    )


def generate_constrained_weights(
    criteria_count: int,
    iterations: int,
    *,
    emphasized_index: int,
    constrained_index: int,
    emphasized_range: tuple[float, float] = (0.40, 0.70),
    constrained_range: tuple[float, float] = (0.05, 0.15),
    seed: int = 42,
) -> np.ndarray:
    if criteria_count < 3:
        raise MCDMValidationError("Constrained scenarios require at least three criteria.")
    if emphasized_index == constrained_index:
        raise MCDMValidationError("Scenario criterion indices must be different.")
    if not 0 <= emphasized_index < criteria_count or not 0 <= constrained_index < criteria_count:
        raise MCDMValidationError("Scenario criterion index is outside the matrix.")

    rng = np.random.default_rng(seed)
    weights = np.zeros((iterations, criteria_count), dtype=float)
    remaining_indices = [
        index
        for index in range(criteria_count)
        if index not in {emphasized_index, constrained_index}
    ]
    for row in range(iterations):
        emphasized = rng.uniform(*emphasized_range)
        constrained = rng.uniform(*constrained_range)
        remainder = 1.0 - emphasized - constrained
        if remainder < 0:
            raise MCDMValidationError("Scenario weight ranges can produce a negative remainder.")
        distribution = rng.dirichlet(np.ones(len(remaining_indices))) * remainder
        weights[row, emphasized_index] = emphasized
        weights[row, constrained_index] = constrained
        weights[row, remaining_indices] = distribution
    return weights


def summarize_rank_simulation(
    alternatives: Sequence[str], baseline_ranks: Sequence[int], rank_matrix: np.ndarray
) -> pd.DataFrame:
    alternatives = list(alternatives)
    baseline = np.asarray(baseline_ranks, dtype=int)
    ranks = np.asarray(rank_matrix, dtype=int)
    if ranks.ndim != 2 or ranks.shape[1] != len(alternatives):
        raise MCDMValidationError("Rank matrix columns must match the alternatives.")
    if baseline.shape != (len(alternatives),):
        raise MCDMValidationError("Baseline ranks must match the alternatives.")

    alternatives_count = len(alternatives)
    rows: list[dict[str, Any]] = []
    for index, alternative in enumerate(alternatives):
        values = ranks[:, index]
        rows.append(
            {
                "Alternative": alternative,
                "Baseline_Rank": int(baseline[index]),
                "Mean_Rank": float(values.mean()),
                "Rank_SD": float(values.std()),
                "Min_Rank": int(values.min()),
                "Max_Rank": int(values.max()),
                "Top_5_Freq_Pct": float(np.mean(values <= min(5, alternatives_count)) * 100),
                "Bottom_3_Freq_Pct": float(np.mean(values >= max(1, alternatives_count - 2)) * 100),
            }
        )
    return pd.DataFrame(rows).sort_values("Baseline_Rank").reset_index(drop=True)


def rank_acceptability_table(
    alternatives: Sequence[str], rank_matrix: np.ndarray
) -> pd.DataFrame:
    """Return the percentage probability of every alternative attaining each rank."""

    alternative_names = [str(alternative) for alternative in alternatives]
    raw_ranks = np.asarray(rank_matrix)
    if raw_ranks.ndim != 2 or raw_ranks.shape[1] != len(alternative_names):
        raise MCDMValidationError("Rank matrix columns must match the alternatives.")
    if raw_ranks.shape[0] == 0:
        raise MCDMValidationError("Rank matrix must contain at least one simulation.")
    if not np.isfinite(raw_ranks).all() or not np.equal(raw_ranks, np.floor(raw_ranks)).all():
        raise MCDMValidationError("Simulated ranks must be finite integers.")

    ranks = raw_ranks.astype(int)
    alternatives_count = len(alternative_names)
    if np.any(ranks < 1) or np.any(ranks > alternatives_count):
        raise MCDMValidationError(
            f"Simulated ranks must be between 1 and {alternatives_count}."
        )

    rows: list[dict[str, Any]] = []
    for alternative_index, alternative in enumerate(alternative_names):
        for rank in range(1, alternatives_count + 1):
            rows.append(
                {
                    "Alternative": alternative,
                    "Rank": rank,
                    "Probability_Pct": float(
                        np.mean(ranks[:, alternative_index] == rank) * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_ranked_dataset(
    path: str | Path, *, criteria_count: int | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    data = pd.read_csv(source)
    if "Alternatives" not in data or "Rank" not in data:
        raise MCDMValidationError("Dataset requires Alternatives and Rank columns.")
    if criteria_count is None:
        derived = {
            "D+ (PIS)", "D- (NIS)", "D_avg (AS)", "Utility Score", "Rank"
        }
        criterion_columns = [
            column for column in data.columns if column not in derived | {"Alternatives"}
        ]
    else:
        criterion_columns = data.columns[1 : 1 + criteria_count].tolist()
    matrix = validate_crisp_matrix(
        data.set_index("Alternatives")[criterion_columns]
    )
    ranks = pd.to_numeric(data["Rank"], errors="raise").to_numpy(dtype=int)
    utilities = (
        pd.to_numeric(data["Utility Score"], errors="raise").to_numpy(dtype=float)
        if "Utility Score" in data
        else np.full(len(data), np.nan)
    )
    return data, matrix.to_numpy(dtype=float), ranks, utilities
