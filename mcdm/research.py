"""Reproducible MCDM research and Monte Carlo utilities.

AURA simulations share its canonical calculator kernel. The other non-fuzzy
methods use vectorized equivalents that are regression-tested against their
interactive calculators.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aura_calculator import calculate_aura, prepare_aura_matrix
from .analysis import calculate_method
from .criteria import (
    CriterionPreference,
    CriterionType,
    normalize_directions,
    validate_method_capabilities,
)
from .presentation import RESULT_PRESENTATION
from .validation import MCDMValidationError, validate_crisp_matrix, validate_weights


MAX_MONTE_CARLO_WORKLOAD = 10_000_000
MAX_MONTE_CARLO_ITERATIONS = 20_000
_MONTE_CARLO_CHUNK_CELL_BUDGET = 2_000_000
CRISP_MONTE_CARLO_METHODS = (
    "AURA",
    "ARAS",
    "SYAI",
    "ARIE",
    "MOORA",
    "TOPSIS",
    "SAW",
    "VIKOR",
)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MCDMValidationError(f"{name} must be a positive integer.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise MCDMValidationError(f"{name} must be a positive integer.") from exc
    if integer <= 0:
        raise MCDMValidationError(f"{name} must be a positive integer.")
    return int(integer)


def validate_monte_carlo_iterations(iterations: int) -> int:
    """Return a validated public Monte Carlo iteration count."""

    iteration_count = _positive_integer(iterations, "iterations")
    if iteration_count > MAX_MONTE_CARLO_ITERATIONS:
        raise MCDMValidationError(
            f"Monte Carlo simulations are limited to {MAX_MONTE_CARLO_ITERATIONS:,} "
            "iterations per run."
        )
    return iteration_count


def validate_monte_carlo_workload(
    iterations: int, alternatives: int, criteria: int
) -> int:
    """Return the simulation workload after enforcing the public ceilings."""

    iteration_count = _positive_integer(iterations, "iterations")
    alternative_count = _positive_integer(alternatives, "alternatives")
    criterion_count = _positive_integer(criteria, "criteria")
    workload = iteration_count * alternative_count * criterion_count
    if workload > MAX_MONTE_CARLO_WORKLOAD:
        raise MCDMValidationError(
            f"Monte Carlo workload is {workload:,} cells, exceeding the "
            f"{MAX_MONTE_CARLO_WORKLOAD:,}-cell limit. Reduce the iterations or data size."
        )
    validate_monte_carlo_iterations(iteration_count)
    return workload


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


def _calculate_aura_utility_batch(
    normalized_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    *,
    alpha: float,
    p: int,
) -> np.ndarray:
    """Apply the scalar AURA scoring operations to a batch of weight rows."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if p not in {1, 2}:
        raise ValueError("p must be 1 (Manhattan) or 2 (Euclidean).")

    normalized_weights = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
    weighted = normalized_matrix[None, :, :] * normalized_weights[:, None, :]
    pis = weighted.max(axis=1)
    nis = weighted.min(axis=1)
    average = weighted.mean(axis=1)

    def distance(reference: np.ndarray) -> np.ndarray:
        deviations = np.abs(weighted - reference[:, None, :])
        if p == 1:
            return deviations.sum(axis=2)
        return np.power(np.power(deviations, p).sum(axis=2), 1 / p)

    def correct(values: np.ndarray) -> np.ndarray:
        sigma = values.max(axis=1) - values.min(axis=1)
        return values + sigma[:, None] * np.square(values)

    d_plus = correct(distance(pis))
    d_minus = correct(distance(nis))
    d_average = correct(distance(average))
    return (alpha * (d_plus - d_minus) + (1.0 - alpha) * d_average) / 2.0


def _monte_carlo_method_key(method: str) -> str:
    method_key = str(method).strip().upper()
    if method_key not in CRISP_MONTE_CARLO_METHODS:
        supported = ", ".join(CRISP_MONTE_CARLO_METHODS)
        raise MCDMValidationError(
            "Monte Carlo simulation supports non-fuzzy ranking methods only: "
            f"{supported}."
        )
    return method_key


def _validated_simulated_weights(
    simulated_weights: np.ndarray, criteria_count: int
) -> np.ndarray:
    try:
        weight_matrix = np.asarray(simulated_weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MCDMValidationError("Simulated weights must be numeric.") from exc
    if weight_matrix.ndim != 2 or weight_matrix.shape[1] != criteria_count:
        raise MCDMValidationError(
            "Each simulated weight vector must match the criteria count."
        )
    if not np.isfinite(weight_matrix).all():
        raise MCDMValidationError("Simulated weights must contain only finite numbers.")
    if np.any(weight_matrix < 0):
        raise MCDMValidationError("Simulated weights must be non-negative.")
    with np.errstate(over="ignore"):
        weight_totals = weight_matrix.sum(axis=1)
    if not np.isfinite(weight_totals).all() or np.any(weight_totals <= 0):
        raise MCDMValidationError(
            "Each simulated weight vector must have a finite, positive total."
        )
    return weight_matrix


def _validated_baseline_ranks(
    baseline_ranks: Sequence[int] | None, alternatives_count: int
) -> np.ndarray | None:
    if baseline_ranks is None:
        return None
    try:
        baseline = np.asarray(baseline_ranks, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MCDMValidationError(
            "Baseline ranks must be finite and match the alternatives count."
        ) from exc
    if baseline.shape != (alternatives_count,) or not np.isfinite(baseline).all():
        raise MCDMValidationError(
            "Baseline ranks must be finite and match the alternatives count."
        )
    return baseline


def _preference_masks(
    columns: Sequence[str], preferences: Mapping[str, CriterionPreference]
) -> tuple[np.ndarray, np.ndarray]:
    benefit = np.asarray(
        [preferences[str(column)].kind is CriterionType.BENEFIT for column in columns],
        dtype=bool,
    )
    cost = np.asarray(
        [preferences[str(column)].kind is CriterionType.COST for column in columns],
        dtype=bool,
    )
    return benefit, cost


def _prepare_method_batch_context(
    method: str,
    frame: pd.DataFrame,
    preferences: Mapping[str, CriterionPreference],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Precompute weight-independent arrays for a crisp-method score kernel."""

    values = frame.to_numpy(dtype=float)
    columns = [str(column) for column in frame.columns]
    benefit, cost = _preference_masks(columns, preferences)

    if method == "ARAS":
        optimal = np.where(benefit, values.max(axis=0), values.min(axis=0))
        extended = np.vstack([optimal, values])
        normalized = np.zeros_like(extended, dtype=float)
        for criterion_index in range(values.shape[1]):
            column = extended[:, criterion_index]
            if benefit[criterion_index]:
                denominator = column.sum()
                if abs(denominator) > 1e-9:
                    normalized[:, criterion_index] = column / denominator
            else:
                reciprocal = 1.0 / column
                denominator = reciprocal.sum()
                if abs(denominator) > 1e-9 and np.isfinite(denominator):
                    normalized[:, criterion_index] = reciprocal / denominator
        return {"alternatives": normalized[1:], "optimal": normalized[0]}

    if method == "ARIE":
        normalized = np.empty_like(values, dtype=float)
        for criterion_index, column_name in enumerate(columns):
            column = values[:, criterion_index]
            maximum = column.max()
            minimum = column.min()
            preference = preferences[column_name]
            if preference.kind is CriterionType.TARGET:
                target = float(preference.target_value)
                max_difference = max(abs(maximum - target), abs(minimum - target))
                normalized[:, criterion_index] = (
                    1.0
                    if abs(max_difference) < 1e-9
                    else 1.0 - (np.abs(column - target) / max_difference)
                )
            elif preference.kind is CriterionType.BENEFIT:
                normalized[:, criterion_index] = (
                    0.0 if abs(maximum) < 1e-9 else column / maximum
                )
            else:
                normalized[:, criterion_index] = minimum / (column + 1e-9)
        return {
            "normalized": normalized,
            "gamma": float(parameters.get("gamma", 1.0)),
            "kappa": float(parameters.get("kappa", 0.5)),
        }

    if method in {"MOORA", "TOPSIS"}:
        denominators = np.sqrt(np.square(values).sum(axis=0))
        normalized = np.divide(
            values,
            denominators,
            out=np.zeros_like(values, dtype=float),
            where=np.abs(denominators) > 1e-9,
        )
        if method == "MOORA":
            return {
                "normalized": normalized,
                "signs": np.where(benefit, 1.0, -1.0),
            }
        maximum = normalized.max(axis=0)
        minimum = normalized.min(axis=0)
        ideal = np.where(benefit, maximum, minimum)
        anti_ideal = np.where(benefit, minimum, maximum)
        return {
            "ideal_squared_difference": np.square(normalized - ideal),
            "anti_ideal_squared_difference": np.square(normalized - anti_ideal),
        }

    if method == "SAW":
        normalized = np.zeros_like(values, dtype=float)
        maximum = values.max(axis=0)
        minimum = values.min(axis=0)
        for criterion_index in range(values.shape[1]):
            column = values[:, criterion_index]
            if benefit[criterion_index]:
                if abs(maximum[criterion_index]) > 1e-9:
                    normalized[:, criterion_index] = (
                        column / maximum[criterion_index]
                    )
            else:
                nonzero = np.abs(column) > 1e-9
                normalized[nonzero, criterion_index] = (
                    minimum[criterion_index] / column[nonzero]
                )
        return {"normalized": normalized}

    if method == "SYAI":
        normalized = np.empty_like(values, dtype=float)
        for criterion_index, column_name in enumerate(columns):
            column = values[:, criterion_index]
            maximum = column.max()
            minimum = column.min()
            value_range = maximum - minimum
            if value_range < 1e-9:
                normalized[:, criterion_index] = 1.0
                continue
            preference = preferences[column_name]
            if preference.kind is CriterionType.TARGET:
                ideal_value = float(preference.target_value)
            elif preference.kind is CriterionType.BENEFIT:
                ideal_value = maximum
            else:
                ideal_value = minimum
            normalized[:, criterion_index] = 0.01 + 0.99 * (
                1.0 - (np.abs(column - ideal_value) / value_range)
            )
        return {
            "ideal_difference": np.abs(normalized - normalized.max(axis=0)),
            "anti_ideal_difference": np.abs(
                normalized - normalized.min(axis=0)
            ),
            "beta": float(parameters.get("beta", 0.5)),
        }

    if method == "VIKOR":
        normalized_distance = np.zeros_like(values, dtype=float)
        maximum = values.max(axis=0)
        minimum = values.min(axis=0)
        for criterion_index in range(values.shape[1]):
            if benefit[criterion_index]:
                denominator = maximum[criterion_index] - minimum[criterion_index]
                if abs(denominator) >= 1e-9:
                    normalized_distance[:, criterion_index] = (
                        maximum[criterion_index] - values[:, criterion_index]
                    ) / denominator
            else:
                denominator = maximum[criterion_index] - minimum[criterion_index]
                if abs(denominator) >= 1e-9:
                    normalized_distance[:, criterion_index] = (
                        values[:, criterion_index] - minimum[criterion_index]
                    ) / denominator
        return {
            "normalized_distance": normalized_distance,
            "v": float(parameters.get("v", 0.5)),
        }

    raise MCDMValidationError(f"No Monte Carlo score kernel is available for {method}.")


def _calculate_method_score_batch(
    method: str, context: Mapping[str, Any], weight_matrix: np.ndarray
) -> np.ndarray:
    """Calculate one method's final score for a batch of normalized weights."""

    weights = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)

    if method == "ARAS":
        numerator = weights @ np.asarray(context["alternatives"], dtype=float).T
        denominator = weights @ np.asarray(context["optimal"], dtype=float)
        return np.divide(
            numerator,
            denominator[:, None],
            out=np.zeros_like(numerator),
            where=np.abs(denominator[:, None]) > 1e-9,
        )

    if method == "ARIE":
        normalized = np.asarray(context["normalized"], dtype=float)
        gamma = float(context["gamma"])
        kappa = float(context["kappa"])
        sim_best = np.zeros((weights.shape[0], normalized.shape[0]), dtype=float)
        sim_worst = np.zeros_like(sim_best)
        for criterion_index in range(normalized.shape[1]):
            weighted_column = (
                weights[:, criterion_index, None]
                * normalized[None, :, criterion_index]
            )
            best = weighted_column.max(axis=1)
            worst = weighted_column.min(axis=1)
            sim_best += np.power(
                weighted_column / (best[:, None] + 1e-9), gamma
            )
            sim_worst += np.power(
                worst[:, None] / (weighted_column + 1e-9), gamma
            )
        numerator = kappa * sim_best
        return numerator / (
            numerator + (1.0 - kappa) * sim_worst + 1e-9
        )

    if method == "MOORA":
        signed_weights = weights * np.asarray(context["signs"], dtype=float)
        return signed_weights @ np.asarray(context["normalized"], dtype=float).T

    if method == "TOPSIS":
        squared_weights = np.square(weights)
        distance_ideal = np.sqrt(
            squared_weights
            @ np.asarray(context["ideal_squared_difference"], dtype=float).T
        )
        distance_anti_ideal = np.sqrt(
            squared_weights
            @ np.asarray(context["anti_ideal_squared_difference"], dtype=float).T
        )
        denominator = distance_ideal + distance_anti_ideal
        return np.divide(
            distance_anti_ideal,
            denominator,
            out=np.zeros_like(distance_anti_ideal),
            where=denominator > 1e-9,
        )

    if method == "SAW":
        return weights @ np.asarray(context["normalized"], dtype=float).T

    if method == "SYAI":
        distance_ideal = (
            weights @ np.asarray(context["ideal_difference"], dtype=float).T
        )
        distance_anti_ideal = (
            weights @ np.asarray(context["anti_ideal_difference"], dtype=float).T
        )
        beta = float(context["beta"])
        numerator = (1.0 - beta) * distance_anti_ideal
        denominator = beta * distance_ideal + numerator
        return np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator >= 1e-9,
        )

    if method == "VIKOR":
        normalized_distance = np.asarray(context["normalized_distance"], dtype=float)
        utility = weights @ normalized_distance.T
        regret = np.zeros_like(utility)
        for criterion_index in range(normalized_distance.shape[1]):
            regret = np.maximum(
                regret,
                weights[:, criterion_index, None]
                * normalized_distance[None, :, criterion_index],
            )
        utility_min = utility.min(axis=1)
        utility_range = utility.max(axis=1) - utility_min
        regret_min = regret.min(axis=1)
        regret_range = regret.max(axis=1) - regret_min
        utility_term = np.divide(
            utility - utility_min[:, None],
            utility_range[:, None],
            out=np.zeros_like(utility),
            where=np.abs(utility_range[:, None]) > 1e-9,
        )
        regret_term = np.divide(
            regret - regret_min[:, None],
            regret_range[:, None],
            out=np.zeros_like(regret),
            where=np.abs(regret_range[:, None]) > 1e-9,
        )
        v_parameter = float(context["v"])
        return v_parameter * utility_term + (1.0 - v_parameter) * regret_term

    raise MCDMValidationError(f"No Monte Carlo score kernel is available for {method}.")


def simulate_method_weights(
    method: str,
    matrix: np.ndarray | pd.DataFrame,
    simulated_weights: np.ndarray,
    directions: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any] | None = None,
    baseline_ranks: Sequence[int] | None = None,
    chunk_size: int = 500,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate rank robustness for any supported non-fuzzy MCDM method."""

    method_key = _monte_carlo_method_key(method)
    frame = _matrix_frame(matrix)
    weight_matrix = _validated_simulated_weights(
        simulated_weights, frame.shape[1]
    )
    simulation_count = weight_matrix.shape[0]
    validate_monte_carlo_workload(
        simulation_count, frame.shape[0], frame.shape[1]
    )
    validated_chunk_size = _positive_integer(chunk_size, "chunk_size")
    effective_chunk_size = min(
        validated_chunk_size,
        max(
            1,
            _MONTE_CARLO_CHUNK_CELL_BUDGET
            // (frame.shape[0] * frame.shape[1]),
        ),
    )
    if progress_callback is not None and not callable(progress_callback):
        raise MCDMValidationError("progress_callback must be callable.")
    baseline = _validated_baseline_ranks(baseline_ranks, frame.shape[0])
    method_parameters = dict(parameters or {})

    preferences = validate_method_capabilities(
        method_key, frame.columns, directions
    )
    validation_weights = dict(
        zip(frame.columns, weight_matrix[0], strict=True)
    )
    calculate_method(
        method_key,
        frame,
        validation_weights,
        directions,
        parameters=method_parameters,
    )

    if method_key == "AURA":
        criteria_types, target_values = types_and_targets_from_directions(
            frame.columns.tolist(), directions
        )
        return simulate_aura_weights(
            frame,
            weight_matrix,
            criteria_types,
            target_val=target_values,
            alpha=float(method_parameters.get("alpha", 0.5)),
            p=int(method_parameters.get("p", 2)),
            baseline_ranks=baseline,
            chunk_size=effective_chunk_size,
            progress_callback=progress_callback,
        )

    context = _prepare_method_batch_context(
        method_key, frame, preferences, method_parameters
    )
    score_ascending = RESULT_PRESENTATION[method_key].score_ascending
    rank_matrix = np.empty(
        (simulation_count, frame.shape[0]), dtype=np.int32
    )
    correlations = np.full(simulation_count, np.nan, dtype=float)

    for start in range(0, simulation_count, effective_chunk_size):
        stop = min(start + effective_chunk_size, simulation_count)
        scores = _calculate_method_score_batch(
            method_key, context, weight_matrix[start:stop]
        )
        if not np.isfinite(scores).all():
            raise MCDMValidationError(
                f"{method_key} produced non-finite Monte Carlo scores."
            )
        chunk_ranks = (
            pd.DataFrame(scores)
            .rank(axis=1, ascending=score_ascending, method="min")
            .to_numpy(dtype=np.int32)
        )
        rank_matrix[start:stop] = chunk_ranks
        if baseline is not None:
            for offset, ranks in enumerate(chunk_ranks):
                correlations[start + offset] = spearman_rank_correlation(
                    baseline, ranks
                )
        if progress_callback is not None:
            progress_callback(stop, simulation_count)
    return rank_matrix, correlations


def simulate_aura_weights(
    matrix: np.ndarray | pd.DataFrame,
    simulated_weights: np.ndarray,
    criteria_types: Sequence[int],
    *,
    target_val: float | Sequence[float] | Mapping[str, float] = 0.65,
    alpha: float = 0.5,
    p: int = 2,
    baseline_ranks: Sequence[int] | None = None,
    chunk_size: int = 500,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate AURA ranks in bounded batches.

    ``progress_callback`` receives ``(completed_iterations, total_iterations)``
    after each completed chunk.
    """

    frame = _matrix_frame(matrix)
    directions = directions_from_types(frame.columns.tolist(), criteria_types, target_val)
    weight_matrix = np.asarray(simulated_weights, dtype=float)
    if weight_matrix.ndim != 2 or weight_matrix.shape[1] != frame.shape[1]:
        raise MCDMValidationError("Each simulated weight vector must match the criteria count.")
    if not np.isfinite(weight_matrix).all():
        raise MCDMValidationError("Simulated weights must contain only finite numbers.")
    if np.any(weight_matrix < 0):
        raise MCDMValidationError("Simulated weights must be non-negative.")
    with np.errstate(over="ignore"):
        weight_totals = weight_matrix.sum(axis=1)
    if not np.isfinite(weight_totals).all() or np.any(weight_totals <= 0):
        raise MCDMValidationError(
            "Each simulated weight vector must have a finite, positive total."
        )

    simulation_count = weight_matrix.shape[0]
    validate_monte_carlo_workload(
        simulation_count, frame.shape[0], frame.shape[1]
    )
    validated_chunk_size = _positive_integer(chunk_size, "chunk_size")
    effective_chunk_size = min(
        validated_chunk_size,
        max(
            1,
            _MONTE_CARLO_CHUNK_CELL_BUDGET
            // (frame.shape[0] * frame.shape[1]),
        ),
    )
    if progress_callback is not None and not callable(progress_callback):
        raise MCDMValidationError("progress_callback must be callable.")

    baseline = None
    if baseline_ranks is not None:
        baseline = np.asarray(baseline_ranks, dtype=float)
        if baseline.shape != (frame.shape[0],) or not np.isfinite(baseline).all():
            raise MCDMValidationError(
                "Baseline ranks must be finite and match the alternatives count."
            )

    normalized = prepare_aura_matrix(frame, directions).to_numpy(dtype=float)
    rank_matrix = np.empty((simulation_count, frame.shape[0]), dtype=int)
    correlations = np.full(simulation_count, np.nan, dtype=float)

    for start in range(0, simulation_count, effective_chunk_size):
        stop = min(start + effective_chunk_size, simulation_count)
        utilities = _calculate_aura_utility_batch(
            normalized,
            weight_matrix[start:stop],
            alpha=alpha,
            p=p,
        )
        chunk_ranks = (
            pd.DataFrame(utilities)
            .rank(axis=1, ascending=True, method="min")
            .to_numpy(dtype=int)
        )
        rank_matrix[start:stop] = chunk_ranks
        if baseline is not None:
            for offset, ranks in enumerate(chunk_ranks):
                correlations[start + offset] = spearman_rank_correlation(baseline, ranks)
        if progress_callback is not None:
            progress_callback(stop, simulation_count)
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

    criterion_count = _positive_integer(criteria_count, "criteria_count")
    iteration_count = validate_monte_carlo_iterations(iterations)

    if center_weights is None:
        alpha_vector = np.ones(criterion_count, dtype=float)
    else:
        center = np.asarray(center_weights, dtype=float)
        if center.shape != (criterion_count,):
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
    return rng.dirichlet(alpha_vector, size=iteration_count)


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
    validate_monte_carlo_workload(iterations, frame.shape[0], frame.shape[1])
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


def run_monte_carlo_method(
    method: str,
    matrix: np.ndarray | pd.DataFrame,
    base_ranks: Sequence[int],
    directions: Mapping[str, Any],
    iterations: int = 10_000,
    *,
    seed: int = 42,
    parameters: Mapping[str, Any] | None = None,
    center_weights: Sequence[float] | None = None,
    concentration: float = 50.0,
    chunk_size: int = 500,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate seeded weights and simulate any supported crisp ranking method."""

    frame = _matrix_frame(matrix)
    validate_monte_carlo_workload(iterations, frame.shape[0], frame.shape[1])
    weights = generate_dirichlet_weights(
        frame.shape[1],
        iterations,
        seed=seed,
        center_weights=center_weights,
        concentration=concentration,
    )
    return simulate_method_weights(
        method,
        frame,
        weights,
        directions,
        parameters=parameters,
        baseline_ranks=base_ranks,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
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
    iteration_count = validate_monte_carlo_iterations(iterations)
    if criteria_count < 3:
        raise MCDMValidationError("Constrained scenarios require at least three criteria.")
    if emphasized_index == constrained_index:
        raise MCDMValidationError("Scenario criterion indices must be different.")
    if not 0 <= emphasized_index < criteria_count or not 0 <= constrained_index < criteria_count:
        raise MCDMValidationError("Scenario criterion index is outside the matrix.")

    rng = np.random.default_rng(seed)
    weights = np.zeros((iteration_count, criteria_count), dtype=float)
    remaining_indices = [
        index
        for index in range(criteria_count)
        if index not in {emphasized_index, constrained_index}
    ]
    for row in range(iteration_count):
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
        rank_counts = np.bincount(
            ranks[:, alternative_index], minlength=alternatives_count + 1
        )[1:]
        probabilities = rank_counts.astype(float) / ranks.shape[0] * 100.0
        for rank, probability in enumerate(probabilities, start=1):
            rows.append(
                {
                    "Alternative": alternative,
                    "Rank": rank,
                    "Probability_Pct": float(probability),
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
