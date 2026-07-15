from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aura_calculator import calculate_aura_score_arrays, prepare_aura_matrix
from mcdm import research
from mcdm.ranking import rank_array
from mcdm.research import (
    MAX_MONTE_CARLO_WORKLOAD,
    directions_from_types,
    generate_dirichlet_weights,
    run_monte_carlo_aura,
    simulate_aura_weights,
    spearman_rank_correlation,
    validate_monte_carlo_workload,
)
from mcdm.validation import MCDMValidationError


def _scalar_reference(
    matrix: pd.DataFrame,
    simulated_weights: np.ndarray,
    criteria_types: list[int],
    *,
    alpha: float,
    p: int,
    baseline_ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    directions = directions_from_types(matrix.columns.tolist(), criteria_types)
    normalized = prepare_aura_matrix(matrix, directions).to_numpy(dtype=float)
    rank_rows: list[np.ndarray] = []
    correlations: list[float] = []
    for weights in simulated_weights:
        kernel = calculate_aura_score_arrays(normalized, weights, alpha=alpha, p=p)
        ranks = rank_array(kernel["utility"], ascending=True)
        rank_rows.append(ranks)
        correlations.append(spearman_rank_correlation(baseline_ranks, ranks))
    return np.vstack(rank_rows), np.asarray(correlations, dtype=float)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("chunk_size", [1, 2, 500])
def test_chunked_simulation_exactly_matches_scalar_kernel_with_ties(
    p: int, chunk_size: int
):
    matrix = pd.DataFrame(
        {
            "C1": [1.0, 1.0, 3.0, 0.0],
            "C2": [1.0, 1.0, 2.0, 5.0],
            "C3": [4.0, 4.0, 0.0, 2.0],
        },
        index=["A1", "A2", "A3", "A4"],
    )
    simulated_weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [0.1, 0.1, 0.8],
        ]
    )
    criteria_types = [1, -1, 1]
    baseline_ranks = np.array([1, 1, 3, 4])
    expected_ranks, expected_correlations = _scalar_reference(
        matrix,
        simulated_weights,
        criteria_types,
        alpha=0.2,
        p=p,
        baseline_ranks=baseline_ranks,
    )

    actual_ranks, actual_correlations = simulate_aura_weights(
        matrix,
        simulated_weights,
        criteria_types,
        alpha=0.2,
        p=p,
        baseline_ranks=baseline_ranks,
        chunk_size=chunk_size,
    )

    np.testing.assert_array_equal(actual_ranks, expected_ranks)
    np.testing.assert_allclose(
        actual_correlations,
        expected_correlations,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_seeded_monte_carlo_results_retain_the_frozen_reference():
    matrix = np.array(
        [[5.0, 2.0, 4.0], [3.0, 4.0, 7.0], [1.0, 8.0, 5.0]], dtype=float
    )
    expected_ranks = np.array(
        [
            [1, 2, 3],
            [3, 1, 2],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [2, 1, 3],
            [2, 1, 3],
        ]
    )
    expected_correlations = np.array([1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])

    ranks, correlations = run_monte_carlo_aura(
        matrix,
        np.array([1, 2, 3]),
        [1, -1, 1],
        iterations=8,
        seed=123,
    )

    np.testing.assert_array_equal(ranks, expected_ranks)
    np.testing.assert_allclose(correlations, expected_correlations, rtol=0, atol=0)


def test_progress_callback_reports_each_completed_chunk():
    matrix = np.array(
        [[5.0, 2.0, 4.0], [3.0, 4.0, 7.0], [1.0, 8.0, 5.0]], dtype=float
    )
    weights = generate_dirichlet_weights(3, 5, seed=7)
    updates: list[tuple[int, int]] = []

    simulate_aura_weights(
        matrix,
        weights,
        [1, -1, 1],
        chunk_size=2,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert updates == [(2, 5), (4, 5), (5, 5)]


def test_workload_validator_accepts_the_limit_and_rejects_one_cell_more():
    assert validate_monte_carlo_workload(1_000, 100, 100) == MAX_MONTE_CARLO_WORKLOAD
    with pytest.raises(MCDMValidationError, match="10,000,001"):
        validate_monte_carlo_workload(MAX_MONTE_CARLO_WORKLOAD + 1, 1, 1)


@pytest.mark.parametrize(
    ("iterations", "alternatives", "criteria"),
    [(0, 1, 1), (1, -1, 1), (1, 1.5, 1), (1, 1, True)],
)
def test_workload_validator_requires_positive_integer_dimensions(
    iterations: object, alternatives: object, criteria: object
):
    with pytest.raises(MCDMValidationError, match="positive integer"):
        validate_monte_carlo_workload(iterations, alternatives, criteria)  # type: ignore[arg-type]


def test_simulation_enforces_the_shared_workload_ceiling(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(research, "MAX_MONTE_CARLO_WORKLOAD", 17)
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    weights = np.full((3, 2), 0.5)

    with pytest.raises(MCDMValidationError, match="18 cells"):
        simulate_aura_weights(matrix, weights, [1, 1])


def test_monte_carlo_wrapper_checks_workload_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
):
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    sampler_called = False

    def unexpected_sampling(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal sampler_called
        sampler_called = True
        raise AssertionError("weight sampling must not run for an oversized workload")

    monkeypatch.setattr(research, "generate_dirichlet_weights", unexpected_sampling)

    with pytest.raises(MCDMValidationError, match="10,000,004 cells"):
        run_monte_carlo_aura(
            matrix,
            np.array([1, 2]),
            [1, -1],
            iterations=2_500_001,
        )

    assert not sampler_called


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_simulation_rejects_invalid_chunk_sizes(chunk_size: object):
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    weights = np.array([[0.5, 0.5]])

    with pytest.raises(MCDMValidationError, match="chunk_size must be a positive integer"):
        simulate_aura_weights(
            matrix,
            weights,
            [1, 1],
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([[np.nan, 1.0]]), "finite numbers"),
        (np.array([[-0.1, 1.1]]), "non-negative"),
        (np.array([[0.0, 0.0]]), "positive total"),
    ],
)
def test_simulation_rejects_invalid_weight_rows(weights: np.ndarray, message: str):
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(MCDMValidationError, match=message):
        simulate_aura_weights(matrix, weights, [1, 1])


def test_simulation_rejects_invalid_callback_and_baseline_shape():
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    weights = np.array([[0.5, 0.5]])

    with pytest.raises(MCDMValidationError, match="progress_callback must be callable"):
        simulate_aura_weights(
            matrix,
            weights,
            [1, 1],
            progress_callback="not callable",  # type: ignore[arg-type]
        )
    with pytest.raises(MCDMValidationError, match="Baseline ranks"):
        simulate_aura_weights(matrix, weights, [1, 1], baseline_ranks=[1])
