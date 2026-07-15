from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mcdm.analysis import calculate_method
from mcdm.criteria import UnsupportedCriterionError
from mcdm.research import (
    CRISP_MONTE_CARLO_METHODS,
    MAX_MONTE_CARLO_ITERATIONS,
    rank_acceptability_table,
    simulate_method_weights,
    spearman_rank_correlation,
)
from mcdm.validation import MCDMValidationError


EXPECTED_CRISP_METHODS = {
    "AURA",
    "ARAS",
    "ARIE",
    "MOORA",
    "SAW",
    "SYAI",
    "TOPSIS",
    "VIKOR",
}

METHOD_PARAMETERS = {
    "AURA": {"alpha": 0.2, "p": 1},
    "ARAS": {},
    "ARIE": {"gamma": 1.3, "kappa": 0.35},
    "MOORA": {},
    "SAW": {},
    "SYAI": {"beta": 0.35},
    "TOPSIS": {},
    "VIKOR": {"v": 0.7},
}


@pytest.fixture
def benefit_cost_problem() -> tuple[pd.DataFrame, dict[str, str], np.ndarray]:
    matrix = pd.DataFrame(
        {
            "Benefit": [9.0, 7.0, 3.0, 7.0],
            "Cost": [2.0, 4.0, 8.0, 4.0],
            "Benefit2": [6.0, 8.0, 5.0, 8.0],
        },
        index=["A1", "A2", "A3", "A4"],
    )
    directions = {
        "Benefit": "maximize",
        "Cost": "minimize",
        "Benefit2": "benefit",
    }
    sampled_weights = np.array(
        [
            [0.40, 0.35, 0.25],
            [1.00, 0.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 0.00, 1.00],
            [2.00, 3.00, 5.00],
        ],
        dtype=float,
    )
    return matrix, directions, sampled_weights


def _scalar_rank_reference(
    method: str,
    matrix: pd.DataFrame,
    sampled_weights: np.ndarray,
    directions: dict[str, object],
    parameters: dict[str, object],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for weight_row in sampled_weights:
        weights = dict(zip(matrix.columns, weight_row))
        result = calculate_method(
            method,
            matrix,
            weights,
            directions,
            parameters=parameters,
        ).reindex(matrix.index)
        rows.append(result["Rank"].to_numpy(dtype=int))
    return np.vstack(rows)


def test_crisp_monte_carlo_registry_contains_every_non_fuzzy_ranking_method():
    assert set(CRISP_MONTE_CARLO_METHODS) == EXPECTED_CRISP_METHODS


@pytest.mark.parametrize("method", sorted(EXPECTED_CRISP_METHODS))
@pytest.mark.parametrize("chunk_size", [1, 2, 500])
def test_every_crisp_method_exactly_matches_scalar_rankings(
    method: str,
    chunk_size: int,
    benefit_cost_problem: tuple[pd.DataFrame, dict[str, str], np.ndarray],
):
    matrix, directions, sampled_weights = benefit_cost_problem
    parameters = METHOD_PARAMETERS[method]
    expected_ranks = _scalar_rank_reference(
        method,
        matrix,
        sampled_weights,
        directions,
        parameters,
    )
    baseline_ranks = expected_ranks[0]
    expected_correlations = np.array(
        [
            spearman_rank_correlation(baseline_ranks, rank_row)
            for rank_row in expected_ranks
        ],
        dtype=float,
    )

    actual_ranks, actual_correlations = simulate_method_weights(
        method,
        matrix,
        sampled_weights,
        directions,
        parameters=parameters,
        baseline_ranks=baseline_ranks,
        chunk_size=chunk_size,
    )

    np.testing.assert_array_equal(actual_ranks, expected_ranks)
    np.testing.assert_allclose(
        actual_correlations,
        expected_correlations,
        rtol=0,
        atol=1e-15,
        equal_nan=True,
    )


@pytest.mark.parametrize("method", ["AURA", "ARIE", "SYAI"])
@pytest.mark.parametrize("chunk_size", [1, 500])
def test_native_target_methods_exactly_match_scalar_rankings(
    method: str, chunk_size: int
):
    matrix = pd.DataFrame(
        {
            "Benefit": [1.0, 4.0, 7.0, 10.0],
            "Cost": [8.0, 4.0, 2.0, 1.0],
            "Target": [0.0, 5.0, 7.0, 10.0],
        },
        index=["A1", "A2", "A3", "A4"],
    )
    directions: dict[str, object] = {
        "Benefit": "benefit",
        "Cost": "cost",
        "Target": {"type": "target", "value": 6.0},
    }
    sampled_weights = np.array(
        [
            [0.4, 0.3, 0.3],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [0.6, 0.4, 0.0],
        ],
        dtype=float,
    )
    parameters = METHOD_PARAMETERS[method]
    expected = _scalar_rank_reference(
        method,
        matrix,
        sampled_weights,
        directions,
        parameters,
    )

    actual, _ = simulate_method_weights(
        method,
        matrix,
        sampled_weights,
        directions,
        parameters=parameters,
        chunk_size=chunk_size,
    )

    np.testing.assert_array_equal(actual, expected)


def test_twenty_thousand_iterations_are_accepted():
    assert MAX_MONTE_CARLO_ITERATIONS == 20_000
    matrix = pd.DataFrame({"C1": [1.0]}, index=["A1"])
    sampled_weights = np.ones((MAX_MONTE_CARLO_ITERATIONS, 1), dtype=float)

    ranks, correlations = simulate_method_weights(
        "SAW",
        matrix,
        sampled_weights,
        {"C1": "maximize"},
        chunk_size=MAX_MONTE_CARLO_ITERATIONS,
    )

    assert ranks.shape == (MAX_MONTE_CARLO_ITERATIONS, 1)
    assert np.all(ranks == 1)
    assert correlations.shape == (MAX_MONTE_CARLO_ITERATIONS,)
    assert np.isnan(correlations).all()


def test_more_than_twenty_thousand_iterations_are_rejected():
    matrix = pd.DataFrame({"C1": [1.0]}, index=["A1"])
    sampled_weights = np.ones((MAX_MONTE_CARLO_ITERATIONS + 1, 1), dtype=float)

    with pytest.raises(MCDMValidationError, match=r"20,?000"):
        simulate_method_weights(
            "SAW",
            matrix,
            sampled_weights,
            {"C1": "maximize"},
            chunk_size=MAX_MONTE_CARLO_ITERATIONS,
        )


@pytest.mark.parametrize("method", ["Fuzzy ARAS", "UNKNOWN METHOD"])
def test_fuzzy_and_unknown_methods_are_rejected(method: str):
    matrix = pd.DataFrame({"C1": [1.0, 2.0]}, index=["A1", "A2"])

    with pytest.raises(ValueError):
        simulate_method_weights(
            method,
            matrix,
            np.array([[1.0]]),
            {"C1": "maximize"},
        )


@pytest.mark.parametrize("method", ["ARAS", "MOORA", "SAW", "TOPSIS", "VIKOR"])
def test_target_incompatible_methods_are_rejected(method: str):
    matrix = pd.DataFrame({"Target": [1.0, 2.0, 3.0]}, index=["A1", "A2", "A3"])

    with pytest.raises(UnsupportedCriterionError, match="does not natively support"):
        simulate_method_weights(
            method,
            matrix,
            np.array([[1.0]]),
            {"Target": {"type": "target", "value": 2.0}},
        )


def test_non_aura_progress_callback_reports_completed_chunks(
    benefit_cost_problem: tuple[pd.DataFrame, dict[str, str], np.ndarray],
):
    matrix, directions, sampled_weights = benefit_cost_problem
    updates: list[tuple[int, int]] = []

    simulate_method_weights(
        "TOPSIS",
        matrix,
        sampled_weights,
        directions,
        chunk_size=2,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert updates == [(2, 5), (4, 5), (5, 5)]


def test_vikor_matches_scalar_at_the_exact_epsilon_range_boundary():
    matrix = pd.DataFrame(
        {"C1": [0.0, 1e-9]},
        index=["A1", "A2"],
    )
    directions = {"C1": "maximize"}
    sampled_weights = np.array([[1.0]])
    expected = _scalar_rank_reference(
        "VIKOR", matrix, sampled_weights, directions, {}
    )

    actual, _ = simulate_method_weights(
        "VIKOR", matrix, sampled_weights, directions
    )

    np.testing.assert_array_equal(actual, expected)


def test_backend_rank_dtype_does_not_overflow_beyond_uint16():
    alternatives_count = np.iinfo(np.uint16).max + 1
    matrix = pd.DataFrame(
        {"C1": np.arange(1, alternatives_count + 1, dtype=float)},
        index=[f"A{index}" for index in range(alternatives_count)],
    )

    ranks, _ = simulate_method_weights(
        "SAW",
        matrix,
        np.array([[1.0]]),
        {"C1": "maximize"},
    )

    assert ranks.dtype == np.int32
    assert ranks.min() == 1
    assert ranks.max() == alternatives_count
    assert not np.any(ranks == 0)


def test_rank_acceptability_optimized_result_matches_naive_tie_reference():
    alternatives = ["A1", "A2", "A3", "A4"]
    rank_matrix = np.array(
        [
            [1, 1, 3, 4],
            [2, 1, 2, 4],
            [1, 3, 1, 3],
            [1, 1, 1, 4],
        ],
        dtype=int,
    )
    expected_rows = []
    for alternative_index, alternative in enumerate(alternatives):
        for rank in range(1, len(alternatives) + 1):
            expected_rows.append(
                {
                    "Alternative": alternative,
                    "Rank": rank,
                    "Probability_Pct": float(
                        np.mean(rank_matrix[:, alternative_index] == rank) * 100
                    ),
                }
            )
    expected = pd.DataFrame(expected_rows)

    actual = rank_acceptability_table(alternatives, rank_matrix)

    pd.testing.assert_frame_equal(actual, expected)
