from __future__ import annotations

import numpy as np
import pandas as pd

from aura_calculator import calculate_aura
from mcdm.research import (
    calculate_aura_baseline,
    generate_dirichlet_weights,
    rank_acceptability_table,
    run_monte_carlo_aura,
    types_and_targets_from_directions,
)
from mcdm.state import calculation_fingerprint


def test_calculation_fingerprint_changes_with_every_result_input():
    matrix = pd.DataFrame({"C1": [1.0, 2.0], "C2": [3.0, 4.0]}, index=["A1", "A2"])
    directions = {"C1": "maximize", "C2": "minimize"}
    base = calculation_fingerprint(
        method="AURA",
        matrix=matrix,
        weights={"C1": 0.5, "C2": 0.5},
        directions=directions,
        parameters={"alpha": 0.5, "p": 2},
    )
    changed_weight = calculation_fingerprint(
        method="AURA",
        matrix=matrix,
        weights={"C1": 0.6, "C2": 0.4},
        directions=directions,
        parameters={"alpha": 0.5, "p": 2},
    )
    changed_parameter = calculation_fingerprint(
        method="AURA",
        matrix=matrix,
        weights={"C1": 0.5, "C2": 0.5},
        directions=directions,
        parameters={"alpha": 0.7, "p": 2},
    )
    edited = matrix.copy()
    edited.loc["A1", "C1"] = 1.5
    changed_matrix = calculation_fingerprint(
        method="AURA",
        matrix=edited,
        weights={"C1": 0.5, "C2": 0.5},
        directions=directions,
        parameters={"alpha": 0.5, "p": 2},
    )
    assert len({base, changed_weight, changed_parameter, changed_matrix}) == 4


def test_research_baseline_matches_application_calculator():
    matrix = pd.DataFrame(
        {"C1": [5.0, 3.0, 1.0], "C2": [2.0, 4.0, 8.0], "C3": [4.0, 7.0, 5.0]},
        index=["A1", "A2", "A3"],
    )
    weights = [0.4, 0.35, 0.25]
    types = [1, -1, 1]
    utility, ranks = calculate_aura_baseline(matrix, weights, types)
    result = calculate_aura(
        matrix,
        dict(zip(matrix.columns, weights)),
        {"C1": "maximize", "C2": "minimize", "C3": "maximize"},
        0.5,
        2,
    ).reindex(matrix.index)
    np.testing.assert_allclose(utility, result["Utility Score"])
    np.testing.assert_array_equal(ranks, result["Rank"])


def test_monte_carlo_supports_dynamic_criteria_count_and_seed():
    matrix = np.array(
        [[5.0, 2.0, 4.0], [3.0, 4.0, 7.0], [1.0, 8.0, 5.0]], dtype=float
    )
    base_ranks = np.array([1, 2, 3])
    first_ranks, first_corr = run_monte_carlo_aura(
        matrix, base_ranks, [1, -1, 1], iterations=25, seed=123
    )
    second_ranks, second_corr = run_monte_carlo_aura(
        matrix, base_ranks, [1, -1, 1], iterations=25, seed=123
    )
    assert first_ranks.shape == (25, 3)
    np.testing.assert_array_equal(first_ranks, second_ranks)
    np.testing.assert_allclose(first_corr, second_corr, equal_nan=True)


def test_application_directions_convert_to_research_types_and_targets():
    criteria_types, targets = types_and_targets_from_directions(
        ["Benefit", "Cost", "Goal"],
        {
            "Benefit": "maximize",
            "Cost": "minimize",
            "Goal": {"type": "target", "value": 7.5},
        },
    )
    assert criteria_types == [1, -1, 0]
    assert targets == {"Goal": 7.5}


def test_dirichlet_weight_sampling_supports_global_and_local_modes():
    global_first = generate_dirichlet_weights(3, 50, seed=9)
    global_second = generate_dirichlet_weights(3, 50, seed=9)
    local = generate_dirichlet_weights(
        3,
        2_000,
        seed=9,
        center_weights=[0.6, 0.3, 0.1],
        concentration=100.0,
    )

    np.testing.assert_array_equal(global_first, global_second)
    np.testing.assert_allclose(global_first.sum(axis=1), 1.0)
    np.testing.assert_allclose(local.sum(axis=1), 1.0)
    np.testing.assert_allclose(local.mean(axis=0), [0.6, 0.3, 0.1], atol=0.01)


def test_rank_acceptability_probabilities_sum_to_one_hundred_per_alternative():
    ranks = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2]], dtype=int)
    table = rank_acceptability_table(["A1", "A2", "A3"], ranks)
    totals = table.groupby("Alternative")["Probability_Pct"].sum()

    np.testing.assert_allclose(totals.to_numpy(), 100.0)
    assert np.isclose(
        table.loc[
            (table["Alternative"] == "A1") & (table["Rank"] == 1),
            "Probability_Pct",
        ].item(),
        200 / 3,
    )
