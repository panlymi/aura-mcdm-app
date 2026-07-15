"""Targeted policy-weight scenarios using the canonical AURA kernel."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mcdm.research import (
    generate_constrained_weights as _generate_constrained_weights,
    load_ranked_dataset,
    simulate_aura_weights,
    spearman_rank_correlation,
)


DEFAULT_DATASET = "Full result AURA_new_weights.csv"
DEFAULT_CRITERIA_TYPES = (1, 1, -1, -1, 0)


def generate_constrained_weights(
    scenario: str, iterations: int = 10_000, *, criteria_count: int = 5, seed: int = 42
) -> np.ndarray:
    scenario_key = scenario.upper()
    if scenario_key == "B":
        emphasized_index, constrained_index = 0, 3
    elif scenario_key == "C":
        emphasized_index, constrained_index = 3, 0
    else:
        raise ValueError("Scenario must be 'B' or 'C'.")
    return _generate_constrained_weights(
        criteria_count,
        iterations,
        emphasized_index=emphasized_index,
        constrained_index=constrained_index,
        seed=seed,
    )


def run_targeted_perturbation(
    matrix,
    base_ranks,
    criteria_types,
    scenario: str = "B",
    iterations: int = 10_000,
    *,
    seed: int = 42,
    target_val: float = 0.65,
):
    simulated_weights = generate_constrained_weights(
        scenario,
        iterations,
        criteria_count=np.asarray(matrix).shape[1],
        seed=seed,
    )
    rank_matrix, correlations = simulate_aura_weights(
        matrix,
        simulated_weights,
        criteria_types,
        target_val=target_val,
        baseline_ranks=base_ranks,
    )
    finite = correlations[np.isfinite(correlations)]
    average = float(finite.mean()) if len(finite) else float("nan")
    print(f"--- Scenario {scenario.upper()} ({iterations:,} Iterations) ---")
    print(f"Average Spearman Correlation (r_s): {average:.4f}")
    print(f"Probability first alternative retains Rank 1: {np.mean(rank_matrix[:, 0] == 1) * 100:.2f}%")
    return rank_matrix, correlations


def scenario_summary(
    alternatives, baseline_ranks, ranks_b: np.ndarray, ranks_c: np.ndarray
) -> pd.DataFrame:
    iterations_b = ranks_b.shape[0]
    iterations_c = ranks_c.shape[0]
    rows = []
    for index, name in enumerate(alternatives):
        values_b = ranks_b[:, index]
        values_c = ranks_c[:, index]
        rows.append(
            {
                "Alternative": name,
                "Baseline Rank": int(baseline_ranks[index]),
                "Mean Rank (Capitalist)": float(values_b.mean()),
                "Top-5 Freq % (Capitalist)": float(np.sum(values_b <= 5) / iterations_b * 100),
                "Min-Max (Capitalist)": f"{int(values_b.min())} - {int(values_b.max())}",
                "Mean Rank (Welfare)": float(values_c.mean()),
                "Top-5 Freq % (Welfare)": float(np.sum(values_c <= 5) / iterations_c * 100),
                "Min-Max (Welfare)": f"{int(values_c.min())} - {int(values_c.max())}",
                "Absolute Mean Shift": float(abs(values_b.mean() - values_c.mean())),
            }
        )
    return pd.DataFrame(rows).sort_values("Baseline Rank").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_DATASET)
    parser.add_argument("--output", default="AURA_Scenarios_BC_Results.xlsx")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--criteria-types", type=int, nargs="+", default=list(DEFAULT_CRITERIA_TYPES))
    args = parser.parse_args()

    source, matrix, baseline_ranks, _ = load_ranked_dataset(
        args.input, criteria_count=len(args.criteria_types)
    )
    ranks_b, _ = run_targeted_perturbation(
        matrix,
        baseline_ranks,
        args.criteria_types,
        scenario="B",
        iterations=args.iterations,
        seed=args.seed,
        target_val=args.target,
    )
    ranks_c, _ = run_targeted_perturbation(
        matrix,
        baseline_ranks,
        args.criteria_types,
        scenario="C",
        iterations=args.iterations,
        seed=args.seed + 1,
        target_val=args.target,
    )
    summary = scenario_summary(
        source["Alternatives"].tolist(), baseline_ranks, ranks_b, ranks_c
    )
    summary.to_excel(args.output, index=False)
    print(f"Scenario report saved to {args.output}")


if __name__ == "__main__":
    main()
