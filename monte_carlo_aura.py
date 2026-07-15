"""Command-line entry point for canonical AURA Monte Carlo simulation."""

from __future__ import annotations

import argparse

import numpy as np

from mcdm.research import (
    calculate_aura_baseline,
    load_ranked_dataset,
    run_monte_carlo_aura as _run_monte_carlo_aura,
)


DEFAULT_DATASET = "Full result AURA_new_weights.csv"
DEFAULT_CRITERIA_TYPES = (1, 1, -1, -1, 0)


def run_monte_carlo_aura(
    matrix,
    base_ranks,
    criteria_types,
    iterations=10_000,
    *,
    seed=42,
    target_val=0.65,
):
    rank_matrix, correlations = _run_monte_carlo_aura(
        matrix,
        base_ranks,
        criteria_types,
        iterations=iterations,
        seed=seed,
        target_val=target_val,
    )
    finite = correlations[np.isfinite(correlations)]
    average = float(finite.mean()) if len(finite) else float("nan")
    print(f"--- Monte Carlo Results ({iterations:,} Iterations) ---")
    print(f"Average Spearman Correlation (r_s): {average:.4f}")
    print(f"Probability first alternative retains Rank 1: {np.mean(rank_matrix[:, 0] == 1) * 100:.2f}%")
    return rank_matrix, correlations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_DATASET, help="Ranked baseline CSV file")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument(
        "--criteria-types",
        type=int,
        nargs="+",
        default=list(DEFAULT_CRITERIA_TYPES),
        help="One value per criterion: 1 benefit, -1 cost, 0 target",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, matrix, baseline_ranks, _ = load_ranked_dataset(
        args.input, criteria_count=len(args.criteria_types)
    )
    run_monte_carlo_aura(
        matrix,
        baseline_ranks,
        args.criteria_types,
        iterations=args.iterations,
        seed=args.seed,
        target_val=args.target,
    )


if __name__ == "__main__":
    main()
