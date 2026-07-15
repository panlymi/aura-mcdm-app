"""Generate a reproducible JSON Monte Carlo report from the current baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mcdm.research import load_ranked_dataset, run_monte_carlo_aura, summarize_rank_simulation


DEFAULT_DATASET = "Full result AURA_new_weights.csv"
DEFAULT_CRITERIA_TYPES = (1, 1, -1, -1, 0)


def generate_report(
    input_path: str = DEFAULT_DATASET,
    output_path: str = "report.json",
    *,
    iterations: int = 10_000,
    seed: int = 42,
    target: float = 0.65,
    criteria_types=DEFAULT_CRITERIA_TYPES,
) -> dict:
    source, matrix, baseline_ranks, _ = load_ranked_dataset(
        input_path, criteria_count=len(criteria_types)
    )
    rank_matrix, correlations = run_monte_carlo_aura(
        matrix,
        baseline_ranks,
        criteria_types,
        iterations=iterations,
        seed=seed,
        target_val=target,
    )
    summary = summarize_rank_simulation(
        source["Alternatives"].tolist(), baseline_ranks, rank_matrix
    )
    finite = correlations[np.isfinite(correlations)]
    result = {
        "source": str(input_path),
        "seed": seed,
        "target": target,
        "criteria_types": list(criteria_types),
        "average_spearman": float(finite.mean()) if len(finite) else None,
        "iterations": iterations,
        "metrics": summary.to_dict(orient="records"),
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_DATASET)
    parser.add_argument("--output", default="report.json")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--criteria-types", type=int, nargs="+", default=list(DEFAULT_CRITERIA_TYPES))
    args = parser.parse_args()
    generate_report(
        args.input,
        args.output,
        iterations=args.iterations,
        seed=args.seed,
        target=args.target,
        criteria_types=tuple(args.criteria_types),
    )
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
