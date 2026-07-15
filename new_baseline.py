"""Regenerate the AURA baseline through the canonical calculator."""

from __future__ import annotations

import argparse

import pandas as pd

from aura_calculator import calculate_aura
from mcdm.research import directions_from_types
from mcdm.validation import validate_crisp_matrix, validate_weights


DEFAULT_DATASET = "Full result AURA_new_weights.csv"
DEFAULT_CRITERIA_TYPES = (1, 1, -1, -1, 0)
DEFAULT_WEIGHTS = (0.20, 0.25, 0.20, 0.25, 0.10)


def generate_new_baseline(
    input_path: str = DEFAULT_DATASET,
    output_path: str = DEFAULT_DATASET,
    *,
    criteria_types=DEFAULT_CRITERIA_TYPES,
    weights=DEFAULT_WEIGHTS,
    target: float = 0.65,
    alpha: float = 0.5,
    p: int = 2,
) -> pd.DataFrame:
    source = pd.read_csv(input_path)
    criteria = source.columns[1 : 1 + len(criteria_types)].tolist()
    matrix = validate_crisp_matrix(source.set_index("Alternatives")[criteria])
    directions = directions_from_types(criteria, criteria_types, target)
    weight_dict = validate_weights(dict(zip(criteria, weights)), criteria, normalize=True)
    result = calculate_aura(matrix, weight_dict, directions, alpha, p).reindex(matrix.index)

    output = source[["Alternatives", *criteria]].copy()
    for column in ["D+ (PIS)", "D- (NIS)", "D_avg (AS)", "Utility Score", "Rank"]:
        output[column] = result[column].to_numpy()
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_DATASET)
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--p", type=int, choices=(1, 2), default=2)
    parser.add_argument("--criteria-types", type=int, nargs="+", default=list(DEFAULT_CRITERIA_TYPES))
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    args = parser.parse_args()
    if len(args.criteria_types) != len(args.weights):
        parser.error("--criteria-types and --weights must contain the same number of values")
    generate_new_baseline(
        args.input,
        args.output,
        criteria_types=tuple(args.criteria_types),
        weights=tuple(args.weights),
        target=args.target,
        alpha=args.alpha,
        p=args.p,
    )
    print(f"Baseline saved to {args.output}")


if __name__ == "__main__":
    main()
