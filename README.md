# AURA MCDM App

A transparent Streamlit decision-support application for AURA and established
multi-criteria decision-making methods. It supports interactive decision-matrix
editing, manual or objective criterion weights, detailed calculation steps,
sensitivity analysis, an interactive AURA Monte Carlo workflow, and
capability-aware cross-method comparison.

## Supported methods

| Method | Benefit | Cost | Native target |
|---|:---:|:---:|:---:|
| AURA | Yes | Yes | Yes |
| ARIE | Yes | Yes | Yes |
| SYAI | Yes | Yes | Yes |
| ARAS / Fuzzy ARAS | Yes | Yes | No |
| MOORA | Yes | Yes | No |
| TOPSIS | Yes | Yes | No |
| SAW | Yes | Yes | No |
| VIKOR | Yes | Yes | No |

The application never silently treats a target as a benefit or cost. During a
comparative analysis, methods that do not natively support the configured
criterion types are excluded with an explanation. An explicit distance-from-target
preprocessing step would constitute an adapted method and should be reported as such.

## Architecture

- `aura_app.py` — Streamlit UI and detailed method presentation.
- `mcdm/criteria.py` — canonical criterion preferences and method capabilities.
- `mcdm/validation.py` — crisp, fuzzy, weight, and method precondition checks.
- `mcdm/analysis.py` — framework-neutral calculator dispatch and comparison.
- `mcdm/state.py` — deterministic input fingerprints used to invalidate stale results.
- `mcdm/presentation.py` — result-column and score-direction metadata.
- `mcdm/research.py` — canonical AURA Monte Carlo and reporting utilities.
- `*_calculator.py` — each method's published mathematical procedure.

The shared layer standardizes validation and preference meaning. It does **not**
replace each method's own normalization formula.

## Installation

Python 3.11 or newer is required.

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run streamlit run aura_app.py
```

Using `pip`:

```bash
python -m venv .venv
python -m pip install -e ".[research]"
streamlit run aura_app.py
```

For a Streamlit deployment, `requirements.txt` intentionally contains only the
runtime dependencies. Research-only plotting packages remain in the
`research` optional dependency group in `pyproject.toml`.

## Input format

Upload CSV or Excel with alternative names in the first column and criteria in
the remaining columns:

```csv
Alternative,Cost,Quality,Durability
Car A,20000,8,5
Car B,25000,9,7
Car C,18000,6,4
```

Rules enforced by the application:

- alternative and criterion names must be unique;
- all crisp cells must be finite numeric values;
- weights must be finite, non-negative, and have a positive total;
- crisp weights are always normalized to sum to one;
- reciprocal cost methods reject zero or negative cost values;
- fuzzy numbers must be ordered TFNs or TrFNs with consistent arity;
- fuzzy matrix values and fuzzy weights must use the same arity.

## Monte Carlo simulation in Streamlit

After running an AURA baseline calculation, open the **Monte Carlo Simulation**
tab. Two reproducible weight-sampling modes are available:

- **Global robustness** samples the complete weight simplex using
  `Dirichlet(1, ..., 1)`, matching the research simulation.
- **Local uncertainty** samples around the current normalized weights. Its
  concentration control determines how tightly samples remain near that centre.

The tab reports average Spearman rank correlation, Rank-1 retention, rank
dispersion, and a rank-acceptability heatmap. Summary results, acceptability
probabilities, raw simulated ranks, and sampled weights can all be downloaded as
CSV files. The random seed and iteration count are recorded for reproducibility.
The interactive workflow currently applies only to AURA because it calls the
canonical AURA simulation kernel.

## Tests

```bash
uv run pytest
uv run ruff check .
```

The suite covers golden benchmark outputs, dominant alternatives, target
capabilities, weight-scale invariance, ties, invalid reciprocal inputs, fuzzy
shape validation, calculation fingerprints, and agreement between the app and
research AURA paths, including global/local Monte Carlo sampling and rank
acceptability probabilities.

## Reproducing the research outputs

All scripts default to `Full result AURA_new_weights.csv` and accept CLI options:

```bash
uv run python new_baseline.py --help
uv run python new_baseline.py
uv run python generate_report.py --iterations 10000 --seed 42
uv run python generate_export.py
uv run python generate_paper_figures.py
uv run python monte_carlo_scenarios.py --iterations 10000 --seed 42
```

The Streamlit app and simulations call the same AURA normalization and scoring
kernel. Monte Carlo criterion counts are derived from the matrix instead of
being fixed at five, and ties use competition ranking consistently.

Generated Excel reports, JSON summaries, and PNG figures are intentionally not
stored in the repository. The commands above recreate them locally from the
curated research dataset, keeping the deployment checkout small without losing
research reproducibility.

## Method references

- Kamarul Zaman et al., “AURA: An Adaptive Utility Ranking Algorithm for
  Multi-Criteria Decision Making with Python-based decision support interface,”
  *SoftwareX* 32 (2025), 102395. DOI: 10.1016/j.softx.2025.102395.
- Fauzi et al., “A New Method for Multi-Criteria Decision-Making: Adaptive
  Ranking with Ideal Evaluation (ARIE),” *European Journal of Pure and Applied
  Mathematics* 18(4) (2025), 6578.
- Brauers and Zavadskas, “Multi-objective Optimization with Discrete Alternatives
  on the Basis of Ratio Analysis,” *Intellectual Economics* 2(6) (2009).

## License

MIT. See `LICENSE`.
