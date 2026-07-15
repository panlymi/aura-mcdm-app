# AURA MCDM App

A transparent Streamlit decision-support application for AURA and established
multi-criteria decision-making methods. It supports interactive decision-matrix
editing, manual or objective criterion weights, detailed calculation steps,
sensitivity analysis, an interactive non-fuzzy Monte Carlo workflow, and
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

Entropy weighting (EWM) and MEREC are implemented as their native benefit/cost
objective-weighting procedures. If any target criterion is configured, use
manual weights; applying an uncited distance-to-target transformation would be
an adapted weighting method.

## Architecture

- `aura_app.py` — Streamlit UI and detailed method presentation.
- `mcdm/criteria.py` — canonical criterion preferences and method capabilities.
- `mcdm/validation.py` — crisp, fuzzy, weight, and method precondition checks.
- `mcdm/analysis.py` — framework-neutral calculator dispatch and comparison.
- `mcdm/state.py` — deterministic input fingerprints used to invalidate stale results.
- `mcdm/uploads.py` — bounded, fingerprinted CSV/XLSX parsing for the public app.
- `mcdm/presentation.py` — result-column and score-direction metadata.
- `mcdm/research.py` — vectorized Monte Carlo and research-reporting utilities.
- `*_calculator.py` — each method's published mathematical procedure.

The shared layer standardizes validation and preference meaning. It does **not**
replace each method's own normalization formula.

## Installation

Python 3.11 or newer is required.

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-extras
uv run --locked streamlit run aura_app.py
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

- uploads must be UTF-8 CSV or XLSX files no larger than 10 MiB;
- matrices are limited to 500 alternatives, 50 criteria, and 25,000 data cells;
- alternative and criterion names must be unique;
- all crisp cells must be finite numeric values;
- weights must be finite, non-negative, and have a positive total;
- crisp weights are always normalized to sum to one;
- ratio-normalized benefit criteria in ARAS, Fuzzy ARAS, ARIE, and SAW must be
  non-negative and contain at least one positive value;
- reciprocal cost methods reject zero or negative cost values;
- fuzzy numbers must be ordered TFNs or TrFNs with consistent arity;
- fuzzy matrix values and fuzzy weights must use the same arity;
- fuzzy weights must match the criteria exactly and have a positive total;
- the SYAI trade-off parameter uses the published open interval `0 < beta < 1`.

## Monte Carlo simulation in Streamlit

After running any non-fuzzy method's baseline calculation, open the **Monte
Carlo Simulation** tab. AURA, ARAS, SYAI, ARIE, MOORA, TOPSIS, SAW, and VIKOR
are supported. Fuzzy ARAS is excluded because fuzzy-weight uncertainty requires
a separate sampling model. Two reproducible weight-sampling modes are available:

- **Global robustness** samples the complete weight simplex using
  `Dirichlet(1, ..., 1)`, matching the research simulation.
- **Local uncertainty** samples around the current normalized weights. Its
  concentration control determines how tightly samples remain near that centre.

The tab reports average Spearman rank correlation, Rank-1 retention, rank
dispersion, and a rank-acceptability heatmap. Summary results, acceptability
probabilities, raw simulated ranks, and sampled weights can all be downloaded as
CSV files. Raw tables are materialized only when their downloads are requested.
The random seed, iteration count, sampling mode, and local concentration are
recorded and fingerprinted for reproducibility; changed controls are marked stale
until the simulation is rerun. Runs can contain up to 20,000 simulations.
Interactive workloads remain capped at 10,000,000
iteration-alternative-criterion operations, so 20,000 simulations are available
when `alternatives × criteria <= 500`; larger matrices must use fewer
simulations. Vectorized method-specific kernels process work in chunks of at
most 500 while retaining each method's own score orientation, parameters,
criterion capabilities, tie handling, and seeded results.

Sensitivity analysis and cross-method comparison run only when their explicit
Run buttons are selected. Their results persist across ordinary Streamlit widget
reruns and are invalidated when the baseline or relevant analysis controls change.

## Tests

```bash
uv run --locked --all-extras pytest
uv run --locked --all-extras ruff check .
```

The suite covers golden benchmark outputs, dominant alternatives, target
capabilities, weight-scale invariance, ties, invalid reciprocal inputs, fuzzy
shape validation, calculation fingerprints, and agreement between the app and
research paths. Monte Carlo regression cases compare all eight non-fuzzy batch
kernels with their scalar calculators, including global/local sampling, native
targets, ties, and rank-acceptability probabilities. CI installs exclusively
from `uv.lock`, verifies the exported runtime `requirements.txt`, and runs on
Python 3.11, 3.12, and 3.13.

## Reproducing the research outputs

All scripts default to `Full result AURA_new_weights.csv` and accept CLI options:

```bash
uv run --locked --extra research python new_baseline.py --help
uv run --locked --extra research python new_baseline.py
uv run --locked --extra research python generate_report.py --iterations 10000 --seed 42
uv run --locked --extra research python generate_export.py
uv run --locked --extra research python generate_paper_figures.py
uv run --locked --extra research python monte_carlo_scenarios.py --iterations 10000 --seed 42
```

The Streamlit app and AURA simulations call the same AURA normalization and
scoring kernel. The other non-fuzzy methods use vectorized equivalents verified
against their public calculators. Monte Carlo criterion counts are derived from
the matrix instead of being fixed at five, and ties use competition ranking
consistently.

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
- [Wan Abdul Rahman et al., “A Novel Simplified Yielded Aggregation Index
  (SYAI) Method for Enhancing Multi-Criteria Decision-Making,” *European
  Journal of Pure and Applied Mathematics* 18(4) (2025), 6560](https://www.ejpam.com/index.php/ejpam/article/view/6560/2443).
- Brauers and Zavadskas, “Multi-objective Optimization with Discrete Alternatives
  on the Basis of Ratio Analysis,” *Intellectual Economics* 2(6) (2009).

## License

MIT. See `LICENSE`.
