# probabilisticnumerics

Bayesian finite-size scaling and active-learning experiments on the 2D Ising
model and scalar φ⁴ theory. Monte Carlo observables are treated as noisy
measurements; critical exponents and a nonparametric scaling function are
inferred jointly (Harada-style GP FSS).

Packages:

- `ising` — Wolff sampling, FSS / GP models, active learning, notebooks
- `phi4` — lattice φ⁴ sampling and Binder / collapse analysis (reuses `ising` utilities)

## Install

Requires Python ≥ 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Tests

```bash
uv run pytest tests/unit
# or
pytest tests/unit
```

## Notebooks

Editable install puts `ising` and `phi4` on the path. Explorers live under
`ising/` (e.g. `fss_explorer.ipynb`, `greedy_al_stepper.ipynb`) and `phi4/`
(e.g. `m_collapse_explorer.ipynb`).

```bash
uv sync --extra notebook
```

See [`ising/ROADMAP.md`](ising/ROADMAP.md) for the Ising FSS prototype plan.
