# Bayesian finite-size scaling on the 2D Ising model

Roadmap for a probabilistic-numerics prototype: noisy MC estimates of observables
near criticality, a Harada-style GP scaling function, and a marginal posterior on
critical exponents. Inspired by [Harada, Phys. Rev. E **84**, 056704 (2011)](https://arxiv.org/abs/1102.4149).

## Goal

Given short MC runs at a sparse grid of \((L, T)\), infer a posterior on
\((T_c, \nu, \beta)\) (and later \(\gamma\)) while treating the universal
scaling function \(\Phi\) nonparametrically with a Gaussian process.

**Validation targets (exact 2D square-lattice Ising, \(J=1\), \(k_B=1\)):**

| Quantity | Exact value |
|---|---|
| \(T_c\) | \(2 / \ln(1 + \sqrt{2}) \approx 2.269\) |
| \(\nu\) | \(1\) |
| \(\beta\) | \(1/8 = 0.125\) |
| \(\gamma\) | \(7/4 = 1.75\) |

Phase 1 fits magnetization only and recovers \((T_c, \nu, \beta)\). Susceptibility
and \(\gamma\) are a natural follow-on once the pipeline works.

## Target layout

```
ising/
├── ROADMAP.md              # this file
├── sampler.py              # Step 1: 2D Ising MC interface
├── simulate.py             # Step 1: run grid of (L, T), save observables + errors
├── model.py                # Step 2: PyMC FSS + GP model
├── fit.py                  # Step 3: load data, sample, save posterior
├── plot_posterior.py       # Step 3: histograms and data-collapse diagnostic
└── data/                   # simulated MC outputs (gitignored or small fixtures)
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

(`pymc>=5` and `arviz>=0.18` are already listed under optional dependencies in
`pyproject.toml`.)

---

## Step 1 — 2D Ising sampler

**Objective:** produce noisy estimates \(\hat{y}_i \pm \sigma_i\) of an observable
at each \((L_i, T_i)\), with \(\sigma_i\) derived from the effective sample size.

### Observable (phase 1)

Use the absolute magnetization per spin below \(T_c\):

\[
\hat{m}_i = \left\langle \frac{1}{L_i^2}\left|\sum_{x,y} s_{x,y}\right|\right\rangle
\]

Finite-size scaling ansatz (log form, better for PyMC):

\[
\log m(L, T) = -\frac{\beta}{\nu}\log L + \log \Phi_m\!\left(t\, L^{1/\nu}\right),
\qquad t = \frac{T - T_c}{T_c}.
\]

### Sampler choice

Prefer **importing** a maintained implementation if one fits; otherwise add a
minimal in-repo **Wolff (cluster) algorithm** (recommended for 2D Ising near
\(T_c\); local Metropolis is much slower and is only useful if we deliberately
want to simulate expensive sampling).

| Option | Notes |
|---|---|
| In-repo Wolff (`sampler.py`) | ~100 lines, no extra dependency, full control over \(N_{\mathrm{sweeps}}\) and \(N_{\mathrm{eff}}\) reporting |
| External package | Evaluate at implementation time; wrap behind the same `run_ising(L, T, n_sweeps, seed) -> dict` interface so the rest of the pipeline is unchanged |

### `sampler.py` interface

```python
@dataclass
class IsingRunResult:
    L: int
    T: float
    magnetization: float      # <|m|>
    magnetization_std: float  # std of <|m|> across bins / batch means
    n_eff: float              # effective sample count for magnetization
    n_sweeps: int

def run_ising(L: int, T: float, *, n_thermalize: int, n_sweeps: int, seed: int) -> IsingRunResult:
    ...
```

### `simulate.py` — initial sparse grid

Deliberately **short** runs so the exponent posterior starts wide:

| Parameter | Initial values |
|---|---|
| \(L\) | \(\{16, 32\}\) |
| \(T\) | \(\{2.10,\, 2.20,\, 2.269,\, 2.32\}\) (span \(T_c\), stay mostly below \(T_c\) for \(\langle|m|\rangle\)) |
| `n_thermalize` | 500 sweeps |
| `n_sweeps` | 2 000 sweeps (tune so \(\sigma_i\) is visibly large) |

Save to `data/ising_observables.parquet` (or `.csv`) with columns:
`L, T, magnetization, magnetization_std, n_eff, n_sweeps, seed`.

**Noise model for Step 2:**

\[
\sigma_i = \frac{\text{magnetization\_std}_i}{\sqrt{n_{\mathrm{eff},i}}}
\]

If blocking is not implemented in v1, use batch means over fixed-length segments
as a crude estimate of `magnetization_std` and set `n_eff` from integrated
autocorrelation time (even a binned estimate is fine for the prototype).

### Step 1 done when

- [ ] One command generates the sparse grid and writes `data/ising_observables.parquet`
- [ ] Data-collapse plot of \(\log \hat{m}\) vs \(T\) at fixed \(L\) looks qualitatively sensible (peaks / crossing near \(T_c\))
- [ ] Error bars are non-negligible relative to the signal

---

## Step 2 — PyMC model (GP scaling function + exponents)

**Objective:** joint posterior \(P(T_c, \nu, \beta, \Phi_m \mid \text{data})\) with
\(\Phi_m\) a GP on the scaling variable \(z = t\, L^{1/\nu}\).

### Collapse construction

For each data point \(i\) with \((L_i, T_i)\) and observed \(\hat{m}_i\):

1. Sample \((T_c, \nu, \beta)\) from priors.
2. Compute \(t_i = (T_i - T_c) / T_c\) and \(z_i = t_i\, L_i^{1/\nu}\).
3. Compute corrected response \(r_i = \hat{m}_i\, L_i^{\beta/\nu}\) (equivalently
   \(\log r_i = \log \hat{m}_i + (\beta/\nu)\log L_i\)).
4. Place a GP prior on \(\log \Phi_m\): \(\log \Phi_m(z) \sim \mathcal{GP}(0, k)\).
5. Likelihood: \(r_i \sim \mathcal{N}\!\left(\Phi_m(z_i),\, \sigma_i'\right)\) with
   \(\sigma_i'\) propagated from \(\hat{m}_i\) (linearization or log-normal
   approximation; log-normal is fine for v1).

### Priors (weakly informative)

```python
T_c  ~ Uniform(2.0, 2.5)
nu   ~ Uniform(0.5, 1.5)
beta ~ Uniform(0.05, 0.25)
# GP: Matérn 5/2 or RBF on z, plus GP noise alpha
```

### Implementation notes

- Use `pymc.gp.Marginal` with a squared-exponential or Matérn kernel on the
  **scaled abscissa** \(z_i\). Because \(z_i\) depends nonlinearly on \((T_c, \nu)\),
  recompute \(z\) inside the PyMC model graph at each log-prob evaluation.
- Start with a **single observable** (magnetization) and a **single GP**; do not
  add \(\gamma\) / susceptibility until Step 3 validates.
- If sampling is slow, a two-stage fallback is acceptable for v1:
  1. Grid or `scipy.optimize` over \((T_c, \nu, \beta)\) using the GP marginal
     likelihood (Harada's approach).
  2. NUTS on \((T_c, \nu, \beta)\) with the GP collapsed out.

`model.py` should expose:

```python
def build_fss_gp_model(
    L: np.ndarray,
    T: np.ndarray,
    magnetization: np.ndarray,
    sigma: np.ndarray,
) -> pm.Model:
    ...
```

### Step 2 done when

- [ ] Model builds without error on the Step 1 data
- [ ] A short NUTS run (e.g. 200 tuning + 200 draws) completes and produces finite \(T_c, \nu, \beta\)
- [ ] Posterior mass is in the vicinity of the exact values (loose check; data are sparse and noisy)

---

## Step 3 — Fit on few samples and plot histograms

**Objective:** run the model on the low-sample grid from Step 1, save the posterior,
and plot marginal histograms for \(T_c\), \(\nu\), and \(\beta\).

### `fit.py`

```bash
python -m ising.fit \
    --data data/ising_observables.parquet \
    --draws 1000 \
    --tune 1000 \
    --out data/posterior.nc
```

- Use `arviz.InferenceData` and save to NetCDF (`posterior.nc`).
- Log summary stats: mean, sd, 94% HDI for each exponent.

### `plot_posterior.py`

Produce:

1. **Marginal histograms** of \(T_c\), \(\nu\), \(\beta\) with vertical lines at exact
   values.
2. **Data-collapse plot:** for posterior mean \((T_c, \nu, \beta)\), plot
   \(L^{\beta/\nu} \hat{m}\) vs \(t\, L^{1/\nu}\) with one color per \(L\); points
   should approximately fall on a single curve.
3. **GP posterior mean** of \(\log \Phi_m(z)\) over a grid of \(z\), with 94% band
   (optional but good sanity check).

```bash
python -m ising.plot_posterior \
    --posterior data/posterior.nc \
    --data data/ising_observables.parquet \
    --out plots/
```

### Step 3 done when

- [ ] Histograms saved to `plots/posterior_exponents.png`
- [ ] Posterior is **wide** (as expected from short runs) but centered roughly near exact values
- [ ] Data-collapse plot shows approximate collapse at posterior mean exponents
- [ ] Exact values lie inside (or near) the 94% HDIs — if not, widen temperature/L grid before tightening priors

---

## Later (out of scope for this roadmap)

- **Active learning:** acquire next \((L, T)\) or extra sweeps to maximize expected
  reduction in \(\mathrm{Var}(\nu)\).
- **Susceptibility channel:** add \(\chi\) and infer \(\gamma\) jointly.
- **Binder cumulant:** useful for pinning \(T_c\) with less sensitivity to \(\beta\).
- **Expensive samplers:** replace Wolff with HMC or other costly MC when studying sampler cost.

---

## Suggested implementation order

1. `sampler.py` — Wolff + magnetization measurement
2. `simulate.py` — sparse grid, write parquet
3. `model.py` — PyMC GP + exponent model
4. `fit.py` — NUTS, save `posterior.nc`
5. `plot_posterior.py` — histograms + collapse plot

Work in a notebook first if preferred, then factor into the modules above once the
model is debugged.
