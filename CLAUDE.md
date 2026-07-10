# CLAUDE.md

## Project Overview

Pypomp is a Python package for modeling and inference using Partially Observed Markov Process (POMP) models (also known as state-space models or hidden Markov models). It implements particle filtering, iterated filtering, gradient-based optimization, and other inference algorithms for highly nonlinear, non-Gaussian dynamical systems. The package leverages JAX for GPU support and just-in-time compilation, achieving significant performance improvements over the R Pomp package.

## Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.10+)
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Do not overwrite an existing environment.
```

### Additional Environment Setup for Testing

```bash
# Check that pytest is installed with the xdist plugin
pip list | grep pytest-

# Install pypomp in editable mode for development. This is mandatory: without
# `pip install -e .`, `import pypomp` may resolve to a stale site-packages copy
# and pytest/coverage will silently exercise the wrong files.
pip install -e .[tests,benchmarks,viz]

# `statsmodels` (pulled in by the [benchmarks] extra) is needed for some tests
# under tests/test_models/.
```

### Testing
```bash
# Run all tests (pytest.ini sets -n auto via xdist; tests/conftest.py
# configures a JAX persistent compilation cache under .pytest_cache/jax_cache).
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
pytest tests/test_pomp/test_pomp_pfilter.py

# Run specific test function
pytest tests/test_pomp/test_pomp_pfilter.py::test_class_basic_default

# Run tests for a specific module
pytest tests/test_pomp/
pytest tests/test_panel/
pytest tests/test_models/
```

### Linting and Type Checking
```bash
# Lint with ruff (configured in pyproject.toml)
ruff check .

# Type checking with pyright. Strict mode is enabled in pyproject.toml.
# CI only runs `pyright pypomp` (excluding tests); running `pyright` from
# the repo root will additionally surface test-file errors.
pyright
```


## Architecture Overview

### Top-level package layout

```
pypomp/
  __init__.py        # Public API surface (re-exports below)
  maths.py           # logmeanexp, logmeanexp_se, etc.
  mcap.py            # Monte Carlo adjusted profile (loess-based CI utility)
  types.py           # ThetaInput and other type aliases
  benchmarks/        # ARIMA / NegBin baselines (statsmodels wrappers)
  core/              # Implementation
    pomp.py          # The Pomp class (single-unit POMP model)
    parameters.py    # PompParameters, PanelParameters
    par_trans.py     # ParTrans (parameter transformations)
    rw_sigma.py      # RWSigma (random-walk sigma for IF2)
    model_struct.py  # _RInit, _RProc, _DMeas, _RMeas wrappers
    optimizer.py     # Optimizer config classes (Adam, SGD, BFGS, Newton, ...)
    learning_rate.py # LearningRate schedules
    metadata.py      # Model creation metadata
    viz.py           # plotly trace/simulation plots
    algorithms/      # Internal algorithm implementations
      pfilter.py
      mif.py         # IF2 (perturbed filter)
      mop.py         # Mollified One-Particle gradient
      train.py       # Differentiable particle filter optimization
      dpop.py        # DPOP gradient
      train_dpop.py  # DPOP-based training loop
      simulate.py
      helpers.py     # Covariate interpolation, key helpers, ...
    results/         # Result containers
      base.py        # BaseResult and shared mixins
      history.py     # ResultsHistory
      pomp.py        # PompPFilterResult, PompMIFResult, PompTrainResult
      panel.py       # Panel equivalents
  functional/        # Thin public functional API over core/algorithms
    pfilter.py, mif.py, mop.py, train.py, simulate.py, dpop.py
    structs.py       # PompStruct PyTree
  models/            # Example models
    linear_gaussian.py
    dacca.py         # Cholera (Dacca, Bangladesh)
    spx.py           # S&P 500 stochastic volatility
    sir.py           # SIR with seasonal forcing
    ctmc_multinom.py # CTMC multinomial-step helper
    measles/         # UK measles models (multiple variants 001 / 001b / ... / 003)
  panel/             # PanelPomp implementation
    panel.py         # PanelPomp class
    estimation_mixin.py
    analysis_mixin.py
    validation_mixin.py
    interfaces.py
  random/            # JAX random distributions (binom, gamma, nbinom, poisson)
  data/              # Bundled datasets used by example models
```

Note: `pypomp/panelPomp/` exists in the tree but currently contains only stale
`__pycache__` from a previous refactor. The actual panel implementation lives in
`pypomp/panel/`. Treat `panelPomp/` as a deletion candidate.

### Core Components

#### 1. Model classes

- **`Pomp`** (`pypomp/core/pomp.py`): Single-unit POMP model.
  - Encapsulates observation data (`ys`), parameters (`theta` via `PompParameters`),
    state names, and four wrapped components (`rinit`, `rproc`, `dmeas`, `rmeas`).
  - Provides inference methods: `simulate()`, `pfilter()`, `mif()`, `train()`,
    `dpop_train()`, `mop()` (via the underlying functional API).
  - Maintains `results_history` (`ResultsHistory`) and a `fresh_key` for JAX RNG.
  - Defines `to_struct()` which returns a `PompStruct` PyTree used by the functional
    API.
  - Implements `__eq__`, `merge`, `prune`, custom `__getstate__`/`__setstate__`
    (cloudpickle for model functions; raw key data for the JAX key).

- **`PanelPomp`** (`pypomp/panel/panel.py`): Multi-unit POMP models.
  - Manages multiple `Pomp` objects with shared and unit-specific parameters.
  - Uses mixin pattern: `PanelEstimationMixin`, `PanelAnalysisMixin`,
    `PanelValidationMixin`.
  - Note: `panel/estimation_mixin.py` is large (~1300 lines) and mixes
    `pfilter`, `mif`, `train`, `probe`, and synthetic-model code.

#### 2. Model component wrappers (`pypomp/core/model_struct.py`)
- **`_RInit`, `_RProc`, `_DMeas`, `_RMeas`**: Wrapper classes that provide a
  dictionary-based interface for users while using JAX arrays internally.
- User writes functions with dict arguments; wrappers handle array conversion via
  type-hint-based alignment.
- Features: time interpolation, covariate handling, parameter transformation.

#### 3. Parameter management
- **`PompParameters`** (`core/parameters.py`): Single-unit parameter sets as
  ordered dictionaries with attached `_logLik` array. Supports replicate
  collections (list-of-dicts).
- **`PanelParameters`** (`core/parameters.py`): Panel-level parameters with
  shared/unit-specific handling. Maintains `canonical_param_names` (all) and
  `canonical_shared_param_names` (shared only).
- **`ParTrans`** (`core/par_trans.py`): Parameter transformations between
  estimation and natural scales. `direction="to_est"` / `"from_est"` selects
  forward/inverse.
- **`RWSigma`** (`core/rw_sigma.py`): Random-walk sigma parameters for IF2,
  including per-parameter `init` sigmas applied only at iteration 0.

#### 4. Inference algorithms

There are **two algorithm layers**:

- **`pypomp/functional/`** — the public functional API. Pure JAX functions that
  take a `PompStruct`, a parameter array, and a key, and return arrays.
  Re-exported as `pypomp.functional`.
- **`pypomp/core/algorithms/`** — internal implementations behind the functional
  layer. The `_internal()` functions here are not part of the public API but are
  where the algorithm logic actually lives.

Algorithms:

- **`pfilter`**: Bootstrap particle filter with systematic resampling. Returns
  log-likelihood and optional diagnostics (CLL, ESS, filter mean, prediction
  mean). Implementation at `core/algorithms/pfilter.py`; public wrapper at
  `functional/pfilter.py`.
- **`mif`**: Iterated filtering (IF2 algorithm) via a perturbed particle filter.
  Uses geometric cooling. Implementation at `core/algorithms/mif.py`.
- **`train`**: Differentiable particle filter training with gradient descent.
  Supports Adam, SGD, FullMatrixAdam, BFGS, Newton, WeightedNewton (configured by
  passing an `Optimizer` instance from `core/optimizer.py`). Implementation at
  `core/algorithms/train.py` (note: ~1000 lines; optimizer step math lives here
  rather than in `core/optimizer.py`).
- **`mop`**: Mollified One-Particle gradient estimator
  (arXiv:2407.03085). Provides differentiable gradients for `train`.
  Implementation at `core/algorithms/mop.py`.
- **`dpop` / `dpop_train`**: Discounted POP gradient estimator and matching
  training loop. Accessed via `Pomp.dpop_train`. Note: `dpop_train` returns
  `(nll_hist, theta_hist)` directly rather than pushing onto `results_history`
  like other methods.
- **`simulate`**: Forward simulation of states and observations.

### JAX integration

JAX is deeply integrated throughout the codebase:

- **JIT compilation**: `@jax.jit` (often via `functools.partial`) on hot loops.
  Most files use `static_argnames=(...)`; `core/algorithms/pfilter.py:329` is an
  exception that uses positional `static_argnums` — when modifying, prefer
  `static_argnames`.
- **Vectorization**: `jax.vmap()` for parallelization across parameter
  replicates, panel units, and observation batches.
- **Multi-device sharding**: `Pomp.pfilter` and `Pomp.mif` shard `theta_reps`
  across devices when `len(jax.devices()) > 1`.
- **Random keys**: All stochastic functions take explicit JAX keys
  (`jax.random.key()`, `jax.random.split()`). The `fresh_key` attribute on
  `Pomp` is updated after every stochastic method call. `Pomp._update_fresh_key`
  raises if both the argument and `self.fresh_key` are `None`.
- **Differentiation**: Gradient computation for `train()`, `mop()`, and
  `dpop_train()`.
- **Memory efficiency**: `jax.checkpoint()` used inside the gradient paths.

**Important JAX Patterns:**
- Static arguments declared via `static_argnames` on `@jax.jit`.
- Sequential algorithms use `jax.lax.fori_loop()` / `jax.lax.scan()` for
  differentiability.
- All computational functions are pure (no side effects except key generation).
- Pickling: model functions are stored via `cloudpickle` so they don't require
  the original module on the loading side. JAX keys are stored as raw bytes and
  rewrapped with `jax.random.wrap_key_data` on load (see
  `Pomp.__getstate__`/`__setstate__`).

### Key design patterns

#### Parameter handling
1. Parameters stored as ordered dictionaries internally with fixed ordering
   (`canonical_param_names`).
2. Conversion chain: `dict` ↔ `PompParameters` ↔ JAX array.
3. Panel models separate `canonical_param_names` (all) from
   `canonical_shared_param_names` (shared only).
4. Several `PompParameters` methods (`transform`, `prune`) mutate in place — be
   aware when writing code that needs to keep the original.

#### Accumulator variables
- `accumvars`: state variables reset to 0 at each observation time.
- Specified as state names (e.g., `["cases", "deaths"]`); converted to indices
  internally and held as `_accumvars_indices`.

#### Results management
- Each inference method returns a `BaseResult` subclass via `results_history.add()`.
- Results stored in `ResultsHistory` with timestamps.
- Public methods on `Pomp` for accessing history: `traces()`, `results()`,
  `CLL()`, `ESS()`, `time()`, `print_summary()`.

#### Random key management
- Explicit JAX keys required for all stochastic operations.
- `fresh_key` is automatically updated after each method call.
- Deterministic seeding: `jax.random.key(seed)`.

### Example models

The package includes several example models for testing and demonstration:

- **`models/linear_gaussian.py`**: Linear Gaussian model (simple test case).
- **`models/dacca.py`**: Cholera data from Dacca, Bangladesh.
- **`models/spx.py`**: S&P 500 stochastic volatility.
- **`models/sir.py`**: SIR with seasonal forcing.
- **`models/ctmc_multinom.py`**: CTMC multinomial-step helper.
- **`models/measles/`**: UK measles epidemiological models — `uk_measles.py`
  is the main class; `model_001.py`, `model_001b.py`, `model_001c.py`,
  `model_001d.py`, `model_002.py`, `model_003.py` are variants.

### Random distributions (`pypomp/random/`)

JAX-compatible binomial, gamma, negative binomial, and Poisson samplers that
extend `jax.random` with distributions JAX doesn't ship. These are
**approximate** in some regimes (Giles & Beentjes 2024 for binomial; Temme 1992
for gamma; ported NVIDIA CURAND for Poisson) — see the docstrings in each file.

### Test structure

```
tests/
├── conftest.py                # Configures JAX persistent compilation cache
├── README                     # Informal testing notes
├── test_core.py
├── test_maths.py              # logmeanexp etc.
├── test_mcap.py
├── test_ParTrans.py
├── test_ParTrans_traces.py
├── test_RWSigma.py
├── test_model_struct.py
├── test_random/               # Modular JAX random distribution tests
│   ├── helpers.py             # Shared test utilities
│   ├── test_binomial.py
│   ├── test_gamma.py
│   ├── test_inverse.py        # Public inverse CDF checks
│   ├── test_multinomial.py
│   ├── test_nbinomial.py
│   └── test_poisson.py
├── test_ctmc_multinom.py
├── test_functional.py
├── test_benchmarks.py
├── test_pomp/                 # Single-unit Pomp tests
│   ├── test_pomp_class.py
│   ├── test_pomp_pfilter.py
│   ├── test_pomp_mif.py
│   ├── test_pomp_train.py
│   ├── test_pomp_train_dpop.py
│   ├── test_pomp_simulate.py
│   ├── test_pomp_probe.py
│   ├── test_pomp_analysis.py
│   ├── test_pomp_diagnostics.py
│   ├── test_pomp_mif_train_transform.py
│   └── test_pomp_result_equality.py
├── test_panel/                # Multi-unit PanelPomp tests
│   ├── conftest.py
│   ├── test_panelPomp_class.py
│   ├── test_panel_pfilter.py
│   ├── test_panel_mif.py
│   ├── test_panel_mif_transform.py
│   ├── test_panel_train.py
│   ├── test_panel_train_result_merge.py
│   ├── test_panel_simulate.py
│   ├── test_panel_probe.py
│   ├── test_panel_analysis.py
│   ├── test_panel_diagnostics.py
│   └── test_panel_parameters_to_jax_array.py
└── test_models/               # Integration tests for the example models
    ├── test_linear_gaussian.py
    ├── test_sir.py
    ├── test_dacca.py
    ├── test_spx.py
    ├── test_measles.py
    └── test_measles_001d.py
```

**Testing Conventions:**
- Use `conftest.py` fixtures for reusable test setups.
- Tests verify: numerical correctness, JAX JIT/vmap compatibility, parameter
  transformation round-trips, reproducibility with fixed keys.
- Tests are organized by component and integration level.
- `pytest.ini` enables xdist (`-n auto --dist=loadfile`) and durations reporting.

## Important Notes

### JAX-specific considerations

1. **Functions must be pure**: No side effects except random key generation.
2. **Static vs. dynamic arguments**: Functions decorated with `@jax.jit` must
   specify which arguments are static. Prefer `static_argnames` over
   `static_argnums` for refactor safety.
3. **Array operations**: Use `jax.numpy` instead of `numpy` in computational code.
4. **Random keys**: Always pass explicit keys; never use global random state.

### Parameter conventions

1. Parameters are always dictionaries internally with fixed ordering
   (`canonical_param_names`).
2. When modifying parameter-related code, maintain canonical ordering.
3. Panel models distinguish between shared and unit-specific parameters.
4. Use `ParTrans` for constrained optimization in unconstrained space; pass
   `direction="to_est"` to map natural → estimation and `"from_est"` for the
   inverse.

### Code modifications

When modifying inference algorithms or core functionality:
1. Ensure JAX compatibility (pure functions, no implicit state).
2. Add tests in the appropriate test subdirectory.
3. Verify JIT compilation works (no tracer leakage).
4. Check that `fresh_key` is properly updated.
5. Maintain backward compatibility where possible (package is in active
   development; pre-1.0).
6. When touching an algorithm, update **both** the implementation in
   `core/algorithms/` and the public wrapper in `functional/` (this duplication
   is a known wart — see `review.md`).

### Performance considerations

- JAX provides GPU support and significant speedup vs. R pomp.
- First call to JIT-compiled functions will be slow (compilation time);
  `tests/conftest.py` enables a persistent JAX compilation cache to amortize.
- Subsequent calls with same shapes/types will be fast.
- Use `jax.vmap()` for batched operations rather than Python loops.
- Multi-device sharding kicks in automatically in `Pomp.pfilter` / `Pomp.mif`
  when more than one device is available.
- Consider memory usage with large particle counts or long time series.
