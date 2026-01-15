# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pypomp** is a Python package for modeling and inference using Partially Observed Markov Process (POMP) models (also known as state-space models or hidden Markov models). It implements particle filtering, iterated filtering, gradient-based optimization, and other inference algorithms for highly nonlinear, non-Gaussian dynamical systems. The package leverages JAX for GPU support and just-in-time compilation, achieving significant performance improvements over the R **pomp** package.

## Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.10+)
python3.12 -m venv .venv
source .venv/bin/activate  # or `.venv/bin/activate` on Unix

# Install dependencies
pip install -r requirements.txt

# Do not overwrite an existing environment.
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
pytest test/test_pomp/test_pomp_pfilter.py

# Run specific test function
pytest test/test_pomp/test_pomp_pfilter.py::test_class_basic_default

# Run tests for a specific module (e.g., pomp, panel, models)
pytest test/test_pomp/
pytest test/test_panel/
pytest test/test_models/
```

### Linting
```bash
# Lint with ruff (configured in pyproject.toml)
ruff check .

# Type checking with pyright
pyright
```

### Building
```bash
# Install package in editable mode for development
pip install -e .
```

## Architecture Overview

### Core Components

#### 1. Model Classes
- **`Pomp`** (`pomp_class.py`): Single-unit POMP model
  - Encapsulates observation data (`ys`), parameters (`theta`), state names
  - Contains four model components: `RInit`, `RProc`, `DMeas`, `RMeas`
  - Provides inference methods: `simulate()`, `pfilter()`, `mif()`, `train()`, `mop()`
  - Maintains `results_history` and `fresh_key` for random number generation

- **`PanelPomp`** (`panelPomp/panelPomp_class.py`): Multi-unit POMP models
  - Manages multiple `Pomp` objects with shared and unit-specific parameters
  - Uses mixin pattern: `PanelEstimationMixin`, `PanelAnalysisMixin`, `PanelValidationMixin`
  - Supports panel-level inference with parameter sharing

#### 2. Model Component Wrappers (`model_struct.py`)
- **`RInit`, `RProc`, `DMeas`, `RMeas`**: Wrapper classes that provide dictionary-based interface for users while using JAX arrays internally
- User writes functions with dict arguments; wrappers handle array conversion
- Features: time interpolation, covariate handling, parameter transformation

#### 3. Parameter Management
- **`PompParameters`** (`parameters.py`): Single-unit parameter sets as ordered dictionaries
- **`PanelParameters`** (`parameters.py`): Panel-level parameters with shared/unit-specific handling
- **`ParTrans`** (`ParTrans_class.py`): Parameter transformations between estimation and natural scales (e.g., log for positive parameters)

#### 4. Inference Algorithms
- **`pfilter.py`**: Bootstrap particle filter with systematic resampling
  - Core function: `_pfilter_internal()`
  - Returns log-likelihood, optional diagnostics (ESS, filter mean, prediction mean)

- **`mif.py`**: Iterated filtering (IF2 algorithm) for parameter estimation
  - Core function: `_mif_internal()`
  - Uses geometric cooling schedule for parameter perturbations
  - Works with `RWSigma` for random walk sigma parameters

- **`train.py`**: Differentiable particle filter training with gradient descent
  - Core function: `_train_internal()`
  - Supports multiple optimizers (Adam, SGD variants), optional line search

- **`mop.py`**: Mollified One-Particle (MOP) gradient estimator
  - Core function: `_mop_internal()`
  - Provides gradient estimates for optimization

- **`simulate.py`**: Model simulation

### JAX Integration

JAX is deeply integrated throughout the codebase:

- **JIT compilation**: `@jax.jit` decorators on hot loops for performance
- **Vectorization**: `jax.vmap()` for parallelization across parameter replicates, panel units, and observation batches
- **Random number generation**: Explicit JAX random keys (`jax.random.key()`, `jax.random.split()`)
  - All stochastic functions take explicit keys
  - `fresh_key` attribute tracks generated keys between method calls
- **Differentiation**: Gradient computation for `train()` and `mop()` methods
- **Memory efficiency**: `jax.checkpoint()` for gradient computation

**Important JAX Patterns:**
- Static arguments declared via `static_argnames` parameter in `@jax.jit`
- Sequential algorithms use `jax.lax.fori_loop()` for differentiability
- All computational functions are pure (no side effects except key generation)

### Key Design Patterns

#### Parameter Handling
1. Parameters stored as ordered dictionaries internally
2. `canonical_param_names`: Fixed ordering maintained throughout execution
3. Conversion chain: `dict` ↔ `PompParameters` ↔ `JAX array`
4. Panel models separate `canonical_param_names` (all) from `canonical_shared_param_names` (shared only)

#### Accumulator Variables
- `accumvars`: Tracks specific state variables during integration to reduce memory
- Specified as parameter names (e.g., `["cases", "deaths"]`), converted to indices internally
- Only accumulate what's needed rather than full state history

#### Results Management
- Each inference method returns a `BaseResult` subclass
- Results stored in `ResultsHistory` with timestamps
- Methods return: result object + update `fresh_key`
- Results support: `to_dataframe()`, `print_summary()`, equality comparison

#### Random Key Management
- Explicit JAX keys required for all stochastic operations
- `fresh_key` attribute automatically updated after each method call
- Supports both single-key and multi-key (batch) scenarios
- Deterministic seeding: `jax.random.key(seed)`

### Example Models

The package includes several example models for testing and demonstration:

- **`LG.py`**: Linear Gaussian model (simple test case)
- **`dacca.py`**: Cholera data from Dacca, Bangladesh
- **`spx.py`**: Stock price example
- **`measles/`**: UK measles epidemiological models with various parameterizations
  - `measlesPomp.py`: Main measles model class
  - `model_001.py`, `model_001b.py`, `model_001c.py`, `model_002.py`: Different model variants

### Test Structure

```
test/
├── test_core.py                  # Foundation tests
├── test_util.py                  # Utility function tests
├── test_ParTrans.py             # Parameter transformation tests
├── test_RWSigma.py              # IF2 sigma parameter tests
├── test_model_struct.py         # Model component wrapping tests
├── test_pomp/                    # Single-unit Pomp tests
│   ├── test_pomp_pfilter.py
│   ├── test_pomp_mif.py
│   ├── test_pomp_train.py
│   └── test_pomp_mop.py
├── test_panel/                   # Multi-unit PanelPomp tests
│   ├── conftest.py              # Shared fixtures
│   ├── test_panel_pfilter.py
│   └── test_panel_mif.py
└── test_models/                  # Integration tests
    ├── test_measles.py
    ├── test_dacca.py
    └── test_spx.py
```

**Testing Conventions:**
- Use `conftest.py` fixtures for reusable test setups
- Tests verify: numerical correctness, JAX JIT/vmap compatibility, parameter transformation round-trips, reproducibility with fixed keys
- Tests are organized by component and integration level

## Important Notes

### JAX-Specific Considerations

1. **Functions must be pure**: No side effects except random key generation
2. **Static vs. dynamic arguments**: Functions decorated with `@jax.jit` must specify which arguments are static (don't change graph structure)
3. **Array operations**: Use `jax.numpy` instead of `numpy` in computational code
4. **Random keys**: Always pass explicit keys; never use global random state

### Parameter Conventions

1. Parameters are always dictionaries internally with fixed ordering (`canonical_param_names`)
2. When modifying parameter-related code, maintain canonical ordering
3. Panel models distinguish between shared and unit-specific parameters
4. Use `ParTrans` for constrained optimization in unconstrained space

### Code Modifications

When modifying inference algorithms or core functionality:
1. Ensure JAX compatibility (pure functions, no implicit state)
2. Add tests in appropriate test subdirectory
3. Verify JIT compilation works (no tracer leakage)
4. Check that `fresh_key` is properly updated
5. Maintain backward compatibility where possible (package is in active development)

### Performance Considerations

- JAX provides GPU support and significant speedup (up to 16x vs R pomp)
- First call to JIT-compiled functions will be slow (compilation time)
- Subsequent calls with same shapes/types will be fast
- Use `jax.vmap()` for batched operations rather than Python loops
- Consider memory usage with large particle counts or long time series
