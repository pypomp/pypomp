# Running Tests

Install the package in editable mode with the development and testing extras:

```bash
pip install -e ".[tests,benchmarks,viz]"
```

An editable install is required: without it `import pypomp` may resolve to a
stale site-packages copy, and pytest/coverage would silently exercise the wrong
files.

Run the tests from the repository root:

```bash
make test-light      # pytest -m "not heavy"  -- the fast inner loop
make test-heavy      # pytest -m "heavy"      -- accuracy and convergence checks
make test-all        # everything except the CPU-scaling checks
make test-cpu-scaling
```

`pytest.ini` sets `-n auto --dist=loadfile`, so each file is an xdist scheduling
unit and whole files are pinned to one worker.

## Markers

`heavy` marks tests that take a long time: statistical accuracy and convergence
checks, and tests that assert on wall-clock time and so need the machine to
themselves. It is applied at module level in `tests/test_accuracy.py`,
`tests/test_random/test_sampler_accuracy.py`, and
`tests/test_panel/test_panel_performance.py`.

The CPU parallel-scaling checks under `tests/test_cpu_parallel/` are gated
separately: they are not collected at all unless `PYPOMP_CPU_SCALING=1` is set,
because they time particle filters and need the cores to themselves. Use
`make test-cpu-scaling`.

## Layout

| Directory | Contents |
| --- | --- |
| `helpers/` | Shared builders and assertions (see below). Not collected. |
| `test_core/` | `pypomp/core/`: parameters, results, model_mechanics, transforms |
| `test_pomp/` | The single-unit `Pomp` class and its algorithms |
| `test_panel/` | The `PanelPomp` class and its algorithms |
| `test_models/` | The example models under `pypomp/models/` |
| `test_random/` | The JAX distribution samplers |
| `test_properties/` | Cross-cutting properties: invariants, and parity between the OO and functional layers |
| `test_regression/` | Locked numerical baselines (pytest-regressions) |
| `test_cpu_parallel/` | Wall-clock parallel-scaling checks (env-gated) |

## Shared helpers

`tests/helpers/` is an importable package, so use absolute imports:

```python
from tests.helpers.models import lg_panel, sir_panel
from tests.helpers.params import uniform_rw_sd, uniform_eta
from tests.helpers.assertions import pickle_roundtrip
```

Prefer these over rebuilding a model inline. Note that shared model *components*
belong in `tests/helpers/`, not in a `conftest.py`: pytest imports each
`conftest.py` under its own module name, so a test importing a function from one
gets a second, unequal copy of it — which breaks identity-based comparisons such
as `Pomp.merge`.

Files named with a leading underscore (`test_core/_shard_probe.py`,
`test_cpu_parallel/_scaling_worker.py`) are not tests. They are entry points run
as subprocesses by the test beside them, for cases where a setting must be fixed
before JAX is imported.

## Regression baselines

The existing `test_pomp/test_pomp_pickle.py` also covers the `bake`, `stew` and
`freeze` computation archives. Run it without R or any local report files:

```bash
python -m pytest tests/test_pomp/test_pomp_pickle.py -o addopts= -q
```

The R compatibility cases use expected observations from an executed pomp
6.4.0.4 reference, commit `7cfb3f9aa84c85de687b82d71b081016b9cc5762`, on R 4.5.3.
The `U` case IDs translate the predicates in that version's
[`tests/bake.R`](https://github.com/kingaa/pomp/blob/7cfb3f9aa84c85de687b82d71b081016b9cc5762/tests/bake.R).
The other case IDs cover archive boundaries, failure behavior, stress cases and
the documented compatibility adapter. Expected observations were checked against
both the stated contract and live R before being stored here. Long uniform
vectors are compared through SHA-256 of their integer words, serialized as a
compact JSON array; their length and integer bounds are checked separately.

These are portable Python regressions against fixed R observations. They do not
execute R at test time or assert compatibility with every R version, RDS/RData
file decoding, or equal R/JAX model trajectories. Simulation predicates check
reproducibility within each runtime. See the computation archives API page for
the native defaults and the explicit `compatibility="R"` behavior.

Additional tests cover Python validation, malformed pickle bindings and RNG
cleanup. These are separate from exact R comparisons: invalid uniform bounds
raise `ValueError` in Python, while R `runif` returns `NaN` with a warning.
The pypomp archive API rejects RNG kinds outside Mersenne-Twister/Inversion, including
Wichmann-Hill and Box-Muller, which R supports. Equal-bound uniform values and
unchanged RNG state are checked against executed R observations; initialization
without a seed is compared through bounds and state replay, not identical draws.

`test_regression/` compares against CSV baselines stored in a directory named
after each test file. When an algorithm legitimately changes, regenerate with
`--force-regen` and review the diff. Renaming a regression test file means
renaming its baseline directory too.

## Coverage

```bash
pytest --cov
```
