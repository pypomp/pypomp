"""Does pypomp's CPU parallelisation actually make things faster?

`Pomp.pfilter` and `PanelPomp.pfilter` shard parameter-set replicates across
`jax.devices()` (see `run_jax_batch_sharded`). On CPU, the device count is set with
`--xla_force_host_platform_device_count` before JAX is imported.

Each test measures a fixed workload serially vs. in parallel using worker subprocesses.

Wall-clock measurements run only when `PYPOMP_CPU_SCALING=1` (see conftest.py).
Run via: `make test-cpu-scaling`
"""

import json
import math
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_scaling_worker.py"
)
MAX_WIDTH = 8
DEFAULT_MAX_SHARDING_OVERHEAD = 2.0
N_TIMED = 3
SUBPROCESS_TIMEOUT_S = 900


def _width() -> int:
    """Return CPU core count capped by MAX_WIDTH for benchmark execution."""
    return min(MAX_WIDTH, os.cpu_count() or 1)


def _check_environment() -> None:
    """Skip test if running under xdist or lacking multi-core hardware."""
    workers = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if workers is not None and int(workers) > 1:
        pytest.skip(
            f"running under xdist with {workers} workers, wall-clock timings are noise."
        )
    if _width() < 2:
        pytest.skip("need at least 2 CPU cores to measure parallel scaling")


def _measure(**cfg) -> dict:
    """Run worker subprocess with the given benchmark configuration and return JSON results."""
    proc = subprocess.run(
        [sys.executable, WORKER_SCRIPT, json.dumps(cfg)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"scaling worker failed for {cfg}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def _describe(run: dict) -> str:
    """Format worker benchmark timing metrics into a readable string."""
    return (
        f"{run['devices']} device(s): {min(run['times']):.3f}s "
        f"(runs: {[round(t, 3) for t in run['times']]})"
    )


def _assert_scaling(model: str, workload: dict) -> None:
    """Assert log-likelihood consistency and required speedup between serial and parallel runs."""
    width = _width()
    common = dict(model=model, n_param_sets=width, n_timed=N_TIMED, seed=1, **workload)
    serial = _measure(devices=1, **common)
    parallel = _measure(devices=width, **common)

    speedup = min(serial["times"]) / min(parallel["times"])
    per_core = speedup / width
    efficiency = per_core * 100.0
    report = (
        f"{model}: {width} parameter sets across {width} cores, "
        f"serial {_describe(serial)} vs parallel {_describe(parallel)} "
        f"-> speedup {speedup:.2f}x ({per_core:.2f}x/core, {efficiency:.1f}% efficiency)"
    )
    print(report)

    assert math.isclose(serial["logLik"], parallel["logLik"], rel_tol=1e-2), (
        f"log-likelihood mismatch: {serial['logLik']} vs {parallel['logLik']}. {report}"
    )

    assert speedup >= 1.0 / DEFAULT_MAX_SHARDING_OVERHEAD, (
        f"sharding speedup {speedup:.2f}x < minimum required {1.0 / DEFAULT_MAX_SHARDING_OVERHEAD:.2f}x. {report}"
    )


@pytest.mark.parametrize(
    "model,workload",
    [
        ("pomp", {"J": 3000, "T": 400}),
        ("panel", {"J": 2000, "T": 200, "n_units": 2}),
    ],
)
def test_cpu_parallel_scaling(model: str, workload: dict):
    """Verify that particle filtering scales with available CPU devices."""
    _check_environment()
    _assert_scaling(model, workload)
