"""Does pypomp's CPU parallelisation actually make things faster?

`Pomp.pfilter` and `PanelPomp.pfilter` shard parameter-set replicates across
`jax.devices()` (see `pypomp.core.algorithms.helpers.run_jax_batch_sharded` and
the "Optimizing CPU Performance" section of docs/source/best_practices.rst).  On
CPU the device count is a knob, set with
`--xla_force_host_platform_device_count` before JAX is imported, and the
documented advice is to set it to the number of cores.

Each test below holds the total amount of work fixed -- one particle filter per
parameter set, for as many parameter sets as there are devices -- and measures it
twice, in two fresh subprocesses.  The workload is built from parameter sets
rather than from the `reps` argument deliberately: the parameter-set axis is the
one `run_jax_batch_sharded` partitions across devices, whereas `reps` is vmapped
underneath the sharding and so does not scale with the device count.  Two
different knobs are varied, because they are not the same thing:

``cores``
    Serial baseline (1 core, 1 device) against the full documented
    configuration (N cores, N devices).  This is the real "does CPU
    parallelisation work" check: with the work fixed, N cores should finish it
    close to N times faster.  Restricting cores needs `os.sched_setaffinity`,
    so this variant only runs on Linux (which includes GitHub's runners).

``devices``
    All cores in both runs, 1 device against N devices.  XLA backs every forced
    host device with the *same* threadpool ("All of these host devices are
    backed by the same threadpool", per `--xla_force_host_platform_device_count`
    in `XLA_FLAGS=--help`), so raising the device count does not add cores; it
    changes how the replicates are partitioned across the threadpool.  Whether
    that partitioning wins depends on the model, so this variant only guards
    against the sharded path becoming pathologically slower than the unsharded
    one, and reports the measured ratio.

Both variants also check that the two runs agree on the log-likelihood, which
catches a sharded path that parallelises by quietly computing something else.

These are wall-clock measurements, so they are only collected when
`PYPOMP_CPU_SCALING=1` (see conftest.py) and are marked `heavy`.  Run them on an
otherwise idle machine, without xdist:

    make test-cpu-scaling
"""

import json
import math
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.heavy

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scaling_worker.py")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cap the parallel width so the runtime stays bounded on many-core machines.
MAX_WIDTH = 8

# Minimum parallel efficiency (speedup / width) the `cores` variant requires.
# Perfectly linear scaling is 1.0; the default lets a run spend half its time on
# overhead and still pass, which keeps the check meaningful without making it
# hostage to a shared CI runner's noise. Override with
# PYPOMP_CPU_SCALING_MIN_EFFICIENCY.
DEFAULT_MIN_EFFICIENCY = 0.5

# How much slower than the unsharded run the sharded run may be in the `devices`
# variant. Sharding is a partitioning choice rather than extra cores, so this is
# a regression guard, not a scaling claim. Override with
# PYPOMP_CPU_SCALING_MAX_SHARDING_OVERHEAD.
DEFAULT_MAX_SHARDING_OVERHEAD = 2.0

# Timed runs per configuration, after a warm-up run that absorbs JIT
# compilation. The minimum is used: interference from other processes can only
# make a run slower.
N_TIMED = 3

SUBPROCESS_TIMEOUT_S = 900


def _width() -> int:
    """Number of cores/devices to parallelise over."""
    return min(MAX_WIDTH, os.cpu_count() or 1)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _check_environment(mechanism: str) -> None:
    """Skip unless this machine can produce a meaningful timing."""
    workers = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if workers is not None and int(workers) > 1:
        pytest.skip(
            f"running under xdist with {workers} workers, so wall-clock timings "
            "are noise. Re-run with -p no:xdist (see `make test-cpu-scaling`)."
        )
    if _width() < 2:
        pytest.skip("need at least 2 CPU cores to measure parallel scaling")
    if mechanism == "cores" and not hasattr(os, "sched_setaffinity"):
        pytest.skip(
            "os.sched_setaffinity is unavailable (non-Linux), so the number of "
            "cores XLA uses cannot be restricted to build a serial baseline"
        )


def _measure(**cfg) -> dict:
    """Run one configuration in its own process and return its report."""
    proc = subprocess.run(
        [sys.executable, WORKER, json.dumps(cfg)],
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
    cores = "all cores" if run["cores"] is None else f"{run['cores']} core(s)"
    return (
        f"{cores}/{run['devices']} device(s): {min(run['times']):.3f}s "
        f"(runs: {[round(t, 3) for t in run['times']]})"
    )


def _run_pair(model: str, mechanism: str, workload: dict) -> tuple[dict, dict, int]:
    """Measure `workload` serially and in parallel, `width` replicates of work.

    The work is `width` distinct parameter sets rather than `width` `reps` of a
    single one: `run_jax_batch_sharded` partitions the parameter-set axis across
    devices, while `reps` is vmapped underneath it and does not scale with the
    device count.
    """
    width = _width()
    common = dict(model=model, n_param_sets=width, n_timed=N_TIMED, seed=1, **workload)
    if mechanism == "cores":
        serial = _measure(cores=1, devices=1, **common)
        parallel = _measure(cores=width, devices=width, **common)
    else:
        serial = _measure(cores=None, devices=1, **common)
        parallel = _measure(cores=None, devices=width, **common)
    return serial, parallel, width


def _assert_scales(
    label: str, mechanism: str, serial: dict, parallel: dict, width: int
) -> None:
    speedup = min(serial["times"]) / min(parallel["times"])
    report = (
        f"{label} [{mechanism}]: {width} parameter sets, "
        f"serial {_describe(serial)} vs parallel {_describe(parallel)} "
        f"-> speedup {speedup:.2f}x"
    )
    print(report)

    # Identical work in both runs, so the log-likelihoods must agree. The
    # tolerance allows XLA to reassociate floating-point arithmetic differently
    # in the two compiled programs.
    assert math.isclose(serial["logLik"], parallel["logLik"], rel_tol=1e-2), (
        f"the two runs disagree on the log-likelihood: {serial['logLik']} vs "
        f"{parallel['logLik']}, so they did not measure the same work. {report}"
    )

    if mechanism == "cores":
        min_efficiency = _env_float(
            "PYPOMP_CPU_SCALING_MIN_EFFICIENCY", DEFAULT_MIN_EFFICIENCY
        )
        assert speedup >= min_efficiency * width, (
            f"{width} cores did not speed the filter up nearly linearly: "
            f"efficiency {speedup / width:.2f}, need {min_efficiency:.2f}. {report}"
        )
    else:
        max_overhead = _env_float(
            "PYPOMP_CPU_SCALING_MAX_SHARDING_OVERHEAD", DEFAULT_MAX_SHARDING_OVERHEAD
        )
        assert speedup >= 1.0 / max_overhead, (
            f"sharding across {width} devices was {1 / speedup:.2f}x slower than "
            f"a single device, more than the {max_overhead:.2f}x allowed. {report}"
        )


@pytest.mark.parametrize("mechanism", ["cores", "devices"])
def test_pomp_pfilter_cpu_parallel(mechanism: str):
    """`Pomp.pfilter` replicates should use the available CPU cores."""
    _check_environment(mechanism)
    serial, parallel, width = _run_pair("pomp", mechanism, {"J": 3000, "T": 400})
    _assert_scales("Pomp.pfilter", mechanism, serial, parallel, width)


@pytest.mark.parametrize("mechanism", ["cores", "devices"])
def test_panel_pfilter_cpu_parallel(mechanism: str):
    """`PanelPomp.pfilter` replicates should use the available CPU cores."""
    _check_environment(mechanism)
    serial, parallel, width = _run_pair(
        "panel", mechanism, {"J": 2000, "T": 200, "n_units": 2}
    )
    _assert_scales("PanelPomp.pfilter", mechanism, serial, parallel, width)
