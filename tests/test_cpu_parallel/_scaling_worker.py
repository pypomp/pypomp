"""Time a `pfilter` call under a fixed CPU configuration.

Run as a script, one process per measurement::

    python _scaling_worker.py '{"model": "pomp", "devices": 4, "cores": 4, ...}'

A single process cannot measure more than one configuration.  Both knobs that
matter here are frozen at the moment JAX is imported:

* the JAX device count, set with `--xla_force_host_platform_device_count`
  (the knob documented in docs/source/best_practices.rst); and
* the number of cores XLA's CPU threadpool will use, which it derives from the
  process's CPU affinity mask.

The harness in test_cpu_parallel_scaling.py therefore spawns one of these per
configuration and compares the reported times.

Results are written to stdout as a JSON object::

    {"devices": 4, "cores": 4, "times": [...], "logLik": -1234.5}

`times` excludes JIT compilation: the first run is a warm-up and is not
reported.  `logLik` lets the caller check that the configurations being compared
computed the same thing.
"""

import json
import os
import sys
import time
from typing import Any


def _restrict_cores(cores: int) -> int:
    """Pin this process to `cores` CPUs and return the resulting core count.

    Must run before importing JAX: XLA's CPU threadpool is sized from the
    affinity mask when the backend is created.  `os.sched_setaffinity` is
    Linux-only, which is why the core-scaling measurement is skipped elsewhere.
    """
    # Accessed via getattr so that this type-checks on platforms (macOS,
    # Windows) whose `os` module has no affinity functions.
    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    if get_affinity is None or set_affinity is None:
        raise RuntimeError("CPU affinity is not available on this platform")
    available = sorted(get_affinity(0))
    if cores > len(available):
        raise RuntimeError(f"asked for {cores} cores but only {len(available)} usable")
    set_affinity(0, set(available[:cores]))
    return len(get_affinity(0))


def _configure_devices(devices: int) -> None:
    """Force JAX onto `devices` CPU devices.  Must run before importing JAX."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={devices}"
    )


def _build_pomp(cfg: dict[str, Any]):
    import pypomp as pp

    return pp.models.LG(T=cfg["T"])


def _build_panel(cfg: dict[str, Any]):
    import pandas as pd

    import pypomp as pp

    units = {f"unit{i + 1}": pp.models.LG(T=cfg["T"]) for i in range(cfg["n_units"])}
    param_names = next(iter(units.values())).canonical_param_names
    unit_specific = pd.DataFrame(
        {name: [unit.theta[0][p] for p in param_names] for name, unit in units.items()},
        index=pd.Index(param_names),
    )
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])
    return pp.PanelPomp(Pomp_dict=units, theta=theta)


def main(argv: list[str]) -> int:
    cfg = json.loads(argv[1])

    cores = _restrict_cores(cfg["cores"]) if cfg.get("cores") is not None else None
    _configure_devices(cfg["devices"])

    import jax
    import numpy as np

    n_devices = len(jax.devices())
    if n_devices != cfg["devices"]:
        raise RuntimeError(
            f"asked for {cfg['devices']} CPU devices but JAX reports {n_devices}"
        )

    model = _build_pomp(cfg) if cfg["model"] == "pomp" else _build_panel(cfg)

    def timed_run() -> tuple[float, float]:
        key = jax.random.key(cfg["seed"])
        start = time.perf_counter()
        model.pfilter(J=cfg["J"], reps=cfg["reps"], key=key)
        # Results are stored as xarray objects, which already forces the
        # computation to complete, but reduce them explicitly so the timing
        # cannot be fooled by JAX's asynchronous dispatch if that changes.
        logLik = float(np.asarray(model.results_history[-1].logLiks).sum())
        return time.perf_counter() - start, logLik

    timed_run()  # warm-up: pays the JIT compilation cost outside the timings
    times: list[float] = []
    logLiks: list[float] = []
    for _ in range(cfg["n_timed"]):
        elapsed, logLik = timed_run()
        times.append(elapsed)
        logLiks.append(logLik)

    json.dump(
        {
            "devices": n_devices,
            "cores": cores,
            "times": times,
            "logLik": logLiks[0],
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
