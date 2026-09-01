"""Subprocess worker CLI script for CPU parallel-scaling benchmark tests.

Configures JAX host platform device count before importing JAX, then executes particle
filtering timing steps and outputs JSON results to stdout.
"""

import json
import os
import sys
import time


def _configure_devices(devices: int) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={devices}"
    )


def _build_pomp_worker(cfg: dict):
    import pypomp as pp

    model = pp.models.lg(T=cfg["T"])
    base = dict(model.theta[0])
    theta = pp.PompParameters(
        [
            {k: v * (1.0 + 1e-3 * i) for k, v in base.items()}
            for i in range(cfg["n_param_sets"])
        ]
    )
    return model, theta


def _build_panel_worker(cfg: dict):
    import pandas as pd

    import pypomp as pp

    units = {f"unit{i + 1}": pp.models.lg(T=cfg["T"]) for i in range(cfg["n_units"])}
    pnames = next(iter(units.values())).canonical_param_names
    base_df = pd.DataFrame(
        {name: [u.theta[0][p] for p in pnames] for name, u in units.items()},
        index=pd.Index(pnames),
    )
    theta = pp.PanelParameters(
        [
            {"shared": None, "unit_specific": base_df * (1.0 + 1e-3 * i)}
            for i in range(cfg["n_param_sets"])
        ]
    )
    model = pp.PanelPomp(pomp_dict=units, theta=theta)
    return model, theta


def _build_worker_model(cfg: dict):
    if cfg["model"] == "pomp":
        return _build_pomp_worker(cfg)
    return _build_panel_worker(cfg)


def _run_worker(cfg_str: str) -> None:
    """Configure environment, run warm-up and timed pfilter steps, and print JSON metrics."""
    cfg = json.loads(cfg_str)
    _configure_devices(cfg["devices"])

    import jax
    import numpy as np

    n_dev = len(jax.devices())
    if n_dev != cfg["devices"]:
        raise RuntimeError(f"asked for {cfg['devices']} devices, got {n_dev}")

    model, theta = _build_worker_model(cfg)

    def step():
        key = jax.random.key(cfg["seed"])
        t0 = time.perf_counter()
        model.pfilter(J=cfg["J"], reps=1, theta=theta, key=key)  # type: ignore[arg-type]
        log_lik = float(np.asarray(model.results_history[-1].logLiks).sum())
        return time.perf_counter() - t0, log_lik

    step()  # Warm-up (JIT)
    times, log_liks = zip(*[step() for _ in range(cfg["n_timed"])], strict=True)
    json.dump(
        {"devices": n_dev, "times": list(times), "logLik": log_liks[0]},
        sys.stdout,
    )


if __name__ == "__main__":
    _run_worker(sys.argv[1])
