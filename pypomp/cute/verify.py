"""Numerical checks run once before the CuTe backend is chosen for a model.

Both run the model's first observation interval on ``_CHECK_J`` particles with a
Philox key, so XLA and the kernel draw the same random numbers. The coupling probe
perturbs one particle of a ``@vectorized`` rproc and requires the others to stay
bit-identical on XLA: the kernel runs every particle on its own, so any coupling
would be dropped. The comparison then runs the kernel on the same interval and
requires it to match XLA.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ..random.philox import PHILOX

if TYPE_CHECKING:
    from ..core.pomp import Pomp

_CHECK_J = 256
# A particle counts as different from XLA past this relative error. Discrete draws
# can land on the other side of a CDF step from float rounding, so a few may.
_PARTICLE_RTOL = 1e-3
_MAX_DIFFERENT = 0.02
# The median relative error across particles may not exceed this.
_MEDIAN_RTOL = 1e-4


def _interval(model: Pomp) -> tuple[jax.Array, Callable] | None:
    """Return starting particles and a function that runs an rproc_pf from them
    over the first observation interval, or None if the model has no parameters."""
    if model._theta is None:
        return None
    th = model.theta.to_jax_array(model.canonical_param_names)[0]
    covars = model._covars_extended
    cov = None if covars is None else jnp.asarray(covars)
    keys = jax.random.split(jax.random.key(0), _CHECK_J)
    X0 = model.rinit.mechanics_pf(
        th, keys, None if cov is None else cov[0], model.t0, False
    )
    key = jax.random.key(1, impl=PHILOX)
    dts = jnp.asarray(model._dt_array_extended)
    nstep, acc = int(model._nstep_array[0]), model.rproc.accumvars

    def compile_run(rproc_pf: Callable) -> Callable:
        def run(X: jax.Array) -> jax.Array:
            return rproc_pf(X, th, key, cov, dts, model.t0, 0, nstep, acc, False)[0]

        jitted = jax.jit(run)
        return lambda X: np.asarray(jitted(X))

    return X0, compile_run


def _check(model: Pomp, cute_pf: Callable, vectorized: bool) -> str | None:
    """Return why the CuTe rproc can't be used for `model`, or None if it can."""
    interval = _interval(model)
    if interval is None:
        return None
    X0, compile_run = interval
    xla = compile_run(model.rproc.mechanics_pf_interp)
    a = xla(X0)
    if vectorized:
        for k in (0, _CHECK_J // 2):
            moved = xla(X0.at[k].set(X0[k] * 1.25 + 1.0))
            others = np.arange(_CHECK_J) != k
            if not np.array_equal(a[others], moved[others], equal_nan=True):
                return "changing one particle's state changes other particles"
    b = compile_run(cute_pf)(X0)
    both_nan = np.isnan(a) & np.isnan(b)
    rel = np.where(both_nan, 0.0, np.abs(a - b) / (np.abs(a) + 1.0))
    rel = np.nan_to_num(rel, nan=np.inf)
    different = (rel.max(-1) > _PARTICLE_RTOL).mean()
    if different > _MAX_DIFFERENT or np.median(rel) > _MEDIAN_RTOL:
        return f"{different:.0%} of particles differ from XLA on a test interval"
    return None
