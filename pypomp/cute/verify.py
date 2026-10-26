"""Numerical checks run once before the CuTe backend is chosen for a model.

All run the model's first observation interval on ``_CHECK_J`` particles with a key
of the kernel's generator (threefry, or pypomp's Philox), so XLA and the kernel draw
the same random numbers. The coupling probe
perturbs one particle of a ``@vectorized`` rproc and requires the others to stay
bit-identical on XLA: the kernel runs every particle on its own, so any coupling
would be dropped. The comparisons then require the kernel to match XLA with one
parameter vector on the natural scale (as in filtering), with per-particle
parameters on the estimation scale (as in iterated filtering), and in the
gradients of the latter, which run the parameter transform's and the step's VJPs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

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
# Gradients sum more rounding, so are compared more loosely.
_GRAD_PARTICLE_RTOL = 1e-2
_GRAD_MEDIAN_RTOL = 1e-3


class _Interval(NamedTuple):
    """The test interval: starting particles, parameters and a runner."""

    X0: jax.Array
    th: jax.Array
    """One parameter vector on the natural scale."""
    th_per: jax.Array
    """Per-particle parameters on the estimation scale, (J, d)."""
    run: Callable[[Callable, jax.Array, jax.Array, bool], jax.Array]
    """run(rproc, X, theta, should_trans) advances X over the interval."""


def _interval(model: Pomp, rng: str) -> _Interval:
    """Return the test interval for `model`, which must have parameters, with a key
    of generator `rng`."""
    names = model.canonical_param_names
    th = model.theta.to_jax_array(names)[0]
    th_est = model.par_trans._transform_array(th[None], names, direction="to_est")[0]
    th_per = th_est + 0.01 * jax.random.normal(
        jax.random.key(2), (_CHECK_J, th.size), th.dtype
    )
    covars = model._covars_extended
    cov = None if covars is None else jnp.asarray(covars)
    keys = jax.random.split(jax.random.key(0), _CHECK_J)
    X0 = model.rinit.mechanics_pf(
        th, keys, None if cov is None else cov[0], model.t0, False
    )
    key = jax.random.key(1, impl="threefry2x32" if rng == "threefry" else PHILOX)
    dts = jnp.asarray(model._dt_array_extended)
    nstep, acc = int(model._nstep_array[0]), model.rproc.accumvars

    def run(
        rproc: Callable, X: jax.Array, theta: jax.Array, should_trans: bool
    ) -> jax.Array:
        return rproc(X, theta, key, cov, dts, model.t0, 0, nstep, acc, should_trans)[0]

    return _Interval(X0, th, th_per, run)


def _forward(iv: _Interval, rproc: Callable, theta: jax.Array, trans: bool) -> Callable:
    """Return a compiled function from starting particles to the interval's end."""
    jitted = jax.jit(lambda X: iv.run(rproc, X, theta, trans))
    return lambda X: np.asarray(jitted(X))


def _grads(iv: _Interval, rproc_per: Callable) -> np.ndarray:
    """Return each particle's state and parameter gradients, (J, ns + d), of a
    weighted sum of the end states, with per-particle estimation-scale parameters."""
    w = jnp.cos(0.1 * jnp.arange(iv.X0.size, dtype=iv.X0.dtype)).reshape(iv.X0.shape)

    def loss(X: jax.Array, theta: jax.Array) -> jax.Array:
        return jnp.sum(iv.run(rproc_per, X, theta, True) * w)

    gx, gth = jax.jit(jax.grad(loss, argnums=(0, 1)))(iv.X0, iv.th_per)
    return np.concatenate([np.asarray(gx), np.asarray(gth)], axis=1)


def _different(
    a: np.ndarray,
    b: np.ndarray,
    scale: np.ndarray,
    particle_rtol: float,
    median_rtol: float,
) -> float | None:
    """Return the fraction of particles (rows) where `b` differs from `a` if too many
    do or the median error is too large, otherwise None."""
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    rel = np.where(same, 0.0, np.abs(a - b) / scale)
    rel = np.nan_to_num(rel, nan=np.inf)
    different = float((rel.max(-1) > particle_rtol).mean())
    if different > _MAX_DIFFERENT or np.median(rel) > median_rtol:
        return different
    return None


def _check(
    model: Pomp, cute_pf: Callable, cute_per: Callable, vectorized: bool, rng: str
) -> str | None:
    """Return why the CuTe rproc, drawing with `rng`, can't be used for `model`, or
    None if it can."""
    iv = _interval(model, rng)
    rp = model.rproc
    xla = _forward(iv, rp.mechanics_pf_interp, iv.th, False)
    a = xla(iv.X0)
    if vectorized:
        for k in (0, _CHECK_J // 2):
            moved = xla(iv.X0.at[k].set(iv.X0[k] * 1.25 + 1.0))
            others = np.arange(_CHECK_J) != k
            if not np.array_equal(a[others], moved[others], equal_nan=True):
                return "changing one particle's state changes other particles"

    def states(a: np.ndarray, b: np.ndarray, what: str) -> str | None:
        d = _different(a, b, np.abs(a) + 1.0, _PARTICLE_RTOL, _MEDIAN_RTOL)
        if d is None:
            return None
        return f"{d:.0%} of particles differ from XLA on a test interval{what}"

    why = states(a, _forward(iv, cute_pf, iv.th, False)(iv.X0), "")
    if why is not None:
        return why
    per = " with per-particle parameters on the estimation scale"
    why = states(
        _forward(iv, rp.mechanics_per_interp, iv.th_per, True)(iv.X0),
        _forward(iv, cute_per, iv.th_per, True)(iv.X0),
        per,
    )
    if why is not None:
        return why
    ga, gb = _grads(iv, rp.mechanics_per_interp), _grads(iv, cute_per)
    mag = np.where(np.isfinite(ga), np.abs(ga), 0.0)
    d = _different(
        ga,
        gb,
        np.abs(ga) + 1e-3 * mag.max(axis=0) + 1e-12,
        _GRAD_PARTICLE_RTOL,
        _GRAD_MEDIAN_RTOL,
    )
    if d is not None:
        return (
            f"{d:.0%} of particles' gradients differ from XLA on a test interval{per}"
        )
    return None
