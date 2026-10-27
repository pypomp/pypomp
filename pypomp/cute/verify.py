"""Numerical checks run once before the CuTe backend is chosen for a model.

All run the model's first observation interval on ``_CHECK_J`` particles with a key
of the kernel's generator (threefry, or pypomp's Philox), so XLA and the kernel draw
the same random numbers. The coupling probe
perturbs one particle of a ``@vectorized`` rproc and requires the others to stay
bit-identical on XLA: the kernel runs every particle on its own, so any coupling
would be dropped. The comparisons then require the kernel to match XLA with one
parameter vector on the natural scale (as in filtering) and with per-particle
parameters on the estimation scale (as in iterated filtering). The gradients of the
latter, which run the parameter transform's and the step's VJPs, are checked
separately, since a kernel whose gradients differ can still run undifferentiated.
A kernel that also evaluates the measurement density must match XLA's density at
the states it ends the interval in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

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
# A gradient that cancels large terms in float32 is mostly rounding, which XLA and
# the kernel round differently. Its size is measured as XLA's own change at inputs
# nudged by _NUDGE, and differences up to _NOISE_FACTOR times it are allowed.
_NUDGE = 1e-6
_NOISE_FACTOR = 4.0


class _Interval(NamedTuple):
    """The test interval: starting particles, parameters and a runner."""

    X0: jax.Array
    th: jax.Array
    """One parameter vector on the natural scale."""
    th_per: jax.Array
    """Per-particle parameters on the estimation scale, (J, d)."""
    run: Callable[[Callable, jax.Array, jax.Array, bool], jax.Array]
    """run(rproc, X, theta, should_trans) advances X over the interval."""
    rest: tuple
    """The remaining ``rprocess_interp`` arguments after X and theta, up to
    ``should_trans``."""


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

    return _Interval(X0, th, th_per, run, (key, cov, dts, model.t0, 0, nstep, acc))


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
    return states(
        _forward(iv, rp.mechanics_per_interp, iv.th_per, True)(iv.X0),
        _forward(iv, cute_per, iv.th_per, True)(iv.X0),
        " with per-particle parameters on the estimation scale",
    )


def _check_grads(model: Pomp, cute_per: Callable, rng: str) -> str | None:
    """Return why the CuTe rproc's gradients, with per-particle parameters on the
    estimation scale, can't be used for `model`, or None if they can."""
    iv = _interval(model, rng)
    xla = model.rproc.mechanics_per_interp
    ga, gb = _grads(iv, xla), _grads(iv, cute_per)
    mag = np.where(np.isfinite(ga), np.abs(ga), 0.0)
    scale = np.abs(ga) + 1e-3 * mag.max(axis=0) + 1e-12

    def different(noise: np.ndarray | float) -> float | None:
        s = scale + _NOISE_FACTOR * noise / _GRAD_PARTICLE_RTOL
        return _different(ga, gb, s, _GRAD_PARTICLE_RTOL, _GRAD_MEDIAN_RTOL)

    d = different(0.0)
    if d is not None:
        # Only models that fail pay for measuring XLA's rounding.
        nudged = iv._replace(
            X0=iv.X0 * (1 + _NUDGE), th_per=iv.th_per + _NUDGE * jnp.abs(iv.th_per)
        )
        change = np.abs(_grads(nudged, xla) - ga)
        noise = np.nanquantile(change, 1 - _MAX_DIFFERENT, axis=0)
        d = different(np.where(np.isfinite(noise), noise, 0.0))
    if d is not None:
        return (
            f"{d:.0%} of particles' gradients differ from XLA on a test interval with "
            "per-particle parameters on the estimation scale"
        )
    return None


def _check_dmeas(model: Pomp, cute_pf: Any, cute_per: Any, rng: str) -> str | None:
    """Return why the kernels' fused measurement density can't be used, or None.

    Each particle's density from the kernel must match XLA's density at the state the
    kernel ends the test interval in.
    """
    iv = _interval(model, rng)
    y = jnp.asarray(np.asarray(model.ys, dtype=np.float32)[0])
    t_obs = jnp.float32(model.ys.index[0])
    cov = iv.rest[1]
    cases = (
        (cute_pf, iv.th, False, ""),
        (cute_per, iv.th_per, True, " with per-particle parameters"),
    )
    for rproc, theta, trans, what in cases:

        def run(X, rproc=rproc, theta=theta, trans=trans):
            out, t_end, w = rproc.with_dmeasure(X, theta, *iv.rest, trans, y, t_obs)
            cov_t = None if cov is None else cov[t_end]
            return w, rproc.dmeasure(y, out, theta, cov_t, t_obs, trans)

        w, ref = (np.asarray(a)[:, None] for a in jax.jit(run)(iv.X0))
        d = _different(ref, w, np.abs(ref) + 1.0, _PARTICLE_RTOL, _MEDIAN_RTOL)
        if d is not None:
            return f"{d:.0%} of particles' measurement densities differ from XLA{what}"
    return None


def _check_perturb_step(model: Pomp, cute_per: Any, rng: str) -> str | None:
    """Return why the iterated filtering kernel can't be used, or None.

    From ancestors that repeat and reorder the test particles, its parameter draws must
    equal XLA's, its states must match XLA's rproc from the same particles and
    parameters, and its densities XLA's density at its states.
    """
    from .kernel import _perturbation

    iv = _interval(model, rng)
    J, d = iv.th_per.shape
    y = jnp.asarray(np.asarray(model.ys, dtype=np.float32)[0])
    t_obs = jnp.float32(model.ys.index[0])
    cov = iv.rest[1]
    anc = jax.random.randint(jax.random.key(4), (J,), 0, J, jnp.int32)
    key_perturb = jax.random.key(
        3, impl="threefry2x32" if rng == "threefry" else PHILOX
    )
    sig = jnp.full(d, 0.05, jnp.float32)

    def run(X):
        out, th, t_end, w = cute_per.perturb_step(
            X, iv.th_per, anc, key_perturb, sig, *iv.rest, True, y, t_obs
        )
        th_ref = iv.th_per[anc] + sig * _perturbation(rng, key_perturb, J, d)
        out_ref = iv.run(model.rproc.mechanics_per_interp, X[anc], th_ref, True)
        cov_t = None if cov is None else cov[t_end]
        return (
            th,
            th_ref,
            out,
            out_ref,
            w,
            cute_per.dmeasure(y, out, th, cov_t, t_obs, True),
        )

    th, th_ref, out, out_ref, w, w_ref = (np.asarray(a) for a in jax.jit(run)(iv.X0))
    scale = np.abs(th_ref) + 1.0
    if _different(th_ref, th, scale, _PARTICLE_RTOL, _MEDIAN_RTOL) is not None:
        return "its parameter perturbations differ from XLA's"
    d_out = _different(
        out_ref, out, np.abs(out_ref) + 1.0, _PARTICLE_RTOL, _MEDIAN_RTOL
    )
    if d_out is not None:
        return f"{d_out:.0%} of particles differ from XLA when resampled on load"
    w, w_ref = w[:, None], w_ref[:, None]
    d_w = _different(w_ref, w, np.abs(w_ref) + 1.0, _PARTICLE_RTOL, _MEDIAN_RTOL)
    if d_w is not None:
        return f"{d_w:.0%} of measurement densities differ when resampled on load"
    return None
