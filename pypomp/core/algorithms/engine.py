"""How a filter advances its particles over one observation interval.

The particle filters (``pfilter``, and the perturbed filter of iterated filtering)
leave every step that depends on how the model runs to an :class:`IntervalEngine`:
propagating the particles, evaluating the measurement density, perturbing the
parameters and computing resampling indices. :class:`XlaEngine` does these with the
model's own JAX functions; a backend such as CuTe supplies its own engine for its
process model (see :func:`engine_for`), so the filters need not know which runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from .carries import Swarm
from .helpers import _resample, _resample_where


class Interval(NamedTuple):
    """One observation interval's arguments to ``rprocess_interp``, in its order
    after the particles and parameters, so ``rprocess_interp(X, theta, *iv)``."""

    key: jax.Array
    covars_extended: jax.Array | None
    dt_array_extended: jax.Array
    t: Any
    t_idx: Any
    nstep: Any
    accumvars: tuple[int, ...] | None
    should_trans: bool


class IntervalEngine(Protocol):
    """Advances a particle filter over one observation interval.

    `X` holds one row of states per particle and `theta` the parameters, one set
    for all particles (``pfilter``) or one row per particle (iterated filtering).
    """

    def advance(
        self, X: jax.Array, theta: jax.Array, iv: Interval, y: jax.Array, t_obs: Any
    ) -> tuple[jax.Array, Any, jax.Array]:
        """Propagate `X` over `iv` and return the states, the time index at the end
        of the interval and each particle's log-density of observation `y` at
        time `t_obs`."""
        ...

    def resample(self, norm_weights: jax.Array, key: jax.Array) -> jax.Array:
        """Return systematic resampling indices for `norm_weights`, as
        :func:`~pypomp.core.algorithms.helpers._resample` does."""
        ...

    def swarm(self, X: jax.Array, thetas: jax.Array) -> Swarm:
        """Return the carry of iterated filtering for particles `X` and their
        parameters `thetas`."""
        ...

    def advance_perturbed(
        self,
        swarm: Swarm,
        key_perturb: jax.Array,
        sigmas: jax.Array,
        iv: Interval,
        y: jax.Array,
        t_obs: Any,
    ) -> tuple[jax.Array, jax.Array, Any, jax.Array]:
        """Perturb the parameters of `swarm` by `sigmas` times standard normal draws
        of `key_perturb`, then :meth:`advance`. Returns the states, the perturbed
        parameters, the time index and the log-densities."""
        ...

    def resample_swarm(
        self,
        swarm: Swarm,
        X: jax.Array,
        thetas: jax.Array,
        resample: jax.Array,
        counts: jax.Array,
        norm_weights: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, Swarm]:
        """Resample `X` and `thetas`, the outputs of :meth:`advance_perturbed` from
        `swarm`, where `resample` holds, as
        :func:`~pypomp.core.algorithms.helpers._resample_where` does. Returns its
        indices and weights and the next interval's swarm."""
        ...

    def scan(self, body: Callable, init: Any, xs: Any) -> Any:
        """Return the final carry of ``jax.lax.scan(body, init, xs)``."""
        ...

    def parameters(self, swarm: Swarm) -> jax.Array:
        """Return the parameters of `swarm`, resampled."""
        ...


@dataclass(frozen=True)
class XlaEngine:
    """The :class:`IntervalEngine` of the model's own JAX functions."""

    rprocess_interp: Callable
    dmeasure: Callable

    def advance(
        self, X: jax.Array, theta: jax.Array, iv: Interval, y: jax.Array, t_obs: Any
    ) -> tuple[jax.Array, Any, jax.Array]:
        X, t_idx = self.rprocess_interp(X, theta, *iv)
        cov = None if iv.covars_extended is None else iv.covars_extended[t_idx]
        return X, t_idx, self.dmeasure(y, X, theta, cov, t_obs, iv.should_trans)

    def resample(self, norm_weights: jax.Array, key: jax.Array) -> jax.Array:
        return _resample(norm_weights, key)

    def swarm(self, X: jax.Array, thetas: jax.Array) -> Swarm:
        empty = jnp.zeros((0,))
        return Swarm(X, thetas, jnp.zeros((0,), jnp.int32), (empty, empty))

    def advance_perturbed(
        self,
        swarm: Swarm,
        key_perturb: jax.Array,
        sigmas: jax.Array,
        iv: Interval,
        y: jax.Array,
        t_obs: Any,
    ) -> tuple[jax.Array, jax.Array, Any, jax.Array]:
        thetas = swarm.thetas + sigmas * jax.random.normal(
            shape=swarm.thetas.shape, key=key_perturb
        )
        # Draw the perturbations once: otherwise XLA may fuse the draws into each of
        # their consumers (dmeasure, the resampling gather) and draw them twice.
        thetas = jax.lax.optimization_barrier(thetas)
        X, t_idx, logw = self.advance(swarm.particles, thetas, iv, y, t_obs)
        return X, thetas, t_idx, logw

    def resample_swarm(
        self,
        swarm: Swarm,
        X: jax.Array,
        thetas: jax.Array,
        resample: jax.Array,
        counts: jax.Array,
        norm_weights: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, Swarm]:
        counts, norm_weights, (X, thetas) = _resample_where(
            resample, counts, norm_weights, key, (X, thetas), self.resample
        )
        return counts, norm_weights, replace(swarm, particles=X, thetas=thetas)

    def scan(self, body: Callable, init: Any, xs: Any) -> Any:
        return jax.lax.scan(body, init, xs)[0]

    def parameters(self, swarm: Swarm) -> jax.Array:
        return swarm.thetas


def engine_for(rprocess_interp: Callable, dmeasure: Callable) -> IntervalEngine:
    """Return the engine that runs `rprocess_interp` with `dmeasure`.

    A process model of another backend, such as
    :class:`pypomp.cute.KernelRProc`, makes its own through its ``engine``
    method; any other runs on :class:`XlaEngine`.
    """
    make = getattr(rprocess_interp, "engine", None)
    if make is None:
        return XlaEngine(rprocess_interp, dmeasure)
    return make(dmeasure)
