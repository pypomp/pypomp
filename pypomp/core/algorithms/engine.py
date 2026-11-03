"""How a filter advances its particles over one observation interval.

The particle filters (``pfilter``, and the perturbed filter of iterated filtering)
leave every step that depends on how the model runs to an :class:`IntervalEngine`:
propagating the particles, evaluating the measurement density, perturbing the
parameters and computing resampling indices. :class:`XlaEngine` does these with the
model's own JAX functions. A struct whose ``backend`` holds a :class:`Backend`, such
as CuTe's, gets that backend's engine instead (see :func:`engine_for`), so the
filters need not know which runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp

from .carries import Swarm
from .helpers import _resample, _resample_where

if TYPE_CHECKING:
    from ...functional.structs import PanelPompStruct, PompStruct


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

    def propagate(
        self, X: jax.Array, theta: jax.Array, iv: Interval
    ) -> tuple[jax.Array, Any]:
        """Propagate `X` over `iv` and return the states and the time index at the
        end of the interval."""
        ...

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


class Backend(Protocol):
    """What a struct's ``backend`` field holds: the filters' engines for a model,
    built by another backend than XLA."""

    def engine(
        self, struct: PompStruct | PanelPompStruct, per_particle: bool
    ) -> IntervalEngine | None:
        """Return the engine for `struct`'s functions of one flavor, those of
        iterated filtering if `per_particle` and of ``pfilter`` otherwise, or None
        if the backend wasn't built for them, so they run on XLA.

        Raises if the backend can't run `struct`'s time grid.
        """
        ...


@dataclass(frozen=True)
class XlaEngine:
    """The :class:`IntervalEngine` of the model's own JAX functions.

    `dmeasure` may be None for algorithms that only simulate, which call only
    :meth:`propagate`.
    """

    rprocess_interp: Callable
    dmeasure: Callable | None

    def propagate(
        self, X: jax.Array, theta: jax.Array, iv: Interval
    ) -> tuple[jax.Array, Any]:
        return self.rprocess_interp(X, theta, *iv)

    def advance(
        self, X: jax.Array, theta: jax.Array, iv: Interval, y: jax.Array, t_obs: Any
    ) -> tuple[jax.Array, Any, jax.Array]:
        if self.dmeasure is None:
            raise ValueError("dmeasure is required")
        X, t_idx = self.propagate(X, theta, iv)
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


def engine_for(
    struct: PompStruct | PanelPompStruct, per_particle: bool
) -> IntervalEngine:
    """Return the engine that runs `struct`'s functions of one flavor, those of
    iterated filtering (``rproc_per``, ``dmeas_per``) if `per_particle` and of
    ``pfilter`` otherwise.

    That is its backend's engine, if it has one for them (see :class:`Backend`),
    and otherwise :class:`XlaEngine`.
    """
    backend = struct.backend
    engine = None if backend is None else backend.engine(struct, per_particle)
    if engine is not None:
        return engine
    if per_particle:
        return XlaEngine(struct.rproc_per, struct.dmeas_per)
    return XlaEngine(struct.rproc_pf, struct.dmeas_pf)
