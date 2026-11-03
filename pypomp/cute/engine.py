"""The filters' :class:`~pypomp.core.algorithms.engine.IntervalEngine` on the
kernels of a :class:`~pypomp.cute.CuteBackend`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..core.algorithms.carries import Swarm
from ..core.algorithms.engine import Interval, XlaEngine
from ..core.algorithms.helpers import _resample_where
from . import Kernels


@dataclass(frozen=True)
class CuteEngine(XlaEngine):
    """Runs each step on the `kernels` built from ``rprocess_interp`` that it has,
    and the rest as :class:`XlaEngine` does.

    The kernels that evaluate the measurement density, the fused one and that of
    iterated filtering, are used only if `dmeasure` is the density they evaluate,
    so a struct whose density was replaced gets the new one, on XLA.
    """

    kernels: Kernels

    @property
    def _fused(self) -> bool:
        return self.kernels.dmeasure is self.dmeasure

    @property
    def _perturb_step(self) -> Callable | None:
        return self.kernels.perturb_step if self._fused else None

    def propagate(
        self, X: jax.Array, theta: jax.Array, iv: Interval
    ) -> tuple[jax.Array, Any]:
        return self.kernels.rproc(X, theta, *iv)

    def advance(
        self, X: jax.Array, theta: jax.Array, iv: Interval, y: jax.Array, t_obs: Any
    ) -> tuple[jax.Array, Any, jax.Array]:
        fused = self.kernels.with_dmeasure
        if self._fused and fused is not None:
            return fused(X, theta, *iv, y, t_obs)
        return super().advance(X, theta, iv, y, t_obs)

    def resample(self, norm_weights: jax.Array, key: jax.Array) -> jax.Array:
        kernel = self.kernels.resample
        if kernel is None:
            return super().resample(norm_weights, key)
        return kernel(norm_weights, key)

    # With a perturb_step kernel, iterated filtering resamples as the kernel loads
    # the particles: the swarm holds them as they were before the last resampling,
    # with its ancestors.

    def swarm(self, X: jax.Array, thetas: jax.Array) -> Swarm:
        if self._perturb_step is None:
            return super().swarm(X, thetas)
        anc = jnp.arange(X.shape[0], dtype=jnp.int32)
        return Swarm(X, thetas, anc, (jnp.zeros_like(X), jnp.zeros_like(thetas)))

    def advance_perturbed(
        self,
        swarm: Swarm,
        key_perturb: jax.Array,
        sigmas: jax.Array,
        iv: Interval,
        y: jax.Array,
        t_obs: Any,
    ) -> tuple[jax.Array, jax.Array, Any, jax.Array]:
        step = self._perturb_step
        if step is None:
            return super().advance_perturbed(swarm, key_perturb, sigmas, iv, y, t_obs)
        # One kernel resamples on load, perturbs the parameters, propagates the
        # particles and evaluates the measurement density.
        return step(
            swarm.particles,
            swarm.thetas,
            swarm.anc,
            key_perturb,
            sigmas,
            *iv,
            y,
            t_obs,
            swarm.spare,
        )

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
        if self._perturb_step is None:
            return super().resample_swarm(
                swarm, X, thetas, resample, counts, norm_weights, key
            )
        # The kernel resamples as it loads, so only the ancestors are needed. The
        # buffers it read from are spent, so it writes the next outputs over them.
        J = norm_weights.shape[-1]
        counts, norm_weights, anc = _resample_where(
            resample,
            counts,
            norm_weights,
            key,
            jnp.arange(J, dtype=jnp.int32),
            self.resample,
        )
        return (
            counts,
            norm_weights,
            Swarm(X, thetas, anc, (swarm.particles, swarm.thetas)),
        )

    def scan(self, body: Callable, init: Any, xs: Any) -> Any:
        if self._perturb_step is None:
            return super().scan(body, init, xs)
        return _scan_in_pairs(body, init, xs)

    def parameters(self, swarm: Swarm) -> jax.Array:
        if self._perturb_step is None:
            return super().parameters(swarm)
        return swarm.thetas[swarm.anc]


def _scan_in_pairs(body: Callable, init: Any, xs: Any) -> Any:
    """Return the final carry of scanning `body` over `xs`, two steps per
    iteration.

    ``perturb_step`` writes over the spare buffers and :meth:`CuteEngine.resample_swarm`
    swaps them with the carry, so after two steps every buffer is back in its slot
    and XLA need not copy the kernel's outputs into the carry.
    """
    n = len(jax.tree.leaves(xs)[0])
    pairs = jax.tree.map(
        lambda a: a[: n - n % 2].reshape((n // 2, 2) + a.shape[1:]), xs
    )

    def two(carry, xs2):
        for k in range(2):
            carry, _ = body(carry, jax.tree.map(lambda a, k=k: a[k], xs2))
        return carry, None

    carry, _ = jax.lax.scan(two, init, pairs)
    if n % 2:
        carry, _ = body(carry, jax.tree.map(lambda a: a[-1], xs))
    return carry
