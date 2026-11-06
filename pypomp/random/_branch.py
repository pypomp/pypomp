"""Selecting between two computations, computing only one of them where possible.

The samplers have shortcuts for one element: a ``lax.cond`` computes only the
method, piece or number of terms a draw needs. They pay off only where each
element really runs on its own, as when the CuTe backend traces one particle per
thread and turns each cond into a branch. Under ``vmap``, as in XLA's filters, a
cond becomes a select of every side and costs more than the batched code, so the
shortcuts are taken only inside :func:`branching_traces`.
"""

import contextlib
import contextvars
import functools
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

_BRANCHING = contextvars.ContextVar("pypomp_random_branching", default=False)


def branching() -> bool:
    """Return whether one-element sampler calls take their branching shortcuts."""
    return _BRANCHING.get()


@contextlib.contextmanager
def branching_traces() -> Iterator[None]:
    """Trace one-element sampler calls with branches, for a backend that runs one
    element per thread."""
    token = _BRANCHING.set(True)
    try:
        yield
    finally:
        _BRANCHING.reset(token)


def jit(**jit_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return ``jax.jit(fn, **jit_kwargs)``, with a second jitted copy of `fn` for
    calls inside :func:`branching_traces`.

    jit reuses a trace by its function and arguments alone, so a sampler traced
    for XLA would otherwise be reused, without its shortcuts, when the CuTe
    backend traces a particle, or one traced with them would be reused by XLA.
    The copy wraps `fn` in a function of its own, so the two share no trace. Both
    stay jit calls, which the CuTe translator emits once as a device function
    however often a step calls them.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        jitted = jax.jit(fn, **jit_kwargs)
        jitted_branching = jax.jit(
            functools.wraps(fn)(lambda *a, **k: fn(*a, **k)), **jit_kwargs
        )

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            if branching():
                return jitted_branching(*args, **kwargs)
            return jitted(*args, **kwargs)

        return call

    return decorate


def where_lazy(
    pred: jax.Array, true_fn: Callable[[], jax.Array], false_fn: Callable[[], jax.Array]
) -> jax.Array:
    """Return ``jnp.where(pred, true_fn(), false_fn())``.

    Both sides are computed and selected, exactly as ``jnp.where`` does, except
    for a scalar `pred` inside :func:`branching_traces`: there this is a
    ``lax.cond``, which the CuTe backend turns into a branch, so a sampler pays
    only for the method each particle needs.
    """
    if branching() and jnp.ndim(pred) == 0:
        return lax.cond(pred, true_fn, false_fn)
    return jnp.where(pred, true_fn(), false_fn())
