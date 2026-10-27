"""Selecting between two computations, computing only one of them where possible."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax


def where_lazy(
    pred: jax.Array, true_fn: Callable[[], jax.Array], false_fn: Callable[[], jax.Array]
) -> jax.Array:
    """Return ``jnp.where(pred, true_fn(), false_fn())``.

    On arrays both sides are computed and selected, exactly as ``jnp.where`` does.
    On a scalar `pred` this is a ``lax.cond``, which computes one side: the CuTe
    backend traces one particle at a time and turns it into a branch, so a
    sampler pays only for the method each particle needs. Under ``vmap`` the cond
    becomes a select of both sides again.
    """
    if jnp.ndim(pred) == 0:
        return lax.cond(pred, true_fn, false_fn)
    return jnp.where(pred, true_fn(), false_fn())
