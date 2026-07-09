from typing import Mapping
import jax
import jax.numpy as jnp


def align_params(
    params: Mapping[str, jax.Array | float],
    names: list[str],
    axis: int = -1,
) -> jax.Array:
    """Align and stack parameter arrays into the canonical ordering for a model struct.

    Builds a single JAX array from a dictionary of named parameter values,
    reordering them to match the canonical ``param_names`` order expected
    by :func:`pypomp.functional.pfilter`, :func:`mif`, and
    :func:`train`.

    Parameters
    ----------
    params : mapping of str to jax.Array or float
        Dictionary mapping parameter names to JAX arrays or float scalars.
    names : list of str
        Canonical parameter name ordering (e.g. ``struct.param_names``).
    axis : int, optional
        Axis along which to stack.  Defaults to ``-1`` (last axis).

    Returns
    -------
    jax.Array
        Array whose last axis (by default) corresponds to ``names`` in
        order.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import pypomp.functional as F
    >>> params = {"beta": jnp.array(0.5), "gamma": jnp.array(0.1)}
    >>> F.align_params(params, names=["gamma", "beta"])
    Array([0.1, 0.5], dtype=float32)
    """
    try:
        return jnp.stack([jnp.asarray(params[name]) for name in names], axis=axis)
    except KeyError as e:
        raise KeyError(
            f"Parameter '{e.args[0]}' is required by the model structure but missing from inputs."
        )
