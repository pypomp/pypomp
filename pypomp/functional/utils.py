from typing import Mapping
import jax
import jax.numpy as jnp


def align_params(
    params: Mapping[str, jax.Array | float],
    names: list[str],
    axis: int = -1,
) -> jax.Array:
    """
    Stateless utility to align and stack parameter arrays or scalars into a single
    JAX array matching the specified canonical names.

    This is useful for preparing the parameter arrays required by functions under
    `pypomp.functional` (e.g. `mif`, `pfilter`, `train`) from dictionaries.

    Args:
        params: Dictionary mapping parameter names to JAX arrays or float scalars.
        names: List of parameter names in target canonical order (e.g. struct.param_names).
        axis: The axis along which to stack the parameters (defaults to the last axis).

    Returns:
        jax.Array: Stacked array aligned with the canonical names.
    """
    try:
        return jnp.stack([jnp.asarray(params[name]) for name in names], axis=axis)
    except KeyError as e:
        raise KeyError(
            f"Parameter '{e.args[0]}' is required by the model structure but missing from inputs."
        )
