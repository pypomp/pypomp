import jax
from .mop import _mop_internal_mean
from .types import MopConfig, MopInputs

_grad_mop_internal_mean = jax.grad(_mop_internal_mean)
_vg_mop_internal_mean = jax.value_and_grad(_mop_internal_mean)
_hess_mop_internal_mean = jax.hessian(_mop_internal_mean)


def _jgrad_mop(
    theta_ests: jax.Array,
    key: jax.Array,
    config: MopConfig,
    inputs: MopInputs,
):
    """
    Calculates the gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the gradient of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return _grad_mop_internal_mean(
        theta_ests,
        key,
        config,
        inputs,
    )


def _jvg_mop(
    theta_ests: jax.Array,
    key: jax.Array,
    config: MopConfig,
    inputs: MopInputs,
) -> tuple:
    """
    Calculates the both the value and gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using mop_internal_mean function.
        - The gradient of the function mop_internal_mean function w.r.t.
            theta_ests.
    """
    return _vg_mop_internal_mean(
        theta_ests,
        key,
        config,
        inputs,
    )


# get the hessian matrix from mop
def _jhess_mop(
    theta_ests: jax.Array,
    key: jax.Array,
    config: MopConfig,
    inputs: MopInputs,
):
    """
    calculates the Hessian matrix of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the Hessian matrix of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return _hess_mop_internal_mean(
        theta_ests,
        key,
        config,
        inputs,
    )
