import jax
from typing import Callable

from .pfilter import _pfilter_internal_mean
from .mop import _mop_internal_mean


_grad_pfilter_internal_mean = jax.grad(_pfilter_internal_mean)
_vg_pfilter_internal_mean = jax.value_and_grad(_pfilter_internal_mean)
_hess_pfilter_internal_mean = jax.hessian(_pfilter_internal_mean)
_grad_mop_internal_mean = jax.grad(_mop_internal_mean)
_vg_mop_internal_mean = jax.value_and_grad(_mop_internal_mean)
_hess_mop_internal_mean = jax.hessian(_mop_internal_mean)


def _jgrad(
    theta_ests: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    n_obs: int,
):
    """
    calculates the gradient of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the gradient of the pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return _grad_pfilter_internal_mean(
        theta_ests,  # for some reason this needs to be given as a positional argument
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        key=key,
        n_obs=n_obs,
    )


def _jvg(
    theta_ests: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    n_obs: int,
):
    """
    Calculates the both the value and gradient of a mean particle filter
    objective (function 'pfilter_internal_mean') w.r.t. the current estimated
    parameter value using JAX's automatic differentiation.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using pfilter_internal_mean function.
        - The gradient of the function pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return _vg_pfilter_internal_mean(
        theta_ests,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        key=key,
        n_obs=n_obs,
    )


def _jgrad_mop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float | jax.Array,
    key: jax.Array,
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
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        key=key,
    )


def _jvg_mop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    covars_extended: jax.Array | None,
    accumvars: tuple[int, ...] | None,
    alpha: float | jax.Array,
    key: jax.Array,
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
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        accumvars=accumvars,
        alpha=alpha,
        key=key,
    )


def _jhess(
    theta_ests: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    n_obs: int,
):
    """
    calculates the Hessian matrix of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the Hessian matrix of the pfilter_internal_mean function
            w.r.t. theta_ests.
    """
    return _hess_pfilter_internal_mean(
        theta_ests,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        key=key,
        n_obs=n_obs,
    )


# get the hessian matrix from mop
def _jhess_mop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float | jax.Array,
    key: jax.Array,
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
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        key=key,
    )
