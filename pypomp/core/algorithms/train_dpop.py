from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from ..optimizer import Optimizer
from .contexts import DpopTrainContext
from .dpop import (
    _dpop_internal_mean,
)  # DPOP mean negative log-likelihood per observation
from .helpers import _cosine_cooling

# ----------------------------------------------------------------------
# DPOP gradient helpers
# ----------------------------------------------------------------------


def _jgrad_dpop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static conceptually (number of particles)
    rinitializer: Callable,  # static conceptually
    rprocess: Callable,  # static conceptually
    dmeasure: Callable,  # static conceptually
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
    ntimes: int,  # static - number of observation times
    key: jax.Array,
) -> jax.Array:
    """
    Gradient of the DPOP mean negative log-likelihood with respect to theta_ests.

    This wraps `_dpop_internal_mean` with `jax.grad`. The objective is the mean
    negative log-likelihood per observation, so the gradient is scaled
    accordingly (which is fine for optimization).
    """
    return jax.grad(_dpop_internal_mean)(
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
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=key,
    )


def _jvg_dpop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static conceptually (number of particles)
    rinitializer: Callable,  # static conceptually
    rprocess: Callable,  # static conceptually
    dmeasure: Callable,  # static conceptually
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
    ntimes: int,  # static - number of observation times
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Value and gradient of the DPOP mean negative log-likelihood.

    Returns
    -------
    value : scalar jax.Array
        Mean negative log-likelihood per observation under DPOP.
    grad : jax.Array, same shape as theta_ests
        Gradient of the objective with respect to theta_ests.
    """
    return jax.value_and_grad(_dpop_internal_mean)(
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
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=key,
    )


# ----------------------------------------------------------------------
# Unified DPOP optimizer using DpopTrainContext & Optimizer
# ----------------------------------------------------------------------


def _dpop_train_scan_step(
    context: DpopTrainContext,
    optimizer: Optimizer,
    carry: tuple[jax.Array, jax.Array, tuple],
    m: int | jax.Array,
) -> tuple[tuple[jax.Array, jax.Array, tuple], tuple[jax.Array, jax.Array]]:
    theta, key, opt_state = carry
    ntimes = context.series.ys.shape[0]

    key, subkey = jax.random.split(key)
    curr_alpha = 1.0 - (1.0 - context.alpha) * _cosine_cooling(
        m, context.M, context.alpha_cooling
    )
    nll_mean, grad = jax.value_and_grad(_dpop_internal_mean)(
        theta,
        ys=context.series.ys,
        dt_array_extended=context.series.dt_array_extended,
        nstep_array=context.series.nstep_array,
        t0=context.series.t0,
        times=context.series.times,
        J=context.J,
        rinitializer=context.fns.rinitializer,
        rprocess_interp=context.fns.rprocess_interp,
        dmeasure=context.fns.dmeasure,
        accumvars=context.fns.accumvars,
        covars_extended=context.series.covars_extended,
        alpha=curr_alpha,
        process_weight_index=context.process_weight_index,
        ntimes=ntimes,
        key=subkey,
    )

    total_nll = nll_mean * ntimes
    grad_safe = jnp.where(jnp.isnan(grad), 0.0, grad)
    if optimizer.clip_norm is not None:
        grad_safe = jnp.clip(grad_safe, -optimizer.clip_norm, optimizer.clip_norm)

    eta_m = context.eta[m]

    direction, new_opt_state = optimizer.step(
        grad=grad_safe,
        state=opt_state,
        step_num=m,
        eta_i=eta_m,
    )

    if optimizer.scale:
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-8)

    theta_new = theta + eta_m * direction
    new_carry = (theta_new, key, new_opt_state)
    metrics = (total_nll, theta_new)
    return new_carry, metrics


@partial(jit, static_argnames=("optimizer",))
def _dpop_train_internal(
    theta_ests: jax.Array,
    key: jax.Array,
    context: DpopTrainContext,
    optimizer: Optimizer,
) -> tuple[jax.Array, jax.Array]:
    ntimes = context.series.ys.shape[0]

    initial_carry = (
        theta_ests,
        key,
        optimizer.init_state(theta_ests),
    )

    step_fn = jax.tree_util.Partial(_dpop_train_scan_step, context, optimizer)

    (theta_final, key_final, _), (neg_logliks_body, theta_history_body) = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(context.M),
    )

    # Compute final NLL at iteration M
    key_final, subkey = jax.random.split(key_final)
    final_alpha = 1.0 - (1.0 - context.alpha) * _cosine_cooling(
        context.M, context.M, context.alpha_cooling
    )
    final_nll_mean = _dpop_internal_mean(
        theta_final,
        ys=context.series.ys,
        dt_array_extended=context.series.dt_array_extended,
        nstep_array=context.series.nstep_array,
        t0=context.series.t0,
        times=context.series.times,
        J=context.J,
        rinitializer=context.fns.rinitializer,
        rprocess_interp=context.fns.rprocess_interp,
        dmeasure=context.fns.dmeasure,
        accumvars=context.fns.accumvars,
        covars_extended=context.series.covars_extended,
        alpha=final_alpha,
        process_weight_index=context.process_weight_index,
        ntimes=ntimes,
        key=subkey,
    )
    final_total_nll = final_nll_mean * ntimes

    neg_logliks = jnp.concatenate([neg_logliks_body, jnp.array([final_total_nll])])
    theta_history = jnp.concatenate(
        [theta_ests[jnp.newaxis, :], theta_history_body], axis=0
    )

    return neg_logliks, theta_history


_vmapped_dpop_train_internal = jax.vmap(
    _dpop_train_internal,
    in_axes=(0, 0, None, None),
)
