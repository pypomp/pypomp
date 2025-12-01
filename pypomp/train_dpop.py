from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable
from .dpop import _dpop_internal_mean  # DPOP mean negative log-likelihood per observation

# ----------------------------------------------------------------------
# DPOP gradient helpers and a simple SGD-with-decay optimizer
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
    rprocess: Callable,      # static conceptually
    dmeasure: Callable,      # static conceptually
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
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
    rprocess: Callable,      # static conceptually
    dmeasure: Callable,      # static conceptually
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
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
        key=key,
    )


def dpop_sgd_decay(
    theta_init: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,
    rinitializer: Callable,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
    key: jax.Array,
    M: int = 40,
    eta0: float = 0.01,
    decay: float = 0.1,
) -> tuple[jax.Array, jax.Array]:
    """
    Simple SGD with decaying step size on the DPOP objective.

    This function does NOT touch any Pomp object directly. It only works with
    the low-level arrays and callables, so you can use it with any Pomp
    instance by extracting the needed fields.

    Parameters
    ----------
    theta_init : jax.Array, shape (p,)
        Initial parameter vector in *estimation space*, ordered by canonical
        parameter names.
    ys : jax.Array, shape (n_times, y_dim) or (n_times,)
        Observations passed to the DPOP objective.
    dt_array_extended, nstep_array, t0, times, J,
    rinitializer, rprocess_interp, dmeasure, accumvars, covars_extended,
    alpha, process_weight_index, key :
        Exactly the same meaning as in `_dpop_internal_mean`.
    M : int, default 40
        Number of optimization iterations.
    eta0 : float, default 0.01
        Initial learning rate.
    decay : float, default 0.1
        Learning-rate decay coefficient. At iteration m, the effective
        step size is:

            eta_m = eta0 / (1 + decay * m)

    Returns
    -------
    theta_history : jax.Array, shape (M+1, p)
        The parameter vector at each iteration, including the initial point.
    nll_history : jax.Array, shape (M+1,)
        The mean negative log-likelihood per observation at each iteration
        (as returned by `_dpop_internal_mean`).
    """
    theta = theta_init
    n_obs = ys.shape[0]

    theta_history = []
    nll_history = []

    for m in range(M + 1):
        # Record current parameters
        theta_history.append(theta)

        # Compute mean NLL and its gradient at the current theta
        key, subkey = jax.random.split(key)
        nll_mean, grad = _jvg_dpop(
            theta_ests=theta,
            ys=ys,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            t0=t0,
            times=times,
            J=J,
            rinitializer=rinitializer,
            rprocess=rprocess_interp,
            dmeasure=dmeasure,
            accumvars=accumvars,
            covars_extended=covars_extended,
            alpha=alpha,
            process_weight_index=process_weight_index,
            key=subkey,
        )

        nll_history.append(nll_mean)

        # Last iteration: do not update further
        if m == M:
            break

        # Decayed learning rate: eta_m = eta0 / (1 + decay * m)
        m_f = float(m)
        lr = eta0 / (1.0 + decay * m_f)

        # Gradient step
        grad_safe = jnp.where(jnp.isnan(grad), 0.0, grad)
        theta = theta - lr * grad_safe

    theta_history = jnp.stack(theta_history, axis=0)
    nll_history = jnp.stack(nll_history, axis=0)

    return theta_history, nll_history
