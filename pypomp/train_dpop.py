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
    rprocess: Callable,      # static conceptually
    dmeasure: Callable,      # static conceptually
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


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "process_weight_index",
        "ntimes",
        "M",
    ),
)
def dpop_sgd_decay(
    theta_init: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,  # static
    ntimes: int,  # static
    key: jax.Array,
    M: int = 40,  # static
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
    alpha, process_weight_index, ntimes, key :
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
    # Initialize arrays for storing results
    theta_history = jnp.zeros((M + 1, theta_init.shape[0]))
    nll_history = jnp.zeros(M + 1)

    # Set initial values
    theta_history = theta_history.at[0].set(theta_init)

    # Create the step function for fori_loop
    def train_step(m, carry):
        theta, key, theta_history, nll_history = carry

        # Compute mean NLL and its gradient at the current theta
        key, subkey = jax.random.split(key)
        nll_mean, grad = jax.value_and_grad(_dpop_internal_mean)(
            theta,
            ys=ys,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            t0=t0,
            times=times,
            J=J,
            rinitializer=rinitializer,
            rprocess_interp=rprocess_interp,
            dmeasure=dmeasure,
            accumvars=accumvars,
            covars_extended=covars_extended,
            alpha=alpha,
            process_weight_index=process_weight_index,
            ntimes=ntimes,
            key=subkey,
        )

        # Record NLL
        nll_history = nll_history.at[m].set(nll_mean)

        # Decayed learning rate: eta_m = eta0 / (1 + decay * m)
        m_f = m.astype(jnp.float32)
        lr = eta0 / (1.0 + decay * m_f)

        # Gradient step with NaN protection
        grad_safe = jnp.where(jnp.isnan(grad), 0.0, grad)
        theta_new = theta - lr * grad_safe

        # Record new theta for next iteration
        theta_history = theta_history.at[m + 1].set(theta_new)

        return (theta_new, key, theta_history, nll_history)

    # Run the optimization loop
    theta_final, key_final, theta_history, nll_history = jax.lax.fori_loop(
        0,
        M,
        train_step,
        (theta_init, key, theta_history, nll_history),
    )

    # Compute final NLL (at iteration M)
    key_final, subkey = jax.random.split(key_final)
    final_nll = _dpop_internal_mean(
        theta_final,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess_interp,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=subkey,
    )
    nll_history = nll_history.at[M].set(final_nll)

    return theta_history, nll_history
