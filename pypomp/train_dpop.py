from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable
from .dpop import _dpop_internal_mean  # DPOP mean negative log-likelihood per observation

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


# ----------------------------------------------------------------------
# Unified DPOP optimizer: Adam or SGD, with optional per-parameter LR
# and LR decay
# ----------------------------------------------------------------------

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
        "optimizer",
    ),
)
def dpop_train(
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
    eta: jax.Array | None = None,
    optimizer: str = "Adam",  # static
    decay: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Train on the DPOP objective with Adam or SGD, optional LR decay.

    This function does NOT touch any Pomp object directly. It only works
    with the low-level arrays and callables, so you can use it with any
    Pomp instance by extracting the needed fields.

    Parameters
    ----------
    theta_init : jax.Array, shape (p,)
        Initial parameter vector in estimation space.
    eta : jax.Array, shape (p,) or None
        Per-parameter learning rates. If None, defaults to 0.01 for all.
    optimizer : str
        "Adam" or "SGD".
    decay : float
        Learning-rate decay coefficient. At iteration m, the effective
        learning rate is ``eta / (1 + decay * m)``.
    M : int
        Number of optimization iterations.

    Returns
    -------
    theta_history : jax.Array, shape (M+1, p)
        Parameter vector at each iteration.
    nll_history : jax.Array, shape (M+1,)
        Mean negative log-likelihood per observation at each iteration.
    """
    p = theta_init.shape[0]
    if eta is None:
        eta = jnp.full(p, 0.01)

    theta_history = jnp.zeros((M + 1, p))
    nll_history = jnp.zeros(M + 1)
    theta_history = theta_history.at[0].set(theta_init)

    # Adam state: first and second moment estimates
    m_adam = jnp.zeros(p)
    v_adam = jnp.zeros(p)

    def train_step(m, carry):
        theta, key, theta_history, nll_history, m_adam, v_adam = carry

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

        nll_history = nll_history.at[m].set(nll_mean)

        # NaN protection
        grad_safe = jnp.where(jnp.isnan(grad), 0.0, grad)

        # Learning rate decay
        m_f = m.astype(jnp.float32)
        lr_scale = 1.0 / (1.0 + decay * m_f)
        eta_scaled = eta * lr_scale

        if optimizer == "Adam":
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            # Update biased moments (m is 0-indexed iteration)
            m_adam_new = beta1 * m_adam + (1.0 - beta1) * grad_safe
            v_adam_new = beta2 * v_adam + (1.0 - beta2) * grad_safe**2
            # Bias correction (use m+1 since m is 0-indexed)
            step = (m_f + 1.0).astype(jnp.float32)
            m_hat = m_adam_new / (1.0 - beta1**step)
            v_hat = v_adam_new / (1.0 - beta2**step)
            direction = -m_hat / (jnp.sqrt(v_hat) + eps)
            theta_new = theta + eta_scaled * direction
        else:
            # SGD
            m_adam_new = m_adam
            v_adam_new = v_adam
            theta_new = theta - eta_scaled * grad_safe

        theta_history = theta_history.at[m + 1].set(theta_new)

        return (theta_new, key, theta_history, nll_history, m_adam_new, v_adam_new)

    theta_final, key_final, theta_history, nll_history, _, _ = jax.lax.fori_loop(
        0,
        M,
        train_step,
        (theta_init, key, theta_history, nll_history, m_adam, v_adam),
    )

    # Compute final NLL
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
