"""
This module implements the DPOP algorithm for POMP models.

DPOP is a differentiable particle filter objective that augments the
MOP objective with an additional transition-density penalty. The
transition log-weight is assumed to be accumulated inside the latent
state by the user-defined process model.
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit

from .internal_functions import _keys_helper, _normalize_weights, _resampler

# Should transformations be applied to the parameters?
SHOULD_TRANS = True


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "process_weight_index",
    ),
)
def _dpop_internal(
    theta: jax.Array,
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
    process_weight_index: int | None,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for the DPOP algorithm.

    This function is analogous to ``_mop_internal`` but additionally
    incorporates a per-interval transition log-weight that is stored
    in one of the state components.

    The process log-weight is expected to be accumulated over a single
    observation interval by the process model. At the beginning of
    each interval the corresponding state component should be reset to
    zero (this is naturally handled via ``accumvars`` in ``RProc``).

    Args:
        theta: Parameter vector in estimation space.
        ys: Observation array with shape (ntimes, y_dim).
        dt_array_extended: Time step sizes for each internal step.
        nstep_array: Number of internal steps per observation interval.
        t0: Initial time.
        times: Observation times.
        J: Number of particles.
        rinitializer: Vectorized initial-state simulator.
        rprocess_interp: Vectorized time-interpolated process model.
        dmeasure: Vectorized measurement log-density.
        accumvars: Indices of accumulated state variables (as passed to
            ``RProc``). This argument is only forwarded to the process
            model and is not interpreted directly here.
        covars_extended: Precomputed covariates on the internal time
            grid, or None if no covariates are used.
        alpha: Cooling factor for the particle weights.
        process_weight_index: Index of the state component that stores
            the accumulated transition log-weight over a single
            observation interval. If None, no process penalty is
            applied and DPOP reduces to MOP.
        key: JAX random key.

    Returns:
        A scalar JAX array containing the negative DPOP log-likelihood
        estimate for the given parameter vector.
    """
    times = times.astype(float)

    # Initialize particles from the prior.
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0, SHOULD_TRANS)

    # Start from equal log-weights.
    weightsF = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J, dtype=jnp.int32)
    loglik = 0.0

    # Use checkpointing to keep memory usage manageable when backpropagating
    # through many observation times.
    dpop_helper_2 = jax.checkpoint(
        partial(
            _dpop_helper,
            ys=ys,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            times=times,
            theta=theta,
            rprocess_interp=rprocess_interp,
            dmeasure=dmeasure,
            accumvars=accumvars,
            covars_extended=covars_extended,
            alpha=alpha,
            process_weight_index=process_weight_index,
        )
    )

    # Loop over observation times.
    t, particlesF, loglik, weightsF, counts, key, t_idx = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=dpop_helper_2,
        init_val=(t0, particlesF, loglik, weightsF, counts, key, 0),
    )

    # Return the negative log-likelihood (to be minimized).
    return -loglik


def _dpop_helper(
    i: int,
    inputs: tuple[
        jax.Array,  # t
        jax.Array,  # particlesF
        jax.Array,  # loglik
        jax.Array,  # weightsF
        jax.Array,  # counts
        jax.Array,  # key
        int,  # t_idx
    ],
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    times: jax.Array,
    theta: jax.Array,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    int,
]:
    """
    Single DPOP filtering iteration over one observation time.

    This function performs:
        1. A prediction step using the process model.
        2. An update of the cooled weights with the transition
           log-weight (if provided).
        3. A measurement update and resampling step, where resampling
           is driven only by the measurement weights.
    """
    t, particlesF, loglik, weightsF, counts, key, t_idx = inputs
    J = particlesF.shape[0]

    # Cooled weights from the previous time step.
    weightsP = alpha * weightsF

    # Keys for the process model.
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)

    # Predict forward within the current observation interval.
    nstep = nstep_array[i].astype(int)
    particlesP, t_idx = rprocess_interp(
        particlesF,
        theta,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep,
        accumvars,
        SHOULD_TRANS,
    )
    t = times[i]

    # Optional process log-weight from the state.
    if process_weight_index is not None:
        # Interpret the state entry at `process_weight_index` as the
        # accumulated transition log-weight over this interval.
        proc_w = particlesP[:, process_weight_index]
        weightsP = weightsP + (proc_w - jax.lax.stop_gradient(proc_w))

    # Measurement update.
    covars_t = None if covars_extended is None else covars_extended[t_idx]
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t, SHOULD_TRANS)
    if measurements.ndim > 1:
        # Sum over any extra dimension if the measurement density
        # returns a multi-component log-density.
        measurements = measurements.sum(axis=-1)

    # Update the global log-likelihood (data term).
    loglik = (
        loglik
        + jax.scipy.special.logsumexp(weightsP + measurements)
        - jax.scipy.special.logsumexp(weightsP)
    )

    # Resampling is driven by the measurement weights only. This keeps
    # the transition penalty from dominating the ancestral weights.
    norm_meas_w, _ = _normalize_weights(jax.lax.stop_gradient(measurements))
    key, subkey = jax.random.split(key)
    counts, particlesF, _ = _resampler(counts, particlesP, norm_meas_w, subkey=subkey)

    # Propagate the cooled + penalized weights through the resampling
    # step while keeping the gradient unbiased.
    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return (t, particlesF, loglik, weightsF, counts, key, t_idx)


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "process_weight_index",
    ),
)
def _dpop_internal_mean(
    theta: jax.Array,
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
    process_weight_index: int | None,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function returning the DPOP negative log-likelihood per
    observation.

    This is primarily useful as a differentiable objective for
    gradient-based optimization (e.g. via ``jax.grad``).
    """
    neg_ll = _dpop_internal(
        theta=theta,
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
        key=key,
    )
    return neg_ll / ys.shape[0]
