"""
This module implements helper functions for POMP algorithms.
"""

import jax
import jax.numpy as jnp
import numpy as np


# TODO remove this function, as covars should always be dim 2
def _keys_helper(
    key: jax.Array, J: int, covars: jax.Array | None
) -> tuple[jax.Array, jax.Array]:
    """
    This function is a helper for generating random keys for resampling in the
    particle filtering algorithms.
    """
    if covars is not None and len(covars.shape) > 2:
        keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        res_keys = keys[1:].reshape(J, covars.shape[1], 2).astype(jnp.uint32)
    else:
        keys = jax.random.split(key, num=J + 1)
        res_keys = keys[1:]
    return keys[0], res_keys


def _resample(norm_weights: jax.Array, subkey: jax.Array) -> jax.Array:
    """
    Systematic resampling method based on input normalized weights.

    Args:
        norm_weights (array-like): The array containing the logarithm of
            normalized weights.
        subkey: The random key for sampling.

    Returns:
        array-like: An array containing the resampled indices from the
            systematic resampling given the input normalized weights.
    """
    J = norm_weights.shape[-1]
    unifs = (jax.random.uniform(key=subkey) + jnp.arange(J)) / J
    csum = jnp.cumsum(jnp.exp(norm_weights))
    counts = jnp.repeat(
        jnp.arange(J),
        jnp.histogram(
            unifs, bins=jnp.pad(csum / csum[-1], pad_width=(1, 0)), density=False
        )[0].astype(int),
        total_repeat_length=J,
    )
    return counts


def _normalize_weights(weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Acquires the normalized log-weights and calculates the log-likelihood.

    Args:
        weights (array-like): Logarithm of unnormalized weights.

    Returns:
        tuple: A tuple containing:
            - norm_weights (jax.Array): The normalized log-weights.
            - loglik_t (jax.Array): The log of the sum of all particle likelihoods,
                when the weights are associate with particles, which is
                equivalent to the total log-likelihood under the specific
                assumptions.
    """
    mw = jnp.max(weights)
    loglik_t = mw + jnp.log(jnp.nansum(jnp.exp(weights - mw)))
    norm_weights = weights - loglik_t
    return norm_weights, loglik_t


def _resampler(
    counts: jax.Array, particlesP: jax.Array, norm_weights: jax.Array, subkey: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Resamples the particles based on the weighted resampling rule determined by
    norm_weights and the original particles generated from the previous
    prediction.

    Args:
        counts (array-like): Indices of the resampled particles from a previous
            resampling procedure.
        particlesP (array-like): The original particles before resampling
            generated from a prediction procedure.
        norm_weights (array-like): The normalized log-weights of the particles.
        subkey: The random key for sampling.

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after
                the latest resampling.
            - particlesF (array-like): The particles after resampling generated
                from the filtering procedure.
            - norm_weights (array-like): The normalized log-weights of the
                resampled particles.
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights, subkey=subkey)
    particlesF = particlesP[counts]
    norm_weights = (
        norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - jnp.log(J)
    )
    return counts, particlesF, norm_weights


def _no_resampler(
    counts: jax.Array, particlesP: jax.Array, norm_weights: jax.Array, subkey: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Obtains the original input arguments without resampling.
    """
    return counts, particlesP, norm_weights


def _resampler_thetas(
    counts: jax.Array,
    particlesP: jax.Array,
    norm_weights: jax.Array,
    thetas: jax.Array,
    subkey: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Resamples the particles for perturbed particle filtering method, with their
    corresponding parameters also resampled

    Args:
        counts (array-like): Indices of the resampled particles from a previous
            resampling procedure.
        particlesP (array-like): The original particles before resampling
            generated from a prediction procedure.
        norm_weights (array-like): The normalized log-weights of the particles.
        thetas (array-like): Perturbed parameters associated with the particles.
        subkey: The random key for sampling.

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after
                the latest resampling.
            - particlesF (array-like): The particles after resampling generated
                from the filtering procedure.
            - norm_weights (array-like): The normalized log-weights of the
                resampled particles.
            - thetasF (array-like): The perturbed parameters corresponding to
                the latest perturbed particles (particlesF).
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights, subkey=subkey)
    particlesF = particlesP[counts]
    norm_weights = (
        norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - jnp.log(J)
    )
    thetasF = thetas[counts]
    return counts, particlesF, norm_weights, thetasF


def _no_resampler_thetas(
    counts: jax.Array,
    particlesP: jax.Array,
    norm_weights: jax.Array,
    thetas: jax.Array,
    subkey: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Obtains the original input arguments without resampling for perturbed
    filtering method.
    """
    return counts, particlesP, norm_weights, thetas


def _num_fixedstep_steps(
    t1: float, t2: float, dt: float | None, nstep: int | None
) -> tuple[int, float]:
    """
    Calculate the number of steps and the step size for a fixed number of steps.
    """
    assert nstep is not None
    return nstep, (t2 - t1) / nstep


def _num_euler_steps(
    t1: float, t2: float, dt: float | None, nstep: int | None
) -> tuple[int, float]:
    """
    Calculate the number of steps and the step size for a given time step size.
    """
    assert dt is not None
    tol = np.sqrt(np.finfo(float).eps)

    if t1 >= t2:
        return 0, 0.0

    if t1 + dt >= t2:
        return 1, t2 - t1

    nstep_val = int(np.ceil((t2 - t1) / dt / (1 + tol)))
    dt_val = (t2 - t1) / nstep_val

    return nstep_val, dt_val


def _calc_steps(
    times0: np.ndarray, dt: float | None, nstep: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the number of steps and the step size for each time interval.
    """
    if dt is not None and nstep is not None:
        raise ValueError("Only nstep or dt can be provided, not both")
    if dt is not None:
        num_step_func = _num_euler_steps
        dt = float(dt)
    elif nstep is not None:
        num_step_func = _num_fixedstep_steps
        nstep = int(nstep)
    else:
        raise ValueError("Either dt or nstep must be provided")

    nintervals = len(times0) - 1
    nstep_array = np.zeros(nintervals, dtype=int)
    dt_array = np.zeros(nintervals, dtype=float)
    for i in range(nintervals):
        nstep, dt = num_step_func(float(times0[i]), float(times0[i + 1]), dt, nstep)  # type: ignore
        nstep_array[i] = nstep
        dt_array[i] = dt
    return nstep_array, dt_array


def _interp_covars(
    t: float | np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    order: str = "linear",
) -> np.ndarray | None:
    """Interpolate covariates with numpy."""
    # TODO: Add constant interpolation
    if covars is None or ctimes is None:
        return None

    is_scalar = np.isscalar(t)
    t_arr = np.atleast_1d(t)

    upper_idx = np.searchsorted(ctimes, t_arr, side="left")
    upper_idx = np.clip(upper_idx, 1, len(ctimes) - 1)
    lower_idx = upper_idx - 1

    t_diff = (t_arr - ctimes[lower_idx]) / (ctimes[upper_idx] - ctimes[lower_idx])

    if covars.ndim > 1:
        t_diff = t_diff[:, None]

    interpolated = covars[lower_idx] + (covars[upper_idx] - covars[lower_idx]) * t_diff

    return interpolated.ravel() if is_scalar else interpolated


def _calc_interp_covars(
    times0: np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    nstep_array: np.ndarray,
    dt_array: np.ndarray,
    nintervals: int,
    order: str = "linear",
) -> np.ndarray | None:
    """Precompute the interpolated covariates for a given set of time points."""
    if covars is None or ctimes is None:
        return None

    time_steps = [
        times0[i] + np.arange(nstep_array[i]) * dt_array[i] for i in range(nintervals)
    ]

    all_times = np.concatenate(time_steps + [[times0[-1]]])

    return _interp_covars(all_times, ctimes, covars, order)


def _calc_ys_covars(
    t0: float,
    times: np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    dt: float | None,
    nstep: int | None,
    order: str = "linear",
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, int]:
    """Construct extended dt and covars arrays."""
    times0 = np.concatenate(([t0], times))
    nstep_array, dt_array = _calc_steps(times0, dt, nstep)

    interp_covars = _calc_interp_covars(
        times0, ctimes, covars, nstep_array, dt_array, len(nstep_array), order
    )

    dt_extended = np.repeat(dt_array, nstep_array)
    max_nstep = int(nstep_array.max()) if nstep_array.size else 0

    return interp_covars, dt_extended, nstep_array, max_nstep


def _geometric_cooling(nt: int, m: int, ntimes: int, a: float) -> float:
    """
    Calculate geometric cooling parameters for mif.

    Args:
        nt (int): Current time step, starting from 0.
        m (int): Current iteration, starting from 1.
        ntimes (int): Total number of time steps
        a (float): Amount to cool over 50 iterations

    Returns:
        float: The fraction to cool sigmas and sigmas_init by.
    """
    factor = a ** (1 / 50)
    alpha = factor ** (nt / ntimes + m - 1)
    return alpha


def _cosine_cooling(i: int, M: int, c: float) -> float | jax.Array:
    """
    Calculate cosine cooling parameters for train.

    Args:
        i (int): Current iteration index, starting from 0.
        M (int): Total number of iterations.
        c (float): Cooling factor (the factor by which the original value is multiplied by the end of the iterations).

    Returns:
        float: The fraction to cool by.
    """
    return c + (1.0 - c) * 0.5 * (1.0 + jnp.cos(jnp.pi * i / M))
