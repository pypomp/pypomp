"""
This module implements internal functions for POMP models.
"""

import numpy as np
import jax
import jax.numpy as jnp


def _keys_helper(
    key: jax.Array, J: int, covars: jax.Array | None
) -> tuple[jax.Array, jax.Array]:
    """
    This function is a helper for generating random keys for resampling in the
    particle filtering algorithms.
    """
    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = jnp.array(keys).reshape(J, covars.shape[1], 2).astype(jnp.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = jnp.array(keys)
    return key, keys


def _resample(norm_weights: jax.Array, subkey: jax.Array) -> jax.Array:
    """
    Systematic resampling method based on input normalized weights.

    Args:
        norm_weights (array-like): The array containing the logarithm of
            normalized weights.
        subkey (jax.random.PRNGKey): The random key for sampling.

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
        subkey (jax.random.PRNGKey): The random key for sampling.

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
        subkey (jax.random.PRNGKey): The random key for sampling.

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
    t1: float, t2: float, dt: float | None, nstep: int
) -> tuple[int, float]:
    """
    Calculate the number of steps and the step size for a fixed number of steps.
    """
    return nstep, (t2 - t1) / nstep


def _num_euler_steps(
    t1: float, t2: float, dt: float, nstep: int | None
) -> tuple[jax.Array, jax.Array]:
    """
    Calculate the number of steps and the step size for a given time step size.
    """
    tol = jnp.sqrt(jnp.finfo(float).eps)

    nstep2 = jnp.ceil((t2 - t1) / dt / (1 + tol)).astype(int)
    dt2 = (t2 - t1) / nstep2

    check1 = t1 + dt >= t2
    nstep2 = jnp.where(check1, 1, nstep2)
    dt2 = jnp.where(check1, t2 - t1, dt2)

    check2 = t1 >= t2
    nstep2 = jnp.where(check2, 0, nstep2)
    dt2 = jnp.where(check2, 0.0, dt2)

    return nstep2, dt2


def _interp_covars(
    t: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    order: str = "linear",
) -> jax.Array | None:
    """
    Interpolate covariates.

    Args:
        t (float | jax.Array): Time point at which to interpolate the covariates.
            Can be a single float or an array containing one time point.
        ctimes (jax.Array | None): Array of time points for which covariates are
            available. Must be sorted in ascending order.
        covars (jax.Array | None): Array of covariate values corresponding to ctimes.
            Shape should be (len(ctimes), n_covars) where n_covars is the number of covariates.
        order (str, optional): Interpolation method. Currently only 'linear' is
            supported. Defaults to 'linear'.

    Returns:
        jax.Array | None: The interpolated covariates at time t. Returns None if either
            ctimes or covars is None. Shape is (n_covars,).

    Note:
        This function assumes that ctimes is sorted in ascending order. If t is outside
        the range of ctimes, the nearest available covariate value will be used.
    """
    # TODO: Add constant interpolation
    if (covars is None) | (ctimes is None):
        return None
    else:
        assert ctimes is not None
        assert covars is not None
        upper_index = jnp.searchsorted(ctimes, t, side="left")
        lower_index = upper_index - 1
        return (
            covars[lower_index]
            + (covars[upper_index] - covars[lower_index])
            * (t - ctimes[lower_index])
            / (ctimes[upper_index] - ctimes[lower_index])
        ).ravel()


def _interp_covars_np(
    t: float | np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    order: str = "linear",
) -> np.ndarray | None:
    """
    Interpolate covariates with numpy.
    """
    # TODO: Add constant interpolation
    if (covars is None) | (ctimes is None):
        return None
    else:
        assert ctimes is not None
        assert covars is not None
        upper_index = np.searchsorted(ctimes, t, side="left")
        lower_index = upper_index - 1
        return (
            covars[lower_index]
            + (covars[upper_index] - covars[lower_index])
            * (t - ctimes[lower_index])
            / (ctimes[upper_index] - ctimes[lower_index])
        ).ravel()


def _precompute_interp_covars(
    t0: float,
    times: np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    dt: float | None,
    nstep: int | None,
    order: str = "linear",
) -> np.ndarray | None:
    """
    Precompute the interpolated covariates for a given set of time points.

    Returns:
        np.ndarray | None: The interpolated covariates for a given set of time points.
            Shape is (ntimes, max_nstep, ncovars). Returns None if covars or ctimes is
            None.
    """
    if covars is None or ctimes is None:
        return None

    # TODO: optimize this function
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

    times0 = np.concatenate((np.array([t0]), times))
    nintervals = len(times0) - 1
    nstep_array = np.zeros(nintervals, dtype=int)
    dt_array = np.zeros(nintervals, dtype=float)
    for i in range(nintervals):
        nstep, dt = num_step_func(float(times0[i]), float(times0[i + 1]), dt, nstep)  # type: ignore
        nstep_array[i] = nstep
        dt_array[i] = dt
    interp_covars_array = np.full(
        (nintervals, int(np.max(nstep_array)), covars.shape[1]),
        fill_value=np.nan,
    )
    for i in range(interp_covars_array.shape[0]):
        for j in range(interp_covars_array.shape[1]):
            interp_covars_array[i, j, :] = _interp_covars_np(
                times0[i] + j * dt_array[i],
                ctimes,
                covars,
                order,
            )
    return interp_covars_array


def _geometric_cooling(nt: int, m: int, ntimes: int, a: float) -> float:
    """
    Calculate geometric cooling parameters for mif.

    Args:
        nt (int): Current time step
        m (int): Current iteration
        ntimes (int): Total number of time steps
        a (float): Amount to cool over 50 iterations

    Returns:
        float: The fraction to cool sigmas and sigmas_init by.
    """
    factor = a ** (1 / 50)
    alpha = factor ** (nt / ntimes + m - 1)
    return alpha
