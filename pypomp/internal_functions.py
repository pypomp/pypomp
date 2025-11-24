"""
This module implements internal functions for POMP models.
"""

import numpy as np
import warnings
import jax
import jax.numpy as jnp


# TODO remove this function, as covars should always be dim 2
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the number of steps and the step size for a given time step size.
    """
    tol = np.sqrt(np.finfo(float).eps)

    nstep2 = np.ceil((t2 - t1) / dt / (1 + tol)).astype(int)
    dt2 = (t2 - t1) / nstep2

    check1 = t1 + dt >= t2
    nstep2 = np.where(check1, 1, nstep2)
    dt2 = np.where(check1, t2 - t1, dt2)

    check2 = t1 >= t2
    nstep2 = np.where(check2, 0, nstep2)
    dt2 = np.where(check2, 0.0, dt2)

    return nstep2, dt2


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


def _calc_ys_extended(
    ys: np.ndarray,
    nstep_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the exact values of the observations at the given time points.
    """
    n_obs, n_cols = ys.shape
    total_steps = np.sum(nstep_array)
    ys_extended = np.full((total_steps, n_cols), np.nan, dtype=float)
    ys_observed = np.full((total_steps,), False, dtype=bool)
    idx = 0
    for i in range(n_obs):
        idx += nstep_array[i]
        ys_extended[idx - 1] = ys[i]
        ys_observed[idx - 1] = True

    return ys_extended, ys_observed


def _interp_covars(
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


def _calc_interp_covars(
    times0: np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    nstep_array: np.ndarray,
    dt_array: np.ndarray,
    nintervals: int,
    order: str = "linear",
) -> np.ndarray | None:
    """
    Precompute the interpolated covariates for a given set of time points.

    Returns:
        np.ndarray | None: The interpolated covariates for a given set of time points.
            Shape is (times, max_nstep, ncovars). Returns None if covars or ctimes is
            None.
    """
    if covars is None or ctimes is None:
        return None

    # TODO: optimize this function
    total_steps = np.sum(nstep_array)
    interp_covars_array = np.full(
        (total_steps + 1, covars.shape[1]),
        fill_value=np.nan,
    )
    idx = 0
    for i in range(nintervals):
        for j in range(nstep_array[i]):
            interp_covars_array[idx, :] = _interp_covars(
                times0[i] + j * dt_array[i],
                ctimes,
                covars,
                order,
            )
            idx += 1
    interp_covars_array[idx, :] = _interp_covars(times0[-1], ctimes, covars, order)
    return interp_covars_array


def _calc_ys_covars(
    t0: float,
    times: np.ndarray,
    ys: np.ndarray,
    ctimes: np.ndarray | None,
    covars: np.ndarray | None,
    dt: float | None,
    nstep: int | None,
    order: str = "linear",
) -> tuple[jax.Array | None, jax.Array, jax.Array, int]:
    """
    Construct extended ys and covars arrays.
    """
    times0 = np.concatenate((np.array([t0]), times))
    nstep_array, dt_array = _calc_steps(times0, dt, nstep)
    nintervals = len(nstep_array)

    interp_covars_array = _calc_interp_covars(
        times0, ctimes, covars, nstep_array, dt_array, nintervals, order
    )

    # Deprecated: ys_extended and ys_observed computation removed

    dt_array_extended = np.repeat(dt_array, nstep_array)

    if covars is not None and ctimes is not None:
        assert interp_covars_array is not None
        assert interp_covars_array.shape[0] == dt_array_extended.shape[0] + 1

    return (
        None if interp_covars_array is None else jnp.array(interp_covars_array),
        jnp.array(dt_array_extended),
        jnp.array(nstep_array),
        int(nstep_array.max() if nstep_array.size > 0 else 0),
    )


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


def _shard_rows(array: jax.Array) -> jax.Array:
    """
    Evenly shard the rows of `array` across the available local JAX devices.
    When the row count is not divisible by the device count, the function issues
    a warning and distributes the remainder starting from the first device until
    there are no rows left.
    """
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError("No JAX devices available for sharding.")
    if array.ndim == 0:
        raise ValueError("Input array must have at least one dimension to shard rows.")

    n_rows = array.shape[0]
    n_devices = len(devices)
    rows_per_device = n_rows // n_devices
    remainder = n_rows % n_devices

    if remainder:
        warnings.warn(
            (
                f"Row count ({n_rows}) not divisible by device count ({n_devices}). "
                "Assigning one extra row to the first devices until the remainder is exhausted."
            ),
            stacklevel=2,
        )

    shards: list[jax.Array] = []
    start = 0
    for device_idx in range(n_devices):
        extra = 1 if device_idx < remainder else 0
        stop = start + rows_per_device + extra
        shards.append(array[start:stop])
        start = stop

    sharded = jax.device_put_sharded(shards, devices)
    return jnp.reshape(sharded, array.shape)
