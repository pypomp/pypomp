"""
This module implements internal functions for POMP models.
"""

import numpy as np
import warnings
import jax
import jax.numpy as jnp
from typing import Callable


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


def _chunked_vmap(fun: Callable, chunk_size: int) -> Callable:
    """
    A wrapper that acts like vmap, but processes the input in sequential chunks
    to avoid OOM errors.

    Args:
        fun: The function to be vectorized.
        chunk_size: The size of the batch chunks to process sequentially.
    """
    # 1. Create the standard vmapped function for a single chunk
    vmapped_fun = jax.vmap(fun)

    @jax.jit
    def wrapped(*args):
        # Helper to reshape any array: (N, ...) -> (N // chunk, chunk, ...)
        def reshape_input(x):
            # We assume the first dimension is the batch dimension
            n = x.shape[0]
            if n % chunk_size != 0:
                raise ValueError(
                    f"Batch size {n} must be divisible by chunk size {chunk_size}"
                )
            return x.reshape((n // chunk_size, chunk_size, *x.shape[1:]))

        # Helper to flatten output: (N // chunk, chunk, ...) -> (N, ...)
        def flatten_output(x):
            return x.reshape((-1, *x.shape[2:]))

        # 2. Reshape all inputs (supports PyTrees/Multiple Args automatically)
        chunked_inputs = jax.tree_util.tree_map(reshape_input, args)

        # 3. Use lax.map to loop over the first dimension (the chunks)
        # lax.map passes a tuple of args to the lambda if args is a tuple
        if len(args) == 1:
            chunked_outputs = jax.lax.map(vmapped_fun, chunked_inputs[0])
        else:
            # If multiple args, lax.map passes them as a tuple, so we unpack
            chunked_outputs = jax.lax.map(lambda x: vmapped_fun(*x), chunked_inputs)

        # 4. Flatten the results back to the original shape
        return jax.tree_util.tree_map(flatten_output, chunked_outputs)

    return wrapped
