"""
This module implements helper functions for POMP algorithms.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Callable
from functools import partial
from pypomp.functional.structs import PompStruct


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
    counts = jnp.searchsorted(csum / csum[-1], unifs, side="right")
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
    is_collapsed = mw == -jnp.inf
    mw_safe = jnp.where(is_collapsed, 0.0, mw)
    loglik_t = mw_safe + jnp.log(jnp.nansum(jnp.exp(weights - mw_safe)))
    loglik_t = jnp.where(is_collapsed, -jnp.inf, loglik_t)
    norm_weights = jnp.where(
        is_collapsed, -jnp.log(weights.shape[-1]), weights - loglik_t
    )
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
    counts = _resample(norm_weights, subkey=subkey).astype(counts.dtype)
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
    counts = _resample(norm_weights, subkey=subkey).astype(counts.dtype)
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
    if nstep is None:
        raise ValueError("nstep must be provided")
    return nstep, (t2 - t1) / nstep


def _num_euler_steps(
    t1: float, t2: float, dt: float | None, nstep: int | None
) -> tuple[int, float]:
    """
    Calculate the number of steps and the step size for a given time step size.
    """
    if dt is None:
        raise ValueError("dt must be provided")
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
    if covars is None or ctimes is None:
        return None

    if order not in ("linear", "constant"):
        raise ValueError(
            f"Unsupported interpolation order: '{order}'. Must be 'linear' or 'constant'."
        )

    is_scalar = np.isscalar(t)
    t_arr = np.atleast_1d(t)

    if order == "linear":
        upper_idx = np.searchsorted(ctimes, t_arr, side="left")
        upper_idx = np.clip(upper_idx, 1, len(ctimes) - 1)
        lower_idx = upper_idx - 1

        t_diff = (t_arr - ctimes[lower_idx]) / (ctimes[upper_idx] - ctimes[lower_idx])

        if covars.ndim > 1:
            t_diff = t_diff[:, None]

        interpolated = (
            covars[lower_idx] + (covars[upper_idx] - covars[lower_idx]) * t_diff
        )
    else:  # constant
        upper_idx = np.searchsorted(ctimes, t_arr, side="right")
        idx = np.clip(upper_idx, 1, len(ctimes)) - 1
        interpolated = covars[idx]

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


def is_dynamic(val: Any) -> bool:
    """Helper to detect if a value is/contains a JAX array or PyTree structure."""
    try:
        leaves = jax.tree_util.tree_leaves(val)
        return any(isinstance(x, (jax.Array, np.ndarray, PompStruct)) for x in leaves)
    except Exception:
        if isinstance(val, (jax.Array, np.ndarray, PompStruct)):
            return True
        if isinstance(val, (list, tuple)):
            return any(is_dynamic(x) for x in val)
        if isinstance(val, dict):
            return any(is_dynamic(x) for x in val.values())
        return False


@partial(
    jax.jit,
    static_argnames=(
        "func_static",
        "shard_axes_static",
        "static_idxs_static",
        "static_args_static",
        "static_kwargs_static",
        "dynamic_idxs_static",
    ),
)
def _scan_jit(
    func_static: Callable[..., Any],
    shard_axes_static: tuple[tuple[int, int], ...],
    static_idxs_static: tuple[int, ...],
    static_args_static: tuple[Any, ...],
    static_kwargs_static: tuple[tuple[str, Any], ...],
    dynamic_idxs_static: tuple[int, ...],
    sharded_inputs: dict[int, jax.Array],
    *dynamic_args: Any,
    **dynamic_kwargs: Any,
) -> Any:
    """JIT-compiled scan wrapper to run the batches sequentially inside XLA."""

    def step_fn(carry, step_inputs):
        total_args_count = (
            len(shard_axes_static) + len(static_idxs_static) + len(dynamic_idxs_static)
        )
        current_args = [None] * total_args_count

        # 1. Fill sharded inputs
        for arg_idx, sliced in step_inputs.items():
            orig_axis = dict(shard_axes_static)[arg_idx]
            restored = jnp.moveaxis(sliced, 0, orig_axis)
            current_args[arg_idx] = restored

        # 2. Fill static args
        for idx, val in zip(static_idxs_static, static_args_static):
            current_args[idx] = val

        # 3. Fill dynamic args
        for idx, val in zip(dynamic_idxs_static, dynamic_args):
            current_args[idx] = val

        # 4. Reconstruct kwargs
        current_kwargs = dict(dynamic_kwargs)
        current_kwargs.update(dict(static_kwargs_static))

        out = func_static(*current_args, **current_kwargs)
        return carry, out

    _, scan_outputs = jax.lax.scan(step_fn, None, sharded_inputs)
    return scan_outputs


def merge_and_slice(
    arr: Any, out_axis: int | None, size: int, num_batches: int, batch_size: int
) -> Any:
    """Merges scan batches back into the original shape and slices away padding."""
    if out_axis is None:
        if isinstance(arr, (jax.Array, np.ndarray)):
            if arr.ndim == 0:
                return arr
            # For un-sharded outputs, take the first batch's value if it was scanned
            if num_batches > 1:
                return arr[0]
            return arr
        return arr

    if not isinstance(arr, (jax.Array, np.ndarray)):
        return arr

    if arr.ndim == 0:
        return arr

    if num_batches == 1:
        # No need to reshape/merge, just slice the padded dimension
        slice_obj = [slice(None)] * arr.ndim
        slice_obj[out_axis] = slice(0, size)
        return arr[tuple(slice_obj)]

    # 1. Move scan batch axis to the sharded axis position
    arr_moved = jnp.moveaxis(arr, 0, out_axis)
    shape = list(arr_moved.shape)

    # 2. Reshape to combine the scan batch dimension and batch size dimension
    new_shape = shape[:out_axis] + [num_batches * batch_size] + shape[out_axis + 2 :]
    arr_merged = arr_moved.reshape(new_shape)

    # 3. Slice to original unpadded size
    slice_obj = [slice(None)] * arr_merged.ndim
    slice_obj[out_axis] = slice(0, size)
    return arr_merged[tuple(slice_obj)]


def merge_outputs(
    scanned_out: Any,
    shard_output_axes: Any,
    size: int,
    num_batches: int,
    batch_size: int,
) -> Any:
    """Recursively processes output PyTree to merge and slice scanned batches."""
    if isinstance(shard_output_axes, int) or shard_output_axes is None:
        return jax.tree_util.tree_map(
            lambda x: merge_and_slice(
                x, shard_output_axes, size, num_batches, batch_size
            ),
            scanned_out,
        )
    elif isinstance(shard_output_axes, (list, tuple)):
        merged = []
        for i, axis in enumerate(shard_output_axes):
            merged.append(
                merge_and_slice(scanned_out[i], axis, size, num_batches, batch_size)
            )
        return tuple(merged) if isinstance(shard_output_axes, tuple) else merged
    elif isinstance(shard_output_axes, dict):
        merged_dict = {}
        for key, val in scanned_out.items():
            axis = shard_output_axes.get(key, None)
            merged_dict[key] = merge_and_slice(val, axis, size, num_batches, batch_size)
        return merged_dict
    raise TypeError(f"Unsupported shard_output_axes type: {type(shard_output_axes)}")


def pad_array(arr: jax.Array, axis: int, padded_size: int, size: int) -> jax.Array:
    """Pads an array along a given axis to padded_size by repeating the last element."""
    pad_width = padded_size - size
    if pad_width > 0:
        slice_obj = [slice(None)] * arr.ndim
        slice_obj[axis] = slice(-1, None)
        last_element = arr[tuple(slice_obj)]

        repeats = [1] * arr.ndim
        repeats[axis] = pad_width
        padded_slice = jnp.tile(last_element, repeats)
        return jnp.concatenate([arr, padded_slice], axis=axis)
    return arr


def run_jax_batch_sharded(
    func: Callable[..., Any],
    shard_axes: dict[int, int],
    shard_output_axes: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Executes a JAX function in sharded parallel batches or direct SPMD sharding.

    Depending on the device type and replicate dimension size, the execution will
    follow one of two paths:

    1. **CPU Sequential Batching Path** (Running on CPU and replicate size R > device count C):
       - If the device kind is 'cpu' and the size of the sharded axis exceeds the number of
         available CPU devices (cores), the execution is split into sequential batches.
       - The sharded arguments are padded to a size that is the next multiple of the CPU core
         count (i.e. padded_size = num_batches * C) by repeating the last element of each array
         along the sharded axis using :func:`pad_array`.
       - The padded arrays are reshaped from (padded_size, ...) to (num_batches, C, ...).
       - A JIT-compiled `jax.lax.scan` loop runs sequentially over the `num_batches`.
       - For each batch step of the scan, the batch of size C is sharded across the C CPU cores
         using `NamedSharding` (PartitionSpec specifying the batch axis 1 to map to 'devices').
       - This ensures that each CPU core processes exactly one replicate at a time, avoiding core
         juggling/context switching and preserving L1/L2 cache locality.
       - Finally, outputs from the scan are merged, and the extra padded replicates are sliced
         off to restore the original size R.

    2. **Direct Parallel SPMD Sharding Path** (GPU/TPU, or CPU when R <= C):
       - If running on GPU/TPU, or on CPU when the replicate size R is less than or equal to the
         device count D, the replicates are executed in a single parallel step.
       - The sharded arguments are padded to the next multiple of the total device count D (i.e.
         padded_size = ceil(R / D) * D) by repeating the last element along the sharded axis.
       - The padded arguments are sharded directly across all D devices using `NamedSharding`
         (PartitionSpec mapping the sharded axis to 'devices').
       - The function is called directly on these sharded inputs, executing completely in parallel.
       - The outputs are merged and sliced back to the original size R.

    Args:
        func: The functional JAX algorithm (e.g., F.pfilter, F.mif, or a custom function).
        shard_axes: Dictionary mapping positional argument indices (in `args`) to their sharded axis index.
        shard_output_axes: Structure mapping output PyTree leaves to their sharded axes (int, list, tuple, dict, or None).
                           If a leaf output axis is None, it indicates the leaf is a global metric/scalar and not sharded.
        *args: Positional arguments for `func`.
        **kwargs: Keyword arguments for `func`.

    Returns:
        The outputs of `func`, merged and sliced back to the original replicate count.

    Examples:
        **Example 1: Setting up multi-device CPU parallelization**
        Configure the environment before importing JAX to simulate a multi-core CPU cluster:

        >>> import os
        >>> os.environ["JAX_PLATFORMS"] = "cpu"
        >>> os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        >>>
        >>> import jax
        >>> import pypomp as pp
        >>>
        >>> LG = pp.models.LG()
        >>> # If we run 10 replicates, run_jax_batch_sharded executes them in parallel
        >>> # batches of 4 (i.e., batches of [4, 4, 4] where the last 2 replicates are padded).
        >>> LG.pfilter(J=1000, reps=10)

        **Example 2: Direct call to run_jax_batch_sharded**
        We can run a custom sharded function across devices. Here is a simple example using 4 CPU cores:

        >>> import os
        >>> os.environ["JAX_PLATFORMS"] = "cpu"
        >>> os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from pypomp.core.algorithms.helpers import run_jax_batch_sharded
        >>>
        >>> # Define a simple function that adds a constant to a sharded input array
        >>> def add_const(x, c):
        ...     return x + c
        >>>
        >>> # Input data: 10 replicates, sharded along axis 0
        >>> x = jnp.arange(10.0)
        >>> const = 5.0
        >>>
        >>> # Call the helper. Shard axis of the 1st positional argument (index 0) is 0.
        >>> # The output is also sharded along axis 0.
        >>> result = run_jax_batch_sharded(
        ...     add_const,
        ...     shard_axes={0: 0},
        ...     shard_output_axes=0,
        ...     x,
        ...     const
        ... )
        >>> print(result)
        [ 5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
    """
    devices = jax.devices()
    num_devices = len(devices)
    device_kind = devices[0].device_kind.lower()

    # Get size of the sharded dimension from the first sharded argument
    first_arg_idx = list(shard_axes.keys())[0]
    first_axis = shard_axes[first_arg_idx]
    size = args[first_arg_idx].shape[first_axis]

    if device_kind == "cpu" and size > num_devices:
        batch_size = num_devices
        num_batches = (size + batch_size - 1) // batch_size
    else:
        batch_size = ((size + num_devices - 1) // num_devices) * num_devices
        num_batches = 1
    padded_size = num_batches * batch_size

    padded_args = list(args)
    for arg_idx, axis in shard_axes.items():
        padded_args[arg_idx] = pad_array(args[arg_idx], axis, padded_size, size)

    mesh = jax.sharding.Mesh(devices, axis_names=("devices",))

    # CPU Sequential Batching Path
    if num_batches > 1:
        sharded_inputs = {}
        for arg_idx, axis in shard_axes.items():
            arr = padded_args[arg_idx]
            arr_trans = jnp.moveaxis(arr, axis, 0)
            reshaped_trans = arr_trans.reshape(
                num_batches, batch_size, *arr_trans.shape[1:]
            )

            spec_list: list[str | None] = [None] * reshaped_trans.ndim
            spec_list[1] = "devices"
            spec = jax.sharding.PartitionSpec(*spec_list)
            sharding_spec = jax.sharding.NamedSharding(mesh, spec)
            sharded_inputs[arg_idx] = jax.device_put(reshaped_trans, sharding_spec)

        dynamic_args = []
        static_args = []
        dynamic_idxs = []
        static_idxs = []

        for i, arg in enumerate(padded_args):
            if i in shard_axes:
                continue
            if is_dynamic(arg):
                dynamic_args.append(arg)
                dynamic_idxs.append(i)
            else:
                static_args.append(arg)
                static_idxs.append(i)

        dynamic_kwargs = {}
        static_kwargs = {}
        for k, v in kwargs.items():
            if is_dynamic(v):
                dynamic_kwargs[k] = v
            else:
                static_kwargs[k] = v

        shard_axes_static = tuple(shard_axes.items())
        static_idxs_static = tuple(static_idxs)
        static_args_static = tuple(static_args)
        static_kwargs_static = tuple(static_kwargs.items())
        dynamic_idxs_static = tuple(dynamic_idxs)

        scanned_outputs = _scan_jit(
            func,
            shard_axes_static,
            static_idxs_static,
            static_args_static,
            static_kwargs_static,
            dynamic_idxs_static,
            sharded_inputs,
            *dynamic_args,
            **dynamic_kwargs,
        )

        return merge_outputs(
            scanned_outputs, shard_output_axes, size, num_batches, batch_size
        )

    # Parallel Path (GPU/TPU or CPU with size <= num_devices)
    else:
        sharded_args = list(padded_args)
        for arg_idx, axis in shard_axes.items():
            arr = padded_args[arg_idx]
            spec_list: list[str | None] = [None] * arr.ndim
            spec_list[axis] = "devices"
            spec = jax.sharding.PartitionSpec(*spec_list)
            sharding_spec = jax.sharding.NamedSharding(mesh, spec)
            sharded_args[arg_idx] = jax.device_put(arr, sharding_spec)

        outputs = func(*sharded_args, **kwargs)
        return merge_outputs(outputs, shard_output_axes, size, 1, padded_size)
