from typing import Any, overload
import numpy as np
import jax
import jax.numpy as jnp
import warnings


@overload
def logmeanexp(x: Any, axis: None = None, ignore_nan: bool = False) -> float: ...


@overload
def logmeanexp(
    x: Any, axis: int | tuple[int, ...], ignore_nan: bool = False
) -> np.ndarray: ...


def logmeanexp(
    x, axis: int | tuple[int, ...] | None = None, ignore_nan: bool = False
) -> Any:
    """
    Calculates the mean likelihood for an array of log-likelihoods,
    and returns the corresponding log-likelihood. This is appropriate
    when the estimator is unbiased on the natural scale.

    Args:
        x (array-like): collection of log-likelihoods
        axis (int, tuple, or None): axis or axes along which to compute the mean.
            If None (default), compute over the entire array.
        ignore_nan (bool): if True, drop NaNs (or treat as -inf in exp space) before computing.
    """
    x_array = np.asarray(x, dtype=float)

    if ignore_nan:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            x_max = np.nanmax(x_array, axis=axis, keepdims=True)

        mask = np.isnan(x_array)
        x_safe = np.where(mask, -np.inf, x_array)
        counts = np.sum(~mask, axis=axis, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_exp = np.sum(np.exp(x_safe - x_max), axis=axis, keepdims=True) / counts
            res = np.log(mean_exp) + x_max
    else:
        if axis is None and x_array.size == 0:
            warnings.warn("x is an empty array, returning nan")
            return np.nan

        x_max = np.max(x_array, axis=axis, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = (
                np.log(np.mean(np.exp(x_array - x_max), axis=axis, keepdims=True))
                + x_max
            )

    if axis is None:
        return res.item()

    return np.squeeze(res, axis=axis)


@overload
def logmeanexp_se(x: Any, axis: None = None, ignore_nan: bool = False) -> float: ...


@overload
def logmeanexp_se(x: Any, axis: int, ignore_nan: bool = False) -> np.ndarray: ...


def logmeanexp_se(x, axis: int | None = None, ignore_nan: bool = False) -> Any:
    """
    A jack-knife standard error for the log-likelihood estimate
    calculated via logmeanexp(). For comparison with R-pomp::logmeanexp,
    note that np.std divides by n whereas R-sd divides by (n-1), so
    np.var gives the Gaussian MLE and R-var gives the unbiased
    estimator.

    Args:
        x (array-like): collection of log-likelihoods
        axis (int or None): axis along which to compute the SE.
            If None (default), compute over the entire array.
        ignore_nan (bool): if True, drop NaNs before computing.
    """
    if axis is not None:
        return np.apply_along_axis(logmeanexp_se, axis, x, ignore_nan=ignore_nan)

    x_array = np.asarray(x, dtype=float)
    if ignore_nan:
        x_array = x_array[~np.isnan(x_array)]

    n = x_array.size
    if n <= 1:
        return np.nan

    jack = np.asarray(
        [logmeanexp(np.delete(x_array, i), ignore_nan=False) for i in range(n)],
        dtype=float,
    )
    se = np.sqrt(n - 1) * np.std(jack, ddof=0)
    return float(se)


def logit(x: jax.Array | float) -> jax.Array:
    return jnp.log(x / (1 - x))


def expit(x: jax.Array | float) -> jax.Array:
    return 1 / (1 + jnp.exp(-x))
