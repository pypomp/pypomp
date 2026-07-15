"""
Numerical utilities.
"""

from typing import Any, overload
import numpy as np
import warnings
from jax.scipy.special import logit, expit


__all__ = ["logmeanexp", "logmeanexp_se", "logit", "expit"]


@overload
def logmeanexp(x: Any, axis: None = None, ignore_nan: bool = False) -> float: ...


@overload
def logmeanexp(
    x: Any, axis: int | tuple[int, ...], ignore_nan: bool = False
) -> np.ndarray: ...


def logmeanexp(
    x, axis: int | tuple[int, ...] | None = None, ignore_nan: bool = False
) -> Any:
    """Compute the log of the mean likelihood from log-likelihood values.

    Calculates ``log(mean(exp(x)))`` in a numerically stable way using the
    log-sum-exp trick.  This is appropriate when the estimator is unbiased
    on the natural (probability) scale, e.g. for averaging particle filter
    log-likelihood estimates across replicates.

    Parameters
    ----------
    x : array-like
        Collection of log-likelihood values.
    axis : int, tuple of int, or None, optional
        Axis or axes along which to compute the mean.  If ``None``
        (default), compute over the entire array.
    ignore_nan : bool, optional
        If ``True``, treat NaN entries as ``-inf`` (i.e. zero probability)
        before computing.  Defaults to ``False``.

    Returns
    -------
    float or np.ndarray
        The log-mean-exp value.  A scalar ``float`` when ``axis=None``,
        otherwise a ``numpy.ndarray`` with the reduced dimension removed.

    See Also
    --------
    logmeanexp_se : Jackknife standard error for this estimator.
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
    """Compute a jackknife standard error for the :func:`logmeanexp` estimator.

    Estimates the standard error of the log-likelihood estimate produced by
    :func:`logmeanexp` using the jackknife (leave-one-out) method.

    .. note::

        ``numpy.std`` divides by ``n`` (MLE), whereas R's ``sd`` divides by
        ``n - 1`` (unbiased). This function matches the NumPy convention, so
        results will differ slightly from R's ``pomp::logmeanexp`` SE output.

    Parameters
    ----------
    x : array-like
        Collection of log-likelihood values.
    axis : int or None, optional
        Axis along which to compute the SE.  If ``None`` (default), compute
        over the entire array.
    ignore_nan : bool, optional
        If ``True``, remove NaN entries before computing.  Defaults to
        ``False``.

    Returns
    -------
    float or np.ndarray
        The jackknife standard error.  ``np.nan`` if fewer than 2 values
        are present.

    See Also
    --------
    logmeanexp : The estimator whose SE this computes.
    """
    if axis is not None:
        return np.apply_along_axis(logmeanexp_se, axis, x, ignore_nan=ignore_nan)

    x_array = np.asarray(x, dtype=float)
    if ignore_nan:
        x_array = x_array[~np.isnan(x_array)]

    n = x_array.size
    if n <= 1:
        return np.nan

    x_max = np.max(x_array)
    s = np.exp(x_array - x_max)
    S = np.sum(s)

    with np.errstate(divide="ignore", invalid="ignore"):
        jack = np.log((S - s) / (n - 1)) + x_max

    # Handle numerical stability if the max is unique and dominant
    is_max = x_array == x_max
    if np.sum(is_max) == 1:
        idx_max = np.argmax(x_array)
        # S - s[idx_max] might be zero due to underflow
        # If so, re-calculate this single jackknife sample accurately
        if S - s[idx_max] <= 0:
            subset = np.delete(x_array, idx_max)
            jack[idx_max] = logmeanexp(subset, ignore_nan=False)

    se = np.sqrt(n - 1) * np.std(jack, ddof=0)
    return float(se)
