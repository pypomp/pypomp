import numpy as np
import jax
import jax.numpy as jnp
import warnings


def logmeanexp(x) -> float:
    """
    Calculates the mean likelihood for an array of log-likelihoods,
    and returns the corresponding log-likelihood. This is appropriate
    when the estimator is unbiased on the natural scale.

    Args:
        x (array-like): collection of log-likelihoods
    """
    x_array = np.asarray(x)
    if x_array.size == 0:
        warnings.warn("x is an empty array, returning nan")
        return np.nan
    x_max = np.max(x_array)
    log_mean_exp = np.log(np.mean(np.exp(x_array - x_max))) + x_max
    return log_mean_exp


def logmeanexp_se(x) -> float:
    """
    A jack-knife standard error for the log-likelihood estimate
    calculated via logmeanexp(). For comparison with R-pomp::logmeanexp,
    note that np.std divides by n whereas R-sd divides by (n-1), so
    np.var gives the Gaussian MLE and R-var gives the unbiased
    estimator.

    Args:
        x (array-like): collection of log-likelihoods
    """

    x_array = np.asarray(x, dtype=float)
    n = x_array.size
    if n <= 1:
        return np.nan

    jack = np.asarray(
        [logmeanexp(np.delete(x_array, i)) for i in range(n)],
        dtype=float,
    )
    se = np.sqrt(n - 1) * np.std(jack, ddof=0)
    return se


def logit(x: jax.Array | float) -> jax.Array:
    return jnp.log(x / (1 - x))


def expit(x: jax.Array | float) -> jax.Array:
    return 1 / (1 + jnp.exp(-x))
