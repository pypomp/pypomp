import numpy as np


def logmeanexp(x):
    """
    Calculates the mean likelihood for an array of log-likelihoods,
    and returns the corresponding log-likelihood. This is appropriate
    when the estimator is unbiased on the natural scale.

    Args:
        x (array-like): collection of log-likelihoods
    """
    x_array = np.asarray(x)
    x_max = np.max(x_array)
    log_mean_exp = np.log(np.mean(np.exp(x_array - x_max))) + x_max
    return log_mean_exp


def logmeanexp_se(x):
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
    n = x_array.shape[0]
    if n <= 1:
        return np.nan

    x_max = np.max(x_array)
    exps = np.exp(x_array - x_max)
    sum_exp = np.sum(exps)
    loo_sum = sum_exp - exps  # leave-one-out sums
    loo_mean = loo_sum / (n - 1)
    with np.errstate(divide="ignore"):
        jack = np.log(loo_mean) + x_max
    se = np.sqrt(n - 1) * np.std(jack, ddof=0)
    return se
