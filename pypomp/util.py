import jax.numpy as jnp


def logmeanexp(x):
    """
    Calculates the mean likelihood for an array of log-likelihoods,
    and returns the corresponding log-likelihood. This is appropriate
    when the estimator is unbiased on the natural scale.

    Args:
        x (array-like): collection of log-likelihoods
    """
    x_array = jnp.array(x)
    x_max = jnp.max(x_array)
    log_mean_exp = jnp.log(jnp.mean(jnp.exp(x_array - x_max))) + x_max
    return log_mean_exp


def logmeanexp_se(x):
    """
    A jack-knife standard error for the log-likelihood estimate
    calculated via logmeanexp(). For comparison with R-pomp::logmeanexp,
    note that jnp.std divides by n whereas R-sd divides by (n-1), so
    jnp.var gives the Gaussian MLE and R-var gives the unbiased
    estimator.

    Args:
        x (array-like): collection of log-likelihoods
    """

    jack = jnp.zeros(len(x))
    for i in range(len(x)):
        jack = jack.at[i].set(logmeanexp(jnp.delete(x, i)))
    se = jnp.sqrt(len(jack) - 1) * jnp.std(jack)
    return se
