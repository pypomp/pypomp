"""Log-weight increments for DPOP process models.

Each function returns a surrogate for the log-probability of a draw, with the
same gradient in the distribution's parameters, for ``rproc`` to add to its
``_logw`` state.  Terms that do not depend on the parameters are dropped, so
the value is not the log-probability itself.  The draw and the number of
trials are held fixed with :func:`jax.lax.stop_gradient`.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

# Floor on probabilities and rates before taking logs, so that a zero
# probability gives a finite value and a zero gradient.
_FLOOR = 1.0e-12


def _log(p: jax.Array | float) -> jax.Array:
    return jnp.log(jnp.maximum(p, _FLOOR))


def poisson_logw(x: jax.Array, lam: jax.Array | float) -> jax.Array:
    """DPOP log-weight of a Poisson draw.

    Parameters
    ----------
    x : jax.Array
        The Poisson count.
    lam : jax.Array or float
        The Poisson mean.

    Returns
    -------
    jax.Array
        ``x * log(lam) - lam``, with ``x`` held fixed.
    """
    x = jax.lax.stop_gradient(x)
    return x * _log(lam) - lam


def binomial_logw(
    x: jax.Array, n: jax.Array | float, p: jax.Array | float
) -> jax.Array:
    """DPOP log-weight of a binomial draw.

    Parameters
    ----------
    x : jax.Array
        The number of successes.
    n : jax.Array or float
        The number of trials.
    p : jax.Array or float
        The success probability.

    Returns
    -------
    jax.Array
        ``x * log(p) + (n - x) * log(1 - p)``, with ``x`` and ``n`` held fixed.
    """
    x = jax.lax.stop_gradient(x)
    n = jax.lax.stop_gradient(n)
    return x * _log(p) + (n - x) * _log(1.0 - p)


def euler_multinomial_logw(
    x: jax.Array | Sequence[jax.Array],
    n: jax.Array | float,
    rates: jax.Array | Sequence[jax.Array | float],
    dt: jax.Array | float,
) -> jax.Array:
    """DPOP log-weight of an Euler-multinomial draw.

    Over a step of length ``dt``, each of ``n`` individuals leaves via event
    ``k`` with probability ``rates[k] / sum(rates) * (1 - exp(-sum(rates) * dt))``
    and otherwise stays, as in pomp's ``reulermultinom``.

    Parameters
    ----------
    x : jax.Array or sequence of jax.Array
        The ``K`` event counts, excluding those who stay: a sequence of ``K``
        arrays, or an array whose leading axis has length ``K``.
    n : jax.Array or float
        The number of individuals.
    rates : jax.Array or sequence
        The ``K`` event rates, in the same layout as ``x``.
    dt : jax.Array or float
        The step length.

    Returns
    -------
    jax.Array
        The multinomial log-pmf without its normalizing constant, with ``x``
        and ``n`` held fixed.
    """
    xs = [jax.lax.stop_gradient(xk) for xk in x]
    n = jax.lax.stop_gradient(n)
    rates = list(rates)
    r_sum = sum(rates)
    # P(leave via k) = rates[k] * scale; expm1 keeps scale accurate when
    # r_sum * dt is small.
    scale = -jnp.expm1(-r_sum * dt) / jnp.maximum(r_sum, _FLOOR)
    logw = (n - sum(xs)) * (-r_sum * dt)
    for xk, rk in zip(xs, rates, strict=True):
        logw = logw + xk * _log(rk * scale)
    return jnp.asarray(logw)
