import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy import stats

import pypomp.random as ppr

DT = 1.0 / 52


def _euler_multinomial_logpmf(x, n, rates, dt):
    """Full Euler-multinomial log-pmf, as a reference."""
    r_sum = jnp.sum(rates)
    p_stay = jnp.exp(-r_sum * dt)
    probs = jnp.concatenate([p_stay[None], rates / r_sum * (1.0 - p_stay)])
    counts = jnp.concatenate([(n - jnp.sum(x))[None], x])
    return stats.multinomial.logpmf(counts.astype(int), n.astype(int), probs)


@pytest.mark.parametrize(
    "logw, logpmf, x, params",
    [
        (ppr.poisson_logw, stats.poisson.logpmf, 7.0, (12.5,)),
        (ppr.binomial_logw, stats.binom.logpmf, 7.0, (30.0, 0.2)),
    ],
    ids=["poisson", "binomial"],
)
def test_logw_gradient_matches_logpmf(logw, logpmf, x, params):
    """The surrogate has the log-pmf's gradient in the last parameter."""
    *fixed, theta = params

    def f(fn):
        return jax.grad(lambda th: fn(x, *fixed, th))(theta)

    np.testing.assert_allclose(f(logw), f(logpmf), rtol=1e-5)


def test_euler_multinomial_logw_gradient_matches_logpmf():
    x = jnp.array([3.0, 1.0])
    n = jnp.array(40.0)
    rates = jnp.array([2.5, 0.7])

    got = jax.grad(lambda r: ppr.euler_multinomial_logw(x, n, r, DT))(rates)
    want = jax.grad(lambda r: _euler_multinomial_logpmf(x, n, r, DT))(rates)

    np.testing.assert_allclose(got, want, rtol=1e-4)


def test_multinomial_logw_gradient_matches_logpmf():
    """The probabilities are normalized, as in fast_multinomial, so unnormalized
    weights give the gradient of the log-pmf at the normalized probabilities."""
    x = jnp.array([3.0, 1.0, 6.0])
    weights = jnp.array([0.5, 1.0, 2.5])

    got = jax.grad(lambda w: ppr.multinomial_logw(x, w))(weights)
    want = jax.grad(
        lambda w: stats.multinomial.logpmf(x.astype(int), 10, w / jnp.sum(w))
    )(weights)

    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_multinomial_logw_matches_fast_multinomial_batch():
    """Batched draws from fast_multinomial give one log-weight per draw."""
    probs = jnp.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    x = ppr.fast_multinomial(jax.random.key(0), jnp.array([10.0, 20.0]), probs)

    got = ppr.multinomial_logw(x, probs)

    assert got.shape == (2,)
    np.testing.assert_allclose(
        got, [ppr.multinomial_logw(x[i], probs[i]) for i in range(2)]
    )


def test_logw_holds_draws_and_trials_fixed():
    x, n = jnp.array(3.0), jnp.array(40.0)
    rates = jnp.array([2.5, 0.7])

    assert jax.grad(ppr.poisson_logw)(x, 12.5) == 0.0
    assert jax.grad(ppr.binomial_logw, argnums=(0, 1))(x, n, 0.2) == (0.0, 0.0)
    dx, dn = jax.grad(ppr.euler_multinomial_logw, argnums=(0, 1))(
        jnp.array([x, 1.0]), n, rates, DT
    )
    np.testing.assert_array_equal(dx, 0.0)
    assert dn == 0.0
    dx = jax.grad(ppr.multinomial_logw)(jnp.array([3.0, 1.0]), jnp.array([0.4, 0.6]))
    np.testing.assert_array_equal(dx, 0.0)


def test_logw_finite_at_zero_rates():
    """Zero rates (and hence zero counts) give finite values and gradients."""
    zero = jnp.array(0.0)

    for args in [
        (zero, jnp.array([0.0, 0.0])),
        (jnp.array([2.0, 0.0]), jnp.array([1.5, 0.0])),
    ]:
        x, rates = args
        x = jnp.broadcast_to(x, (2,))
        value, grad = jax.value_and_grad(
            lambda r, x=x: ppr.euler_multinomial_logw(x, 10.0, r, DT)
        )(rates)
        assert jnp.isfinite(value)
        assert jnp.all(jnp.isfinite(grad))

    value, grad = jax.value_and_grad(lambda lam: ppr.poisson_logw(zero, lam))(zero)
    assert jnp.isfinite(value) and jnp.isfinite(grad)
    value, grad = jax.value_and_grad(lambda p: ppr.binomial_logw(zero, 5.0, p))(zero)
    assert jnp.isfinite(value) and jnp.isfinite(grad)


def test_euler_multinomial_logw_layouts():
    """Per-event arrays (as in a vectorized rproc) broadcast against a scalar
    rate, and a batch with events on the last axis (as samplers return them)
    gives the same values as one particle at a time."""
    J = 4
    x0 = jnp.array([0.0, 1.0, 2.0, 3.0])
    x1 = jnp.array([1.0, 0.0, 0.0, 2.0])
    n = jnp.full(J, 30.0)
    r0 = jnp.linspace(0.5, 2.0, J)
    mu = 0.02

    got = ppr.euler_multinomial_logw((x0, x1), n, (r0, mu), DT)
    want = jax.vmap(
        lambda x0_, x1_, n_, r0_: ppr.euler_multinomial_logw(
            jnp.stack([x0_, x1_]), n_, jnp.stack([r0_, mu]), DT
        )
    )(x0, x1, n, r0)

    assert got.shape == (J,)
    np.testing.assert_allclose(got, want, rtol=1e-6)

    stacked = ppr.euler_multinomial_logw(
        jnp.stack([x0, x1], axis=-1), n, jnp.stack([r0, jnp.full(J, mu)], axis=-1), DT
    )
    np.testing.assert_allclose(stacked, want, rtol=1e-6)
