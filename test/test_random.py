import jax
import jax.numpy as jnp
import pypomp as pp
import pypomp.random as ppr


def test_rpoisson():
    key = jax.random.key(0)
    lam = jnp.array([1.0, 2.0, 3.0])
    x = ppr.rpoisson(key, lam)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0


def test_rbinom():
    key = jax.random.key(0)
    n = jnp.array([1, 2, 3])
    p = jnp.array([0.5, 0.6, 0.7])
    x = ppr.rbinom(key, n, p)
    assert x.shape == (3,)
    assert x.min() >= 0
    assert all(x <= n)


def test_rgamma():
    key = jax.random.key(0)
    alpha = jnp.array([1.0, 2.0, 3.0])
    x = ppr.rgamma(key, alpha)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0
