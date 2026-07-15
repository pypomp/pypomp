import jax
import jax.numpy as jnp
import numpy as np
import pypomp.random as ppr
from tests.test_random.helpers import jax_x64_enabled, check_moments


def test_nbinomial_basics() -> None:
    key = jax.random.key(0)
    n = jnp.array([1.0, 2.0, 3.0])
    p = jnp.array([0.5, 0.6, 0.7])

    x = ppr.fast_nbinomial(key, n, p=p)
    assert x.shape == (3,)
    assert x.min() >= 0
    assert x.dtype == jnp.float32

    mu = jnp.array([1.0, 2.0, 3.0])
    x_mu = ppr.fast_nbinomial(key, n, mu=mu)
    assert x_mu.shape == (3,)
    assert x_mu.min() >= 0

    x_int32 = ppr.fast_nbinomial(key, n, p=p, dtype=jnp.int32)
    assert x_int32.dtype == jnp.int32


def test_nbinomial_dtypes() -> None:
    key = jax.random.key(0)
    n = jnp.array([1.0, 2.0, 3.0])
    p = jnp.array([0.5, 0.6, 0.7])

    x_float32 = ppr.fast_nbinomial(key, n, p=p, dtype=jnp.float32)
    assert x_float32.dtype == jnp.float32

    x_default = ppr.fast_nbinomial(key, n, p=p)
    assert x_default.dtype == jnp.float32

    with jax_x64_enabled():
        x_float64 = ppr.fast_nbinomial(key, n, p=p, dtype=jnp.float64)
        assert x_float64.dtype == jnp.float64

        x_default_x64 = ppr.fast_nbinomial(key, n, p=p)
        assert x_default_x64.dtype == jnp.float64


def test_nbinomial_parameter_edges() -> None:
    key = jax.random.key(0)
    n = jnp.array([1.0, 2.0, 3.0])

    x_p1 = ppr.fast_nbinomial(key, n, p=1.0)
    assert jnp.all(x_p1 == 0.0)

    x_n0 = ppr.fast_nbinomial(key, 0.0, mu=1.0)
    assert jnp.isnan(x_n0)


def test_nbinomial_custom_accuracy_args() -> None:
    key = jax.random.key(1)
    n = jnp.array([1.0, 5.0, 10.0])
    p = jnp.array([0.5, 0.5, 0.5])

    y_default = ppr.fast_nbinomial(key, n, p=p)
    assert y_default.shape == (3,)
    assert y_default.min() >= 0

    y_custom = ppr.fast_nbinomial(
        key,
        n,
        p=p,
        gamma_newton_loops=2,
        poisson_newton_loops=2,
        poisson_inverse_cdf_loops=5,
        gamma_adjustment_size=2,
    )
    assert y_custom.shape == (3,)
    assert y_custom.min() >= 0


def test_nbinomial_moments() -> None:
    key = jax.random.key(789)
    n_vals = [1.0, 5.0, 20.0]
    p_vals = [0.1, 0.5, 0.9]
    test_params = [(n, p) for n in n_vals for p in p_vals]
    n_samples = 100000

    for n, p in test_params:
        n_arr = jnp.full((n_samples,), n, dtype=jnp.float32)
        p_arr = jnp.full((n_samples,), p, dtype=jnp.float32)
        samples = ppr.fast_nbinomial(key, n_arr, p=p_arr)

        mean_th = n * (1 - p) / p
        var_th = n * (1 - p) / (p**2)

        check_moments(
            dist_name="NBinomial",
            params_str=f"n={n}, p={p}",
            samples=np.array(samples),
            mean_th=mean_th,
            var_th=var_th,
            mean_tol=(0.03, 0.03),
            var_tol=(0.05, 0.05),
            check_skew=False,
        )
