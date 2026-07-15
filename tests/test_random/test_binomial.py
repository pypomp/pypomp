import jax
import jax.numpy as jnp
import numpy as np
import pypomp.random as ppr
from tests.test_random.helpers import jax_x64_enabled, check_moments


def test_binomial_basics() -> None:
    key = jax.random.key(0)
    n = jnp.array([1, 2, 3])
    p = jnp.array([0.5, 0.6, 0.7])
    x = ppr.fast_binomial(key, n, p)
    assert x.shape == (3,)
    assert x.min() >= 0
    assert all(x <= n)


def test_binomial_dtypes() -> None:
    key = jax.random.key(0)
    n = jnp.array([1, 2, 3])
    p = jnp.array([0.5, 0.6, 0.7])

    x_int32 = ppr.fast_binomial(key, n, p, dtype=jnp.int32)
    assert x_int32.dtype == jnp.int32

    x_default = ppr.fast_binomial(key, n, p)
    assert x_default.dtype == jnp.float32

    with jax_x64_enabled():
        x_int64 = ppr.fast_binomial(key, n, p, dtype=jnp.int64)
        assert x_int64.dtype == jnp.int64

        x_default_x64 = ppr.fast_binomial(key, n, p)
        assert x_default_x64.dtype == jnp.float64


def test_binomial_edges_and_invalid_inputs_float() -> None:
    key = jax.random.key(0)
    # Includes deterministic edges (p=0, p=1, n=0) and invalid inputs.
    n = jnp.array([5.0, 5.0, 0.0, -2.0, 5.0, 5.0], dtype=jnp.float32)
    p = jnp.array([0.0, 1.0, 0.4, 0.4, -0.1, 1.2], dtype=jnp.float32)

    x = ppr.fast_binomial(key, n, p)
    ref_x = jax.random.binomial(key, n, p)  # Should match JAX

    assert x.shape == ref_x.shape
    assert x.dtype == ref_x.dtype
    # Check that x returns the same value as jax.random.binomial, handling nans
    assert jnp.all((x == ref_x) | (jnp.isnan(x) & jnp.isnan(ref_x)))


def test_binomial_edges_and_invalid_inputs_int() -> None:
    key = jax.random.key(0)
    n = jnp.array([5.0, 5.0, 0.0, -2.0, 5.0, 5.0], dtype=jnp.float32)
    p = jnp.array([0.0, 1.0, 0.4, 0.4, -0.1, 1.2], dtype=jnp.float32)

    x_int = ppr.fast_binomial(key, n, p, dtype=jnp.int32)
    ref_x = jax.random.binomial(key, n, p)

    assert x_int.shape == n.shape
    assert x_int.dtype == jnp.int32
    # Verify our integer return type maps invalid inputs to -1, else matches ref_x
    expected_x_int = jnp.where(jnp.isnan(ref_x), -1, ref_x.astype(jnp.int32))
    assert jnp.all(x_int == expected_x_int)


def test_binomial_moments() -> None:
    key = jax.random.key(123)
    n_vals = [3, 20, 100, 2000]
    p_vals = [0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]
    test_params = [(n_val, p_val) for n_val in n_vals for p_val in p_vals]
    n_samples = 100000

    for n, p in test_params:
        n_arr = jnp.full((n_samples,), n, dtype=jnp.float32)
        p_arr = jnp.full((n_samples,), p, dtype=jnp.float32)
        samples = ppr.fast_binomial(key, n_arr, p_arr)

        mean_th = n * p
        var_th = n * p * (1 - p)
        if var_th > 0:
            skew_th = (1 - 2 * p) / np.sqrt(var_th)
        else:
            skew_th = 0.0

        check_moments(
            dist_name="Binomial",
            params_str=f"n={n}, p={p}",
            samples=np.array(samples),
            mean_th=mean_th,
            var_th=var_th,
            skew_th=skew_th,
            mean_tol=(0.02, 0.02),
            var_tol=(0.03, 0.03),
            skew_tol=(0.15, 0.06),
            check_skew=(var_th > 2),
        )
