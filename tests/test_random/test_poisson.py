import jax
import jax.numpy as jnp
import numpy as np
import pypomp.random as ppr
from tests.test_random.helpers import jax_x64_enabled, check_moments


def test_poisson_basics() -> None:
    key = jax.random.key(0)
    lam = jnp.array([1.0, 2.0, 3.0])
    x = ppr.fast_poisson(key, lam)
    assert x.shape == (3,)
    assert x.dtype == jnp.int32
    assert x.min() >= 0


def test_poisson_dtypes() -> None:
    key = jax.random.key(0)
    lam = jnp.array([1.0, 2.0, 3.0])

    x_int32 = ppr.fast_poisson(key, lam, dtype=jnp.int32)
    assert x_int32.dtype == jnp.int32

    x_default = ppr.fast_poisson(key, lam)
    assert x_default.dtype == jnp.int32

    with jax_x64_enabled():
        x_int64 = ppr.fast_poisson(key, lam, dtype=jnp.int64)
        assert x_int64.dtype == jnp.int64

        x_default_x64 = ppr.fast_poisson(key, lam)
        assert x_default_x64.dtype == jnp.int64


def test_poisson_custom_accuracy_args() -> None:
    key = jax.random.key(12345)
    lam = jnp.array([1.0, 2.0, 5.0, 10.0])

    x_default = ppr.fast_poisson(key, lam)
    assert x_default.shape == (4,)
    assert x_default.min() >= 0

    x_custom = ppr.fast_poisson(key, lam, max_newton_loops=3, max_inverse_cdf_loops=10)
    assert x_custom.shape == (4,)
    assert x_custom.min() >= 0


def test_poisson_moments() -> None:
    key = jax.random.key(1)
    lam_vals = [
        0.0001,
        0.1,
        1.0,
        4.0,
        4.01,
        8.0,
        15.0,
        19.9,
        20.1,
        25.0,
        30.0,
        100.0,
        500.0,
    ]
    n_samples = 100000

    for lam in lam_vals:
        lam_arr = jnp.full((n_samples,), lam, dtype=jnp.float32)
        samples = ppr.fast_poisson(key, lam_arr)

        # Theoretical moments for Poisson
        mean_th = lam
        var_th = lam
        skew_th = 1.0 / np.sqrt(lam) if lam > 0 else 0.0

        check_moments(
            dist_name="Poisson",
            params_str=f"lam={lam}",
            samples=np.array(samples),
            mean_th=mean_th,
            var_th=var_th,
            skew_th=skew_th,
            mean_tol=(0.02, 0.02),
            var_tol=(0.03, 0.03),
            skew_tol=(0.10, 0.04),
            check_skew=(lam > 0.5),
        )
