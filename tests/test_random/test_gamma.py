import jax
import jax.numpy as jnp
import numpy as np
import pypomp.random as ppr
from tests.test_random.helpers import jax_x64_enabled, check_moments


def test_gamma_basics() -> None:
    key = jax.random.key(0)
    alpha = jnp.array([1.0, 2.0, 3.0])
    x = ppr.fast_gamma(key, alpha)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0


def test_gamma_dtypes() -> None:
    key = jax.random.key(0)
    alpha = jnp.array([1.0, 2.0, 3.0])

    x_float32 = ppr.fast_gamma(key, alpha, dtype=jnp.float32)
    assert x_float32.dtype == jnp.float32

    x_default = ppr.fast_gamma(key, alpha)
    assert x_default.dtype == jnp.float32

    with jax_x64_enabled():
        x_float64 = ppr.fast_gamma(key, alpha, dtype=jnp.float64)
        assert x_float64.dtype == jnp.float64

        x_default_x64 = ppr.fast_gamma(key, alpha)
        assert x_default_x64.dtype == jnp.float64


def test_gamma_custom_accuracy_args() -> None:
    key = jax.random.key(12345)
    alpha = jnp.array([1.0, 2.0, 5.0, 10.0])

    # 1. Test fast_gamma with default limits (newton_steps=3)
    x_default = ppr.fast_gamma(key, alpha)
    assert x_default.shape == (4,)
    assert x_default.min() >= 0

    # 2. Test fast_gamma with custom steps (e.g. 0 steps, should be different from 3 steps)
    x_0steps = ppr.fast_gamma(key, alpha, newton_steps=0)
    assert x_0steps.shape == (4,)
    assert x_0steps.min() >= 0
    assert not jnp.allclose(x_default, x_0steps, rtol=1e-5, atol=1e-5)

    # 3. Test fast_gamma with more steps (e.g. 5 steps, should be very close to 3 steps)
    x_5steps = ppr.fast_gamma(key, alpha, newton_steps=5)
    assert x_5steps.shape == (4,)
    assert x_5steps.min() >= 0
    assert jnp.allclose(x_default, x_5steps, rtol=1e-4, atol=1e-4)


def test_gamma_moments() -> None:
    key = jax.random.key(456)
    alpha_vals = [0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_samples = 100000

    for alpha in alpha_vals:
        alpha_arr = jnp.full((n_samples,), alpha, dtype=jnp.float32)
        samples = ppr.fast_gamma(key, alpha_arr)

        # Theoretical moments for Gamma(shape=alpha, scale=1)
        mean_th = alpha
        var_th = alpha
        if alpha > 0:
            skew_th = 2.0 / np.sqrt(alpha)
        else:
            skew_th = 0.0

        check_moments(
            dist_name="Gamma",
            params_str=f"alpha={alpha}",
            samples=np.array(samples),
            mean_th=mean_th,
            var_th=var_th,
            skew_th=skew_th,
            mean_tol=(0.02, 0.03),
            var_tol=(0.03, 0.03),
            skew_tol=(0.10, 0.06),
            check_skew=(alpha > 0.5),
        )
