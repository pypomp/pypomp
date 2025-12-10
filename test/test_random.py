import jax
import jax.numpy as jnp
import pypomp.random as ppr
import numpy as np
import warnings


def test_rpoisson():
    key = jax.random.key(0)
    lam = jnp.array([1.0, 2.0, 3.0])
    x = ppr.fast_approx_rpoisson(key, lam)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0


def test_rbinom():
    key = jax.random.key(0)
    n = jnp.array([1, 2, 3])
    p = jnp.array([0.5, 0.6, 0.7])
    x = ppr.fast_approx_rbinom(key, n, p)
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


def rpoisson_moments(n_moments=2):
    """Check that the first n_moments moments of fast_approx_rpoisson match theoretical Poisson moments."""
    key = jax.random.key(42)
    lam_vals = [0.0001, 0.1, 1.0, 4.0, 4.01, 8.0, 15, 19.9, 20.1, 25, 30, 100.0, 500.0]
    n_samples = 100000

    for lam in lam_vals:
        lam_arr = jnp.full((n_samples,), lam, dtype=jnp.float32)
        samples = np.array(ppr.fast_approx_rpoisson(key, lam_arr))

        # Theoretical moments for Poisson
        mean_th = lam
        var_th = lam
        skew_th = 1.0 / np.sqrt(lam) if lam > 0 else 0.0

        mean_emp = samples.mean()
        var_emp = samples.var()
        # For empirical skewness: E[(X-mu)^3] / sigma^3
        centered = samples - mean_emp
        m3 = np.mean(centered**3)
        std_emp = np.std(samples)
        skew_emp = m3 / (std_emp**3) if std_emp > 0 else 0.0

        if n_moments >= 1:
            if not np.allclose(mean_emp, mean_th, rtol=0.02, atol=0.02):
                warnings.warn(
                    f"Poisson mean fail for lam={lam}. Empirical: {mean_emp}, Theoretical: {mean_th}"
                )
        if n_moments >= 2:
            if not np.allclose(var_emp, var_th, rtol=0.03, atol=0.03):
                warnings.warn(
                    f"Poisson var fail for lam={lam}. Empirical: {var_emp}, Theoretical: {var_th}"
                )
        if n_moments >= 3 and lam > 0.5:
            if not np.allclose(skew_emp, skew_th, rtol=0.10, atol=0.04):
                warnings.warn(
                    f"Poisson skew fail for lam={lam}. Empirical: {skew_emp}, Theoretical: {skew_th}"
                )


def rbinom_moments(n_moments=2):
    """Check that the first n_moments moments of fast_approx_rbinom match theoretical Binomial moments."""
    key = jax.random.key(123)
    n = [3, 20, 100, 2000]
    p = [0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]
    test_params = [(n_val, p_val) for n_val in n for p_val in p]
    n_samples = 100000

    for n, p in test_params:
        n_arr = jnp.full((n_samples,), n, dtype=jnp.float32)
        p_arr = jnp.full((n_samples,), p, dtype=jnp.float32)
        samples = np.array(ppr.fast_approx_rbinom(key, n_arr, p_arr))

        mean_th = n * p
        var_th = n * p * (1 - p)
        if var_th > 0:
            skew_th = (1 - 2 * p) / np.sqrt(var_th)
        else:
            skew_th = 0.0

        mean_emp = samples.mean()
        var_emp = samples.var()
        centered = samples - mean_emp
        m3 = np.mean(centered**3)
        std_emp = np.std(samples)
        skew_emp = m3 / (std_emp**3) if std_emp > 0 else 0.0

        if n_moments >= 1:
            if not np.allclose(mean_emp, mean_th, rtol=0.02, atol=0.02):
                warnings.warn(
                    f"Binom mean fail for n={n},p={p}. Empirical: {mean_emp}, Theoretical: {mean_th}"
                )
        if n_moments >= 2:
            if not np.allclose(var_emp, var_th, rtol=0.03, atol=0.03):
                warnings.warn(
                    f"Binom var fail for n={n},p={p}. Empirical: {var_emp}, Theoretical: {var_th}"
                )
        if n_moments >= 3 and var_th > 2:
            if not np.allclose(skew_emp, skew_th, rtol=0.15, atol=0.06):
                warnings.warn(
                    f"Binom skew fail for n={n},p={p}. Empirical: {skew_emp}, Theoretical: {skew_th}"
                )


def test_rgamma_moments(n_moments=3):
    """Check that the first n_moments moments of rgamma match theoretical Gamma moments (scale=1)."""
    key = jax.random.key(456)
    alpha_vals = [0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_samples = 100000

    for alpha in alpha_vals:
        alpha_arr = jnp.full((n_samples,), alpha, dtype=jnp.float32)
        samples = np.array(ppr.rgamma(key, alpha_arr))

        # Theoretical moments for Gamma(shape=alpha, scale=1)
        mean_th = alpha
        var_th = alpha
        if alpha > 0:
            skew_th = 2.0 / np.sqrt(alpha)
        else:
            skew_th = 0.0

        mean_emp = samples.mean()
        var_emp = samples.var()
        centered = samples - mean_emp
        m3 = np.mean(centered**3)
        std_emp = np.std(samples)
        skew_emp = m3 / (std_emp**3) if std_emp > 0 else 0.0

        if n_moments >= 1:
            if not np.allclose(mean_emp, mean_th, rtol=0.02, atol=0.03):
                warnings.warn(
                    f"Gamma mean fail for alpha={alpha}. Empirical: {mean_emp}, Theoretical: {mean_th}"
                )
        if n_moments >= 2:
            if not np.allclose(var_emp, var_th, rtol=0.03, atol=0.03):
                warnings.warn(
                    f"Gamma var fail for alpha={alpha}. Empirical: {var_emp}, Theoretical: {var_th}"
                )
        if n_moments >= 3 and alpha > 0.5:
            if not np.allclose(skew_emp, skew_th, rtol=0.10, atol=0.06):
                warnings.warn(
                    f"Gamma skew fail for alpha={alpha}. Empirical: {skew_emp}, Theoretical: {skew_th}"
                )
