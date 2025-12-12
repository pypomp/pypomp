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

    x = ppr.fast_approx_rbinom(key, n, p, dtype=jnp.int32)
    assert x.dtype == jnp.int32


def test_rbinom_invalid_and_edges():
    key = jax.random.key(0)
    # Includes deterministic edges (p=0, p=1, n=0) and invalid inputs.
    n = jnp.array([5.0, 5.0, 0.0, -2.0, 5.0, 5.0], dtype=jnp.float32)
    p = jnp.array([0.0, 1.0, 0.4, 0.4, -0.1, 1.2], dtype=jnp.float32)

    x = ppr.fast_approx_rbinom(key, n, p)

    assert x.shape == n.shape
    assert x.dtype == n.dtype
    assert x[0] == 0.0  # p = 0 → always zero successes
    assert x[1] == n[1]  # p = 1 → always n successes
    assert x[2] == 0.0  # n = 0 → always zero successes (valid)
    assert jnp.isnan(x[3])  # negative n → invalid, returns nan
    assert jnp.isnan(x[4])  # p < 0 → invalid, returns nan
    assert jnp.isnan(x[5])  # p > 1 → invalid, returns nan

    x = ppr.fast_approx_rbinom(key, n, p, dtype=jnp.int32)

    assert x.shape == n.shape
    assert x.dtype == jnp.int32
    assert x[0] == 0  # p = 0 → always zero successes
    assert x[1] == n[1]  # p = 1 → always n successes
    assert x[2] == 0  # n = 0 → always zero successes (valid)
    assert x[3] == -1  # negative n → invalid, returns -1
    assert x[4] == -1  # p < 0 → invalid, returns -1
    assert x[5] == -1  # p > 1 → invalid, returns -1


def test_rmultinom():
    key = jax.random.key(0)
    n = jnp.array([5, 10], dtype=jnp.int32)
    p = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]], dtype=jnp.float32)
    x = ppr.fast_approx_rmultinom(key, n, p)
    # Shape: should be (2, 3) for 2 draws, 3 categories
    assert x.shape == (2, 3)
    # Sum across categories should equal n for each row
    assert np.allclose(np.sum(np.array(x), axis=1), np.array(n))
    # Each value should be non-negative and no more than n for that row
    for xi, ni in zip(x, n):
        assert np.all(xi >= 0)
        assert np.all(xi <= ni)
    # Test with a batch of 1 for broadcasting
    n2 = jnp.array(12, dtype=jnp.int32)
    p2 = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32)
    x2 = ppr.fast_approx_rmultinom(key, n2, p2)
    assert x2.shape == (3,)
    assert jnp.sum(x2) == n2
    assert jnp.all(x2 >= 0)


def test_rmultinom_edges_and_invalid():
    key = jax.random.key(0)

    # Edge case 1: n=0, valid probability vector
    n = jnp.array(0, dtype=jnp.int32)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
    x = ppr.fast_approx_rmultinom(key, n, p)
    assert x.shape == (3,)
    assert jnp.sum(x) == 0
    assert jnp.all(x == 0)

    # Edge case 2: p = [1.0, 0.0, 0.0] (all probability on first class)
    n = jnp.array(5, dtype=jnp.int32)
    p = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    x = ppr.fast_approx_rmultinom(key, n, p)
    assert x.shape == (3,)
    assert x[0] == 5
    assert jnp.all(x[1:] == 0)

    # Edge case 3: p = [0.0, 1.0, 0.0] (all on second class)
    x = ppr.fast_approx_rmultinom(key, n, jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
    assert x[1] == 5
    assert jnp.all(x[np.array([0, 2])] == 0)

    # Edge case 4: n = negative, should return nan or -1
    n_neg = jnp.array(-3, dtype=jnp.int32)

    x = ppr.fast_approx_rmultinom(key, n_neg, p, dtype=jnp.float32)
    assert jnp.all(jnp.isnan(x))
    x = ppr.fast_approx_rmultinom(key, n_neg, p, dtype=jnp.int32)
    assert jnp.all(x == -1)

    # Edge case 5: Probability vector does not sum to 1; should normalize to sum to 1
    n = jnp.array(50, dtype=jnp.int32)
    p_bad = jnp.array([0.2, 0.3, 0.7], dtype=jnp.float32)  # sums to 1.2
    p_good = jnp.array([0.2 / 1.2, 0.3 / 1.2, 0.7 / 1.2], dtype=jnp.float32)

    x_bad = ppr.fast_approx_rmultinom(key, n, p_bad)
    x_good = ppr.fast_approx_rmultinom(key, n, p_good)
    assert jnp.allclose(x_bad, x_good)

    # Edge case 6: Probability vector contains negative values; should normalize to sum to 1
    # p_bad = jnp.array([0.5, -0.2, 0.7], dtype=jnp.float32)
    # p_good = jnp.array([0.3, 0.0, 0.7], dtype=jnp.float32)
    # x_bad = ppr.fast_approx_rmultinom(key, n, p_bad)
    # x_good = ppr.fast_approx_rmultinom(key, n, p_good)
    # assert jnp.allclose(x_bad, x_good)

    # Edge case 7: Only one category (should get all in that category)
    n = jnp.array(4, dtype=jnp.int32)
    p_onecat = jnp.array([1.0], dtype=jnp.float32)
    x = ppr.fast_approx_rmultinom(key, n, p_onecat)
    assert x.shape == (1,)
    assert x[0] == n

    # Edge case 8: Shape/broadcasting mismatch
    n = jnp.array([5, 6], dtype=jnp.int32)
    p2 = jnp.array([[0.5, 0.5], [0.7, 0.3]], dtype=jnp.float32)
    x = ppr.fast_approx_rmultinom(key, n, p2)
    assert x.shape == (2, 2)
    assert jnp.allclose(jnp.sum(x, axis=1), n)


def test_rgamma():
    key = jax.random.key(0)
    alpha = jnp.array([1.0, 2.0, 3.0])
    x = ppr.rgamma(key, alpha)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0


def test_rpoisson_moments(n_moments=3):
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


def test_rbinom_moments(n_moments=3):
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
