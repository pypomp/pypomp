import jax
import jax.numpy as jnp
import numpy as np

import pypomp.random as ppr
from tests.helpers.assertions import check_moments, jax_x64_enabled


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


def test_binomial_small_np_tails_match_exact_quantiles() -> None:
    """Small-np tails match the exact inverse CDF on the float32 uniform lattice:
    np <= 0.5 at the default exact_max, np <= 3 with exact_max=16."""
    from scipy import stats

    from pypomp.random.binom import binominv

    # jax.random.uniform float32 draws are multiples of 2**-23; take the 4096
    # lattice points nearest each end (tail mass ~4.9e-4).
    i = np.arange(1, 4097, dtype=np.float64)
    tails = {"upper": 1.0 - i * 2.0**-23, "lower": i * 2.0**-23}
    # p > 0.5: the lower tail maps to the flipped upper tail
    cases = [
        (627, 7.3e-4, "upper", None),
        (20, 0.025, "upper", None),
        (100_000, 5e-6, "upper", None),
        (20, 0.975, "lower", None),
        (10, 0.3, "upper", 16),
        (1000, 0.0015, "upper", 16),
        (100_000, 3e-5, "upper", 16),
        (50, 0.97, "lower", 16),
    ]
    for n, p, tail, exact_max in cases:
        u = tails[tail]
        n32, p32 = np.float32(n), np.float32(p)
        kwargs = {} if exact_max is None else {"exact_max": exact_max}
        got = np.asarray(
            binominv(
                jnp.asarray(u, dtype=jnp.float32),
                jnp.full(u.shape, n32),
                jnp.full(u.shape, p32),
                **kwargs,
            )
        )
        exact = stats.binom.ppf(u, float(n32), float(p32))
        assert np.all(np.abs(got - exact) <= 1)
        # float32 rounding in the pmf sum can shift a cdf boundary by a few
        # lattice points, so any mismatch must sit that close to one.
        bad = got != exact
        boundary = stats.binom.cdf(np.minimum(got, exact)[bad], float(n32), float(p32))
        dist = np.abs(u[bad] - boundary) / 2.0**-23
        assert np.all(dist <= 4), (
            f"n={n}, p={p}: {tail}-tail mismatch {dist.max():.0f} lattice points "
            "from a cdf boundary"
        )


def test_binomial_tail_discrete_and_accurate() -> None:
    """Extreme upper tail for large n, small np produces discrete counts and no wild divergence."""
    from scipy import stats

    from pypomp.random.binom import binominv

    n = 100_000.0
    p = 0.2 / n  # np = 0.2

    # Probe the 1-in-500k upper tail (u > CDF(4) ~ 0.9999977)
    u = jnp.array([0.999998, 0.999999, 0.9999995], dtype=jnp.float32)
    n_arr = jnp.full_like(u, n)
    p_arr = jnp.full_like(u, p)

    samples = binominv(u, n_arr, p_arr)

    # 1. Must never return fractional counts
    assert jnp.all(samples == jnp.floor(samples)), (
        f"Non-integer counts returned: {samples}"
    )

    # 2. Must not diverge to 7.88, 13, etc. (true scipy quantiles are 5.0 or 6.0)
    exact = stats.binom.ppf(np.array(u), n, p)
    assert np.all(np.abs(np.array(samples) - exact) <= 1.0), (
        f"Tail quantiles diverged: got {samples}, expected {exact}"
    )

    # 3. Test random sampler with float32 output
    key = jax.random.key(1001)
    draws = ppr.fast_binomial(
        key,
        jnp.full((10_000,), n, dtype=jnp.float32),
        jnp.full((10_000,), p, dtype=jnp.float32),
    )
    assert jnp.all(draws == jnp.floor(draws)), (
        "fast_binomial returned non-integer values"
    )
