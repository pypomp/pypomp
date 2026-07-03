import jax
import jax.numpy as jnp
import numpy as np
import pypomp.random as ppr


def test_multinomial_basics() -> None:
    key = jax.random.key(0)
    n = jnp.array([5, 10], dtype=jnp.int32)
    p = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]], dtype=jnp.float32)
    x = ppr.fast_multinomial(key, n, p)

    assert x.shape == (2, 3)  # 2 draws, 3 categories
    assert np.allclose(np.sum(np.array(x), axis=1), np.array(n))
    for xi, ni in zip(x, n):
        assert np.all(xi >= 0)
        assert np.all(xi <= ni)


def test_multinomial_broadcasting() -> None:
    key = jax.random.key(0)
    n = jnp.array(12, dtype=jnp.int32)
    p = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32)
    x = ppr.fast_multinomial(key, n, p)
    assert x.shape == (3,)
    assert jnp.sum(x) == n
    assert jnp.all(x >= 0)


def test_fast_multinomial_custom_accuracy_args() -> None:
    key = jax.random.key(0)
    n = jnp.array([5, 10], dtype=jnp.int32)
    p = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]], dtype=jnp.float32)
    x = ppr.fast_multinomial(key, n, p, order=1, exact_max=3)
    assert x.shape == (2, 3)
    assert jnp.all(x >= 0)


def test_multinomial_shape_broadcasting_mismatch() -> None:
    key = jax.random.key(0)
    n = jnp.array([5, 6], dtype=jnp.int32)
    p = jnp.array([[0.5, 0.5], [0.7, 0.3]], dtype=jnp.float32)
    x = ppr.fast_multinomial(key, n, p)
    assert x.shape == (2, 2)
    assert jnp.allclose(jnp.sum(x, axis=1), n)


def test_multinomial_edge_n_zero() -> None:
    key = jax.random.key(0)
    n = jnp.array(0, dtype=jnp.int32)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
    x = ppr.fast_multinomial(key, n, p)
    ref_x = jax.random.multinomial(key, n, p)
    assert jnp.all(x == ref_x)


def test_multinomial_edge_probability_concentrated() -> None:
    key = jax.random.key(0)
    n = jnp.array(5, dtype=jnp.int32)

    # All probability on first class
    p1 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    x1 = ppr.fast_multinomial(key, n, p1)
    ref_x1 = jax.random.multinomial(key, n, p1)
    assert jnp.all(x1 == ref_x1)

    # All probability on second class
    p2 = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    x2 = ppr.fast_multinomial(key, n, p2)
    ref_x2 = jax.random.multinomial(key, n, p2)
    assert jnp.all(x2 == ref_x2)


def test_multinomial_invalid_negative_n() -> None:
    """Verify negative n inputs map to NaN or -1 depending on the dtype."""
    key = jax.random.key(0)
    n_neg = jnp.array(-3, dtype=jnp.int32)
    p = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)

    x_float = ppr.fast_multinomial(key, n_neg, p, dtype=jnp.float32)
    ref_x = jax.random.multinomial(key, n_neg, p)
    assert jnp.all(jnp.isnan(x_float) == jnp.isnan(ref_x))

    x_int = ppr.fast_multinomial(key, n_neg, p, dtype=jnp.int32)
    assert jnp.all(x_int == -1)


def test_multinomial_bad_probability_normalization() -> None:
    key = jax.random.key(0)
    n = jnp.array(50, dtype=jnp.int32)
    p_bad = jnp.array([0.2, 0.3, 0.7], dtype=jnp.float32)
    p_good = jnp.array([0.2 / 1.2, 0.3 / 1.2, 0.7 / 1.2], dtype=jnp.float32)

    x_bad = ppr.fast_multinomial(key, n, p_bad)
    x_good = ppr.fast_multinomial(key, n, p_good)
    assert jnp.allclose(x_bad, x_good)


def test_multinomial_edge_single_category() -> None:
    key = jax.random.key(0)
    n = jnp.array(4, dtype=jnp.int32)
    p_onecat = jnp.array([1.0], dtype=jnp.float32)
    x = ppr.fast_multinomial(key, n, p_onecat)
    ref_x = jax.random.multinomial(key, n, p_onecat)
    assert jnp.all(x == ref_x)
