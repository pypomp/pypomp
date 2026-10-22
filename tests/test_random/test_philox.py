import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.random import philox


def test_rounds_match_jax_philox4x32():
    from jax._src.random import philox4x32 as jp

    rng = np.random.default_rng(0)
    k = rng.integers(0, 2**32, size=(2,), dtype=np.uint32)
    x = rng.integers(0, 2**32, size=(4, 64), dtype=np.uint32)
    ours = philox.philox4x32(jnp.uint32(k[0]), jnp.uint32(k[1]), *map(jnp.asarray, x))
    theirs = jp.philox4x32_p.bind(jnp.uint32(k[0]), jnp.uint32(k[1]), *map(jnp.asarray, x))
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_four_words_per_call_layout():
    key = jax.random.key(3, impl=pp.random.PHILOX)
    kd = jax.random.key_data(key)
    J = 6
    bits = jax.random.bits(key, (11, J), dtype=jnp.uint32)
    for l, m in [(0, 0), (3, 5), (4, 2), (10, 1)]:
        w = philox.philox4x32(kd[0], kd[1], jnp.uint32((l // 4) * J + m), jnp.uint32(0), jnp.uint32(0),
                              jnp.uint32(0))
        assert int(bits[l, m]) == int(w[l % 4])


def test_split_fold_in_and_distinct_streams():
    key = jax.random.key(0, impl=pp.random.PHILOX)
    a, b = jax.random.split(key)
    assert not np.array_equal(jax.random.key_data(a), jax.random.key_data(b))
    assert not np.array_equal(jax.random.key_data(jax.random.fold_in(key, 1)),
                              jax.random.key_data(jax.random.fold_in(key, 2)))
    u = jax.random.uniform(a, (1000,))
    v = jax.random.uniform(b, (1000,))
    assert abs(float(jnp.corrcoef(u, v)[0, 1])) < 0.1


@pytest.mark.parametrize("shape", [(100_000,), (250, 400)])
def test_uniform_and_normal_moments(shape):
    key = jax.random.key(1, impl=pp.random.PHILOX)
    u = np.asarray(jax.random.uniform(key, shape)).ravel()
    z = np.asarray(jax.random.normal(jax.random.fold_in(key, 7), shape)).ravel()
    n = u.size
    assert abs(u.mean() - 0.5) < 4 * np.sqrt(1 / 12 / n)
    assert abs(u.var() - 1 / 12) < 0.005
    counts = np.histogram(u, bins=20, range=(0, 1))[0]
    chi2 = ((counts - n / 20) ** 2 / (n / 20)).sum()
    assert chi2 < 45  # 19 degrees of freedom; p ~ 0.0007
    assert abs(z.mean()) < 4 / np.sqrt(n)
    assert abs(z.std() - 1) < 0.01
