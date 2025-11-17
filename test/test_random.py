import jax
import jax.numpy as jnp
import pypomp as pp
import time


def test_poissoninvf():
    key = jax.random.PRNGKey(0)
    lam = jnp.array([1.0, 2.0, 3.0])
    x = pp.rpoisson(key, lam)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0


def test_poissoninvf_performance():
    # Prepare parameters
    n = 100_000
    key = jax.random.PRNGKey(42)
    lam = jnp.array([0.01, 0.2, 1.0, 10.0, 50.0, 100.0], dtype=jnp.float32)
    # lam = jnp.array([100.0], dtype=jnp.float32)
    lam_samples = jnp.repeat(lam, n // len(lam))
    key1, key2 = jax.random.split(key)

    # Warmup to trigger JITs
    _ = pp.rpoisson(key1, lam_samples).block_until_ready()
    _ = jax.random.poisson(key2, lam_samples).block_until_ready()

    # JAX's .block_until_ready() ensures we measure actual compute time
    key1, key2 = jax.random.split(key)
    t0 = time.time()
    x_pp = pp.rpoisson(key1, lam_samples).block_until_ready()
    t1 = time.time()
    x_jax = jax.random.poisson(key2, lam_samples).block_until_ready()
    t2 = time.time()

    print(f"pp.rpoisson: {t1 - t0:.4f} seconds for {n} samples")
    print(f"jax.random.poisson: {t2 - t1:.4f} seconds for {n} samples")
    pass
