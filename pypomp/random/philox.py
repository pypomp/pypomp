"""A Philox4x32-10 PRNG implementation whose draws use all four output words.

JAX's own Philox keys run the full ten rounds for every 32-bit word and XOR the four
output words together. Here each call supplies four words, so a draw of n values
costs about n / 4 calls. For a draw of shape (..., L), element (l, m), with l
indexing the flattened leading axes, is word l % 4 of the call with counter
(l // 4) * L + m. When the last axis indexes particles, each particle's draws come
from its own calls, which is what the CuTe backend needs to generate them
inside one GPU thread. On the XLA path, keys made with :data:`PHILOX` give the same
random numbers as the CuTe kernel:

>>> import jax
>>> from pypomp.random.philox import PHILOX
>>> key = jax.random.key(0, impl=PHILOX)
"""

import math

import jax
import jax.extend.random as jxr
import jax.numpy as jnp
import numpy as np
from jax import lax

# Multipliers and key increments from Salmon et al. (2011), "Parallel random
# numbers: as easy as 1, 2, 3".
_M0, _M1 = np.uint32(0xD2511F53), np.uint32(0xCD9E8D57)
_W0, _W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)


def _philox4x32(
    k0: jax.Array,
    k1: jax.Array,
    x0: jax.Array,
    x1: jax.Array,
    x2: jax.Array,
    x3: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the four Philox4x32-10 words for key (k0, k1) and counter (x0..x3)."""
    for r in range(10):
        if r:
            k0, k1 = k0 + _W0, k1 + _W1
        lo0, hi0 = lax.mul(_M0, x0), _mulhi(_M0, x0)
        lo1, hi1 = lax.mul(_M1, x2), _mulhi(_M1, x2)
        x0, x1, x2, x3 = hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0
    return x0, x1, x2, x3


def _mulhi16(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return the high 32 bits of the product of 32-bit unsigned `a` and `b`, from
    products of their 16-bit halves, for JAX before 0.11, which lacks lax.mulhi."""
    a, b = jnp.asarray(a, jnp.uint32), jnp.asarray(b, jnp.uint32)
    m = jnp.uint32(0xFFFF)
    a_lo, a_hi, b_lo, b_hi = a & m, a >> 16, b & m, b >> 16
    lh, hl = a_lo * b_hi, a_hi * b_lo
    mid = ((a_lo * b_lo) >> 16) + (lh & m) + (hl & m)
    return a_hi * b_hi + (lh >> 16) + (hl >> 16) + (mid >> 16)


_mulhi = getattr(lax, "mulhi", _mulhi16)


def _seed(seed: jax.Array) -> jax.Array:
    """Return the key data for an integer seed."""
    seed = jnp.asarray(seed)
    hi = (
        (seed.astype(jnp.uint64) >> 32).astype(jnp.uint32)
        if seed.dtype.itemsize == 8
        else jnp.uint32(0)
    )
    return jnp.stack([jnp.asarray(hi, jnp.uint32), seed.astype(jnp.uint32)])


def _split(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Return the data of `shape` new keys derived from `key`."""
    n = math.prod(shape)
    c = jnp.arange(n, dtype=jnp.uint32)
    z = jnp.zeros(n, jnp.uint32)
    y0, y1, _, _ = _philox4x32(key[0], key[1], z, z, z, c)
    return jnp.stack([y0, y1], axis=-1).reshape(tuple(shape) + (2,))


def _fold_in(key: jax.Array, data: int) -> jax.Array:
    """Return the data of the key derived from `key` and the integer `data`."""
    z = jnp.uint32(0)
    y0, y1, _, _ = _philox4x32(key[0], key[1], z, z, z, jnp.asarray(data, jnp.uint32))
    return jnp.stack([y0, y1])


def _words32(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Return uint32 words for a draw of `shape`, in the layout of the module docstring."""
    shape = tuple(shape)
    n = math.prod(shape)
    lane = shape[-1] if len(shape) >= 2 else 1
    lead = n // lane if lane else 0
    groups = -(-lead // 4)
    c = jnp.arange(groups * lane, dtype=jnp.uint32)
    z = jnp.zeros_like(c)
    w = jnp.stack(_philox4x32(key[0], key[1], c, z, z, z))  # (4, groups * lane)
    w = w.reshape(4, groups, lane).transpose(1, 0, 2).reshape(groups * 4, lane)[:lead]
    return w.reshape(shape)


def _random_bits(key: jax.Array, bit_width: int, shape: tuple[int, ...]) -> jax.Array:
    """Return random unsigned integers of `bit_width` bits and `shape`."""
    if bit_width == 64:
        w = _words32(key, tuple(shape) + (2,))
        return (w[..., 0].astype(jnp.uint64) << 32) | w[..., 1].astype(jnp.uint64)
    w = _words32(key, shape)
    return w if bit_width == 32 else w.astype({8: jnp.uint8, 16: jnp.uint16}[bit_width])


PHILOX = jxr.define_prng_impl(
    key_shape=(2,),
    seed=_seed,
    split=_split,
    random_bits=_random_bits,
    fold_in=_fold_in,
    name="pypomp_philox4x32_dense",
    tag="ppx",
)
"""JAX PRNG implementation for ``jax.random.key(seed, impl=PHILOX)``."""
