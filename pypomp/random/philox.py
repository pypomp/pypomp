"""A Philox4x32-10 PRNG whose draws use all four output words.

JAX's own ``philox4x32`` keys run the full ten rounds for every 32-bit word and XOR
the four output words together. Here each Philox call supplies four words, so a
draw of ``n`` values costs about ``n / 4`` calls. For a draw of shape ``(..., L)``
the four words of a call go to four consecutive elements along the leading axes at
the same position along the last axis: element ``(l, m)`` (``l`` indexing the
flattened leading axes) is word ``l % 4`` of the call with counter
``(l // 4) * L + m``. For pypomp's vectorized ``rproc``, where the last axis is the
particle axis, that keeps each particle's draws in its own calls, which is what the
CuTe backend needs to generate them inside one thread.

Use it by creating keys with this implementation; everything pypomp derives from
them (splits, fold-ins, draws) then uses it::

    key = jax.random.key(0, impl=pp.random.PHILOX)
    model.pfilter(J=1000, key=key)
"""

import math

import jax
import jax.extend.random as jxr
import jax.numpy as jnp
import numpy as np
from jax import lax

M0, M1 = np.uint32(0xD2511F53), np.uint32(0xCD9E8D57)  # Random123 multipliers
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)  # Weyl key increments
ROUNDS = 10


def philox4x32(k0, k1, x0, x1, x2, x3):
    """Philox4x32-10 on uint32 arrays; returns the four output words."""
    for r in range(ROUNDS):
        if r:
            k0, k1 = k0 + W0, k1 + W1
        lo0, hi0 = lax.mul(M0, x0), lax.mulhi(M0, x0)
        lo1, hi1 = lax.mul(M1, x2), lax.mulhi(M1, x2)
        x0, x1, x2, x3 = hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0
    return x0, x1, x2, x3


def _seed(seed):
    seed = jnp.asarray(seed)
    hi = (seed.astype(jnp.uint64) >> 32).astype(jnp.uint32) if seed.dtype.itemsize == 8 else jnp.uint32(0)
    return jnp.stack([jnp.asarray(hi, jnp.uint32), seed.astype(jnp.uint32)])


def _split(key, shape):
    n = math.prod(shape)
    c = jnp.arange(n, dtype=jnp.uint32)
    z = jnp.zeros(n, jnp.uint32)
    y0, y1, _, _ = philox4x32(key[0], key[1], z, z, z, c)
    return jnp.stack([y0, y1], axis=-1).reshape(tuple(shape) + (2,))


def _fold_in(key, data):
    z = jnp.uint32(0)
    y0, y1, _, _ = philox4x32(key[0], key[1], z, z, z, jnp.asarray(data, jnp.uint32))
    return jnp.stack([y0, y1])


def words32(key, shape):
    """32-bit words for a draw of `shape`, laid out as described in the module docstring."""
    shape = tuple(shape)
    n = math.prod(shape)
    lane = shape[-1] if len(shape) >= 2 else 1
    lead = n // lane if lane else 0
    groups = -(-lead // 4)
    c = jnp.arange(groups * lane, dtype=jnp.uint32)
    z = jnp.zeros_like(c)
    w = jnp.stack(philox4x32(key[0], key[1], c, z, z, z))  # (4, groups * lane)
    w = w.reshape(4, groups, lane).transpose(1, 0, 2).reshape(groups * 4, lane)[:lead]
    return w.reshape(shape)


def _random_bits(key, bit_width, shape):
    if bit_width == 64:
        w = words32(key, tuple(shape) + (2,))
        return (w[..., 0].astype(jnp.uint64) << 32) | w[..., 1].astype(jnp.uint64)
    w = words32(key, shape)
    return w if bit_width == 32 else w.astype({8: jnp.uint8, 16: jnp.uint16}[bit_width])


PHILOX = jxr.define_prng_impl(
    key_shape=(2,), seed=_seed, split=_split, random_bits=_random_bits, fold_in=_fold_in,
    name="pypomp_philox4x32_dense", tag="ppx")
