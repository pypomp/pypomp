"""``jax.scipy.special.ndtri``, evaluating only the needed piece on a float32 scalar.

The scalar path restates JAX's float32 ``erf_inv_core`` around its coefficient tables;
tests/test_random/test_unbatched.py checks it against JAX bit for bit.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.lax.erf import _half_mantissa, _NDTRI32_P
from jax._src.scipy.special import _ndtri_grad
from jax.scipy import special as jsp_special

# erf_inv_core's float32 ndtri constants: pieces split at w = 3.5 and 18.0625, the
# central piece's polynomial in w - 1.75 and the tails' in sqrt(w) - shift, and
# sqrt(2) truncated so that _K_H * z_h is exact.
_SPLITS = (3.5, 18.0625)
_C1 = 1.75
_SHIFTS = (3.0, 7.0)
_K_H = float.fromhex("0x1.6a0000p0")
_UNROLL = 9


def ndtri(p: jax.Array) -> jax.Array:
    """Return ``jax.scipy.special.ndtri(p)``, bit for bit.

    On a float32 scalar (or one-element array) this is a ``lax.cond`` over the
    pieces of JAX's approximation, which computes only the piece `p` falls in: the
    CuTe rproc backend traces one particle at a time and turns it into a branch.
    """
    if jnp.size(p) != 1 or jnp.result_type(p) != jnp.float32:
        return jsp_special.ndtri(p)
    return _ndtri_scalar(jnp.reshape(p, ())).reshape(jnp.shape(p))


@jax.custom_jvp
def _ndtri_scalar(p: jax.Array) -> jax.Array:
    """JAX's ``_ndtri`` and float32 ``erf_inv_core``, with the pieces as branches."""
    out_of_bounds = (p < 0) | (p > 1) | jnp.isnan(p)
    is_edge = (p == 0) | (p == 1)
    p_safe = jnp.where(out_of_bounds | is_edge, 0.5, p)
    a_safe = jnp.where(p_safe > 0.5, 1 - p_safe, p_safe)
    x = 2 * (p_safe - 0.5)
    q = 2 * a_safe
    w = -lax.log(q * (1.0 + abs(x)))

    def central() -> jax.Array:
        return x * lax.polynomial(w - _C1, _consts(_NDTRI32_P[0], w), unroll=_UNROLL)

    def tail(k: int) -> jax.Array:
        z = lax.sqrt(w)
        z_h = _half_mantissa(z)
        z_l = (w - z_h * z_h) / (z + z_h)
        poly = lax.polynomial(
            z - _SHIFTS[k], _consts(_NDTRI32_P[k + 1], w), unroll=_UNROLL
        )
        mag = _K_H * z_h + lax.optimization_barrier(_K_H * z_l + poly)
        sign_x = lax.select(x < 0.0, jnp.full_like(x, -1.0), jnp.full_like(x, 1.0))
        return sign_x * mag

    r = lax.cond(
        w < _SPLITS[0],
        central,
        lambda: lax.cond(w < _SPLITS[1], lambda: tail(0), lambda: tail(1)),
    )
    return jnp.where(
        out_of_bounds,
        jnp.nan,
        jnp.where(p == 0, -jnp.inf, jnp.where(p == 1, jnp.inf, r)),
    )


@_ndtri_scalar.defjvp
def _ndtri_scalar_jvp(primals, tangents):
    (p,), (p_dot,) = primals, tangents
    x = _ndtri_scalar(p)
    return x, p_dot * _ndtri_grad(p, x)


def _consts(cs: tuple[float, ...], like: jax.Array) -> list[jax.Array]:
    return [jnp.full_like(like, c) for c in cs]
