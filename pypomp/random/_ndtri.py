"""``jax.scipy.special.ndtri``, evaluating only the needed piece on a float32 scalar.

The scalar path restates JAX's float32 ``_ndtri``, ``erf_inv_core`` and
``_ndtri_grad`` (``jax._src.lax.erf`` and ``jax._src.scipy.special``, JAX 0.12) with
copies of their constants, so it doesn't import JAX's private modules. If JAX
changes its approximation, the copy stops matching it:
tests/test_random/test_unbatched.py checks it against JAX bit for bit.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy import special as jsp_special

from ._branch import where_lazy

# erf_inv_core's float32 ndtri constants: pieces split at w = 3.5 and 18.0625, the
# central piece's polynomial in w - 1.75 and the tails' in sqrt(w) - shift, and
# sqrt(2) truncated so that _K_H * z_h is exact.
_SPLITS = (3.5, 18.0625)
_C1 = 1.75
_SHIFTS = (3.0, 7.0)
_K_H = float.fromhex("0x1.6a0000p0")
_UNROLL = 9

# JAX's _NDTRI32_P1, _NDTRI32_P2 and _NDTRI32_P3: the coefficients, in ascending
# degree, of the central piece and the two tails.
_NDTRI32_P = (
    (
        float.fromhex("0x1.dbf6cep0"),
        float.fromhex("0x1.6aa63p-2"),
        float.fromhex("-0x1.cb58eap-11"),
        float.fromhex("-0x1.608b44p-9"),
        float.fromhex("0x1.2b210ep-12"),
        float.fromhex("0x1.6d1694p-16"),
        float.fromhex("-0x1.f460ecp-18"),
        float.fromhex("0x1.fd779p-24"),
        float.fromhex("0x1.0d3e14p-23"),
    ),
    (
        float.fromhex("-0x1.e35432p-3"),
        float.fromhex("0x1.00abfap-8"),
        float.fromhex("0x1.3d1088p-7"),
        float.fromhex("-0x1.3f797ep-8"),
        float.fromhex("0x1.aaacbep-10"),
        float.fromhex("-0x1.413624p-12"),
        float.fromhex("-0x1.354cacp-14"),
        float.fromhex("0x1.3f1a2ap-14"),
        float.fromhex("-0x1.1c2152p-16"),
    ),
    (
        float.fromhex("-0x1.7931ap-3"),
        float.fromhex("0x1.934bc8p-7"),
        float.fromhex("-0x1.7c3996p-11"),
        float.fromhex("0x1.0fafbep-17"),
        float.fromhex("0x1.244c8cp-17"),
        float.fromhex("-0x1.07e81p-19"),
        float.fromhex("0x1.ad9bd8p-22"),
        float.fromhex("-0x1.08fd02p-23"),
        float.fromhex("0x1.3cee08p-26"),
    ),
)


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


@jax.custom_jvp
def _ndtri_grad(p: jax.Array, x: jax.Array) -> jax.Array:
    """JAX's ``_ndtri_grad``: ndtri'(p) for ``x = ndtri(p)``.

    Within sqrt(2) of 0 this is sqrt(2 pi) exp(x^2 / 2); further out it is
    sqrt(pi / 2) erfcx(|x| / sqrt(2)) / min(p, 1 - p), which doesn't amplify the
    error in `x`. For an argument of at least 1, the public ``erfcx`` evaluates
    JAX's ``erfcx_large``, as ``_ndtri_grad`` does.
    """
    p = jnp.asarray(p, dtype=x.dtype)
    a = jnp.where(p > 0.5, 1.0 - p, p)
    u = lax.abs(x) * math.sqrt(0.5)
    is_tail = u > 1.0

    def central() -> jax.Array:
        x_central = jnp.where(is_tail, 0.0, x)
        return math.sqrt(2.0 * math.pi) * lax.exp(
            0.5 * lax.square(x_central), accuracy=lax.AccuracyMode.HIGHEST
        )

    def tail() -> jax.Array:
        is_finite_tail = is_tail & (a > 0.0)
        u_tail = jnp.where(is_finite_tail, u, 1.0)
        a_safe = jnp.where(is_finite_tail, a, 0.5)
        tail_num = math.sqrt(0.5 * math.pi) * jsp_special.erfcx(u_tail)
        return jnp.where(a == 0.0, np.inf, tail_num / a_safe)

    return where_lazy(is_tail, tail, central)


@_ndtri_grad.defjvp
def _ndtri_grad_jvp(primals, tangents):
    # As JAX's: x is ndtri(p), not an independent input, so only p_dot counts.
    (p, x), (p_dot, _) = primals, tangents
    g = _ndtri_grad(p, x)
    return g, p_dot * ((x * g) * g)


def _half_mantissa(x: jax.Array) -> jax.Array:
    """Zero the low 12 of float32 `x`'s 24 significand bits, so that the product of
    two such values is exact."""
    x_u = lax.bitcast_convert_type(x, np.uint32)
    return lax.bitcast_convert_type(x_u & np.uint32(0xFFFFF000), np.float32)


def _consts(cs: tuple[float, ...], like: jax.Array) -> list[jax.Array]:
    return [jnp.full_like(like, c) for c in cs]
