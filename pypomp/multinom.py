from functools import partial
import jax.numpy as jnp
from jax import jit
import numpy as np
import warnings


from collections.abc import Sequence
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src.interpreters import batching
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import lax
from jax._src.numpy.util import _arraylike, check_arraylike, promote_dtypes_inexact
from jax.scipy.special import gammaln
from jax._src.typing import Array, ArrayLike, DTypeLike
import jax.random
from jax._src import prng

RealArray = ArrayLike
IntegerArray = ArrayLike
DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]


def default_prng_impl():
    """Get the default PRNG implementation.

    The default implementation is determined by ``config.jax_default_prng_impl``,
    which specifies it by name.
    """
    impl_name = config.default_prng_impl.value
    assert impl_name in prng.prngs, impl_name
    return prng.prngs[impl_name]


def _check_prng_key(
    name: str, key: ArrayLike, *, allow_batched: bool = False
) -> tuple[Array, bool]:
    if isinstance(key, Array) and dtypes.issubdtype(key.dtype, dtypes.prng_key):
        wrapped_key = key
        wrapped = False
    elif _arraylike(key):
        # Call random_wrap here to surface errors for invalid keys.
        wrapped_key = prng.random_wrap(key, impl=default_prng_impl())
        wrapped = True
        if config.legacy_prng_key.value == "error":
            raise ValueError(
                "Legacy uint32 key array passed as key to jax.random function. "
                "Please create keys using jax.random.key(). If use of a raw key array "
                'was intended, set jax_legacy_prng_key="allow".'
            )
        elif config.legacy_prng_key.value == "warn":
            warnings.warn(
                "Legacy uint32 key array passed as key to jax.random function. "
                "Please create keys using jax.random.key(). If use of a raw key array "
                'was intended, set jax_legacy_prng_key="allow".',
                stacklevel=2,
            )
        elif config.enable_custom_prng.value:
            # TODO(jakevdp): possibly remove this warning condition.
            warnings.warn(
                "Raw arrays as random keys to jax.random functions are deprecated. "
                "Assuming valid threefry2x32 key for now.",
                FutureWarning,
            )
    else:
        raise TypeError(f"unexpected PRNG key type {type(key)}")

    if (not allow_batched) and wrapped_key.ndim:
        raise ValueError(
            f"{name} accepts a single key, but was given a key array of"
            f" shape {np.shape(key)} != (). Use jax.vmap for batching."
        )

    return wrapped_key, wrapped


def _isnan(x: ArrayLike) -> Array:
    return lax.ne(x, x)


def _check_shape(name: str, shape: Shape, *param_shapes) -> None:
    if param_shapes:
        shape_ = lax.broadcast_shapes(shape, *param_shapes)  # type: ignore
        if shape != shape_:
            msg = (
                "{} parameter shapes must be broadcast-compatible with shape "
                "argument, and the result of broadcasting the shapes must equal "
                "the shape argument, but got result {} for shape argument {}."
            )
            raise ValueError(msg.format(name, shape_, shape))


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _binomial_inversion(key, count, prob, shape, dtype, max_iters):
    if config.enable_checks.value:
        assert dtypes.issubdtype(prob.dtype, np.floating)

    log1minusprob = jnp.log1p(-prob)

    def body_fn(carry):
        i, num_geom, geom_sum, key = carry
        subkey, key = jax.random.split(key)
        num_geom_out = lax.select(geom_sum <= count, num_geom + 1, num_geom)
        u = jax.random.uniform(subkey, shape, prob.dtype)
        geom = jnp.ceil(jnp.log(u) / log1minusprob)
        geom_sum = geom_sum + geom
        return i + 1, num_geom_out, geom_sum, key

    def cond_fn(carry):
        i, geom_sum = carry[0], carry[2]
        return (geom_sum <= count).any() & (i < max_iters)

    num_geom_init = lax.full_like(prob, 0, prob.dtype, shape)
    geom_sum_init = lax.full_like(prob, 0, prob.dtype, shape)
    carry = (0, num_geom_init, geom_sum_init, key)
    k = lax_control_flow.while_loop(cond_fn, body_fn, carry)[1]
    return (k - 1).astype(dtype)


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _btpe(key, count, q, shape, dtype, max_iters):
    """BTPE binomial sampler (Binomial, Triangle, Parallelogram, Exponential).

    This implements the BTPE acceptance-rejection algorithm of
    Kachitvichyanukul and Schmeiser (1988) for sampling from a Binomial(n, p)
    distribution when n * min(p, 1 - p) is large. Here ``q`` is expected to be
    ``min(p, 1 - p)`` (i.e., q <= 0.5), matching upstream call sites.

    Parameters
    ----------
    key : PRNG key
    count : array-like of non-negative counts (n)
    q : array-like of probabilities with q <= 0.5 (i.e., min(p, 1 - p))
    shape : output broadcast shape
    dtype : floating dtype of the output
    max_iters : maximum iterations for internal rejection loop

    Notes
    -----
    This is a scaffold for the BTPE algorithm. It sets up broadcasted inputs
    and precomputes commonly used constants. The acceptance-rejection loop will
    be filled in a subsequent edit.
    """
    # Validate and broadcast shapes/dtypes similar to other samplers
    if shape is None:
        shape = jnp.broadcast_shapes(np.shape(count), np.shape(q))
    else:
        _check_shape("btpe", shape, np.shape(count), np.shape(q))

    (q,) = promote_dtypes_inexact(q)
    n = lax.convert_element_type(count, q.dtype)
    n = jnp.broadcast_to(n, shape)
    q = jnp.broadcast_to(q, shape)
    p = q  # by convention here, q is min(p, 1 - p); treat as p <= 0.5

    # Guard against invalid inputs; mirror behavior of other samplers
    n = jnp.floor(n)
    _invalid_n = _isnan(n) | (n < 0.0)
    _invalid_q = _isnan(p) | (p < 0.0) | (p > 0.5)

    # Precompute constants used by BTPE
    one_minus_p = 1.0 - p
    mean = n * p
    var = n * p * one_minus_p

    # Mode of Binomial (greatest integer <= (n + 1) * p)
    mode = jnp.floor((n + 1.0) * p)

    # Set up BTPE envelope parameters (per Kachitvichyanukul & Schmeiser, 1988)
    sigma = jnp.sqrt(var)
    p1 = jnp.floor(2.195 * sigma - 4.6 * one_minus_p) + 0.5
    xm = mode + 0.5
    xl = xm - p1
    xr = xm + p1
    c = 0.134 + 20.5 / (15.3 + mode)
    a_l = (mean + p - xl) / (mean + p - xl * p)
    lambda_l = a_l * (1.0 + a_l / 2.0)
    a_r = (xr - (mean + p)) / (xr * one_minus_p)
    lambda_r = a_r * (1.0 + a_r / 2.0)
    p2 = p1 * (1.0 + 2.0 * c)
    p3 = p2 + c / lambda_l
    p4 = p3 + c / lambda_r

    # Utility: log Binomial PMF up to constant
    def log_binom_pmf(k):
        k = jnp.clip(k, 0.0, n)
        return (
            gammaln(n + 1.0)
            - gammaln(k + 1.0)
            - gammaln(n - k + 1.0)
            + k * jnp.log(p + 1e-20)
            + (n - k) * jnp.log(one_minus_p + 1e-20)
        )

    log_fm = log_binom_pmf(mode)

    def body_fn(carry):
        key, accepted_mask, result, i = carry
        key_u, key_v, key_w, key_next = jax.random.split(key, 4)
        u = jax.random.uniform(key_u, shape, dtype=jnp.float32) * p4
        v = jax.random.uniform(key_v, shape, dtype=jnp.float32)
        w = jax.random.uniform(key_w, shape, dtype=jnp.float32)

        # Region 1: central triangle
        cond_r1 = u <= p1
        y_r1 = jnp.floor(xm - p1 * v + u)

        # Region 2: parallelogram
        cond_r2 = (u > p1) & (u <= p2)
        x_r2 = xl + (u - p1) / c
        v_r2 = v * c + 1.0 - jnp.abs(mode - x_r2 + 0.5) / p1
        y_r2 = jnp.floor(x_r2)
        ok_r2 = v_r2 <= 1.0

        # Region 3: left exponential
        cond_r3 = (u > p2) & (u <= p3)
        y_r3 = jnp.floor(xl + jnp.log(v + 1e-20) / lambda_l)
        _v_r3 = v * (u - p2) * lambda_l
        ok_r3 = y_r3 >= 0.0

        # Region 4: right exponential
        _cond_r4 = u > p3
        y_r4 = jnp.floor(xr - jnp.log(v + 1e-20) / lambda_r)
        _v_r4 = v * (u - p3) * lambda_r
        ok_r4 = y_r4 <= n

        y_prop = jnp.where(
            cond_r1, y_r1, jnp.where(cond_r2, y_r2, jnp.where(cond_r3, y_r3, y_r4))
        )

        prelim_ok = jnp.where(
            cond_r1,
            jnp.ones_like(v, dtype=jnp.bool_),
            jnp.where(cond_r2, ok_r2, jnp.where(cond_r3, ok_r3, ok_r4)),
        )

        # Acceptance via exact log-PMF ratio
        log_py = log_binom_pmf(y_prop)
        accept = (jnp.log(w + 1e-20) <= (log_py - log_fm)) & prelim_ok
        accept &= (y_prop >= 0.0) & (y_prop <= n)

        take_update = (~accepted_mask) & accept
        y_prop_cast = lax.convert_element_type(y_prop, dtype)
        result = jnp.where(take_update, y_prop_cast, result)
        accepted_mask = accepted_mask | accept
        return (key_next, accepted_mask, result, i + 1)

    def cond_fn(carry):
        _, accepted_mask, _, i = carry
        return (~accepted_mask.all()) & (i < max_iters)

    accepted0 = jnp.zeros(shape, dtype=jnp.bool_)
    result0 = jnp.zeros(shape, dtype=dtype)
    carry0 = (key, accepted0, result0, 0)
    key, accepted_out, result_out, _ = lax_control_flow.while_loop(
        cond_fn, body_fn, carry0
    )

    # Fallback to mode where unaccepted (extremely rare under reasonable max_iters)
    mode_b = lax.convert_element_type(jnp.broadcast_to(mode, shape), dtype)
    accepted_out_arr = jnp.asarray(accepted_out, dtype=jnp.bool_)
    result_out_arr = jnp.asarray(result_out)
    mode_b_arr = jnp.asarray(mode_b)
    result_out = jnp.where(accepted_out_arr, result_out_arr, mode_b_arr)
    return result_out


@partial(jit, static_argnums=(3, 4), inline=True)
def _binomial(key, count, prob, shape, dtype) -> Array:
    # The implementation matches TensorFlow and TensorFlow Probability:
    # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_binomial_op.cc
    # and tensorflow_probability.substrates.jax.distributions.Binomial
    # For n * p < 10, we use the binomial inverse algorithm; otherwise btrs.
    if shape is None:
        shape = jnp.broadcast_shapes(np.shape(count), np.shape(prob))
    else:
        _check_shape("binomial", shape, np.shape(count), np.shape(prob))
    (prob,) = promote_dtypes_inexact(prob)
    count = lax.convert_element_type(count, prob.dtype)
    count = jnp.broadcast_to(count, shape)
    prob = jnp.broadcast_to(prob, shape)
    p_lt_half = prob < 0.5
    q = lax.select(p_lt_half, prob, 1.0 - prob)
    count_nan_or_neg = _isnan(count) | (count < 0.0)
    count_inf = jnp.isinf(count)
    q_is_nan = _isnan(q)
    q_l_0 = q < 0.0
    q = lax.select(q_is_nan | q_l_0, lax.full_like(q, 0.01), q)
    use_inversion = count_nan_or_neg | (count * q <= 10.0)

    # consistent with np.random.binomial behavior for float count input
    count = jnp.floor(count)

    count_inv = lax.select(use_inversion, count, lax.full_like(count, 0.0))
    count_btrs = lax.select(use_inversion, lax.full_like(count, 1e4), count)
    q_btrs = lax.select(use_inversion, lax.full_like(q, 0.5), q)
    max_iters = dtype.type(dtypes.finfo(dtype).max)
    samples = lax.select(
        use_inversion,
        _binomial_inversion(key, count_inv, q, shape, dtype, max_iters),
        _btpe(key, count_btrs, q_btrs, shape, dtype, max_iters),
    )
    # ensure nan q always leads to nan output and nan or neg count leads to nan
    # as discussed in https://github.com/jax-ml/jax/pull/16134#pullrequestreview-1446642709
    invalid = q_l_0 | q_is_nan | count_nan_or_neg
    samples = lax.select(
        invalid,
        jnp.full_like(samples, np.nan, dtype),
        samples,
    )

    # +inf count leads to inf
    samples = lax.select(
        count_inf & (~invalid),
        jnp.full_like(samples, np.inf, dtype),
        samples,
    )

    samples = lax.select(
        p_lt_half | count_nan_or_neg | q_is_nan | count_inf,
        samples,
        count.astype(dtype) - samples,
    )
    return samples


def binomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array:
    r"""Sample Binomial random values with given shape and float dtype.

    The values are returned according to the probability mass function:

    .. math::
        f(k;n,p) = \binom{n}{k}p^k(1-p)^{n-k}

    on the domain :math:`0 < p < 1`, and where :math:`n` is a nonnegative integer
    representing the number of trials and :math:`p` is a float representing the
    probability of success of an individual trial.

    Args:
      key: a PRNG key used as the random key.
      n: a float or array of floats broadcast-compatible with ``shape``
        representing the number of trials.
      p: a float or array of floats broadcast-compatible with ``shape``
        representing the probability of success of an individual trial.
      shape: optional, a tuple of nonnegative integers specifying the result
        shape. Must be broadcast-compatible with ``n`` and ``p``.
        The default (None) produces a result shape equal to ``np.broadcast(n, p).shape``.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).

    Returns:
      A random array with the specified dtype and with shape given by
      ``np.broadcast(n, p).shape``.
    """
    key, _ = _check_prng_key("binomial", key)
    check_arraylike("binomial", n, p)
    dtype = dtypes.canonicalize_dtype(float if dtype is None else dtype)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            f"dtype argument to `binomial` must be a float dtype, got {dtype}"
        )
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _binomial(key, n, p, shape, dtype)


# Functions related to key reuse checking
random_clone_p = core.Primitive("random_clone")
dispatch.simple_impl(random_clone_p)
random_clone_p.def_abstract_eval(lambda x: x)
batching.defvectorized(random_clone_p)
# mlir.register_lowering(random_clone_p, lambda _, k: [k])


def multinomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    *,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    unroll: int | bool = 1,
):
    r"""Sample from a multinomial distribution.

    The probability mass function is

    .. math::
        f(x;n,p) = \frac{n!}{x_1! \ldots x_k!} p_1^{x_1} \ldots p_k^{x_k}

    Args:
      key: PRNG key.
      n: number of trials. Should have shape broadcastable to ``p.shape[:-1]``.
      p: probability of each outcome, with outcomes along the last axis.
      shape: optional, a tuple of nonnegative integers specifying the result batch
        shape, that is, the prefix of the result shape excluding the last axis.
        Must be broadcast-compatible with ``p.shape[:-1]``. The default (None)
        produces a result shape equal to ``p.shape``.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
      unroll: optional, unroll parameter passed to :func:`jax.lax.scan` inside the
        implementation of this function.

    Returns:
      An array of counts for each outcome with the specified dtype and with shape
        ``p.shape`` if ``shape`` is None, otherwise ``shape + (p.shape[-1],)``.
    """

    key, _ = _check_prng_key("multinomial", key)
    check_arraylike("multinomial", n, p)
    n, p = promote_dtypes_inexact(n, p)

    if shape is None:
        shape = p.shape
    n = jnp.broadcast_to(n, shape[:-1])
    p = jnp.broadcast_to(p, shape)

    def f(remainder, ratio_key):
        ratio, key = ratio_key
        count = binomial(key, remainder, ratio.clip(0, 1), dtype=remainder.dtype)
        return remainder - count, count

    p = jnp.moveaxis(p, -1, 0)

    remaining_probs = lax_control_flow.cumsum(p, 0, reverse=True)
    ratios = p / jnp.where(remaining_probs == 0, 1, remaining_probs)

    keys = jax.random.split(key, ratios.shape[0])
    remainder, counts = lax_control_flow.scan(f, n, (ratios, keys), unroll=unroll)
    # final remainder should be zero

    return jnp.moveaxis(counts, 0, -1).astype(dtype)


def normal_approx(
    key: jax.Array, n: jax.Array, p: jax.Array, shape: tuple
) -> jax.Array:
    ntimesp = n * p
    draws = jnp.round(
        jnp.sqrt(ntimesp * (1 - p)) * jax.random.normal(key, shape) + ntimesp + 1 / 2
    )
    return jnp.maximum(draws, 0)


def simple_multinomial(
    key: jax.Array, n: jax.Array, p: jax.Array, shape: tuple | None = None
) -> jax.Array:
    r"""Sample from a multinomial distribution.
    The probability mass function is
    .. math::
      f(x;n,p) = \frac{n!}{x_1! \ldots x_k!} p_1^{x_1} \ldots p_k^{x_k}
    Args:
    key: PRNG key.
    n: number of trials. Should have shape broadcastable to ``p.shape[:-1]``.
    p: probability of each outcome, with outcomes along the last axis.
    shape: optional, a tuple of nonnegative integers specifying the result batch
      shape, that is, the prefix of the result shape excluding the last axis.
      Must be broadcast-compatible with ``p.shape[:-1]``. The default (None)
      produces a result shape equal to ``p.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    Returns:
    An array of counts for each outcome with the specified dtype and with shape
      ``p.shape`` if ``shape`` is None, otherwise ``shape + (p.shape[-1],)``.
    """

    # key, _ = _check_prng_key("multinomial", key)
    # jax._src.numpy.util.check_arraylike("multinomial", n, p)
    # n, p = jax._src.numpy.util.promote_dtypes_inexact(n, p)

    def f(remainder, ratio_key):
        ratio, key = ratio_key
        # normal approximation when |1-2p|/sqrt(np(1-p)) < 0.3 by berry-esseen."
        count = normal_approx(key, remainder, ratio, shape or ())
        # count = jax.lax.cond(np.abs(1-2*ratio)/np.sqrt(remainder * ratio * (1-ratio)) < 0.3,
        #              normal_approx,
        #              jax.random.binomial,
        #              key, remainder, ratio, shape)
        # count = jax.random.binomial(key, remainder, ratio, shape)
        return remainder - count, count

    p_shape = jnp.shape(p)

    if shape is None:
        shape = p_shape[:-1]

    n = jnp.broadcast_to(n, shape)
    p = jnp.broadcast_to(p, (*shape, p_shape[-1]))

    p = jnp.moveaxis(p, -1, 0)

    remaining_probs = jax.lax.cumsum(p, 0, reverse=True)
    ratios = p / jnp.where(remaining_probs == 0, 1, remaining_probs)

    keys = jax.random.split(key, ratios.shape[0])
    remainder, counts = jax.lax.scan(f, n, (ratios, keys), unroll=True)
    # final remainder should be zero

    counts = jnp.moveaxis(counts, 0, -1)

    return counts
