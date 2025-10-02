import jax
import jax.numpy as jnp
import typing
from functools import partial
import math
from collections.abc import Sequence

from jax._src.lax import lax
from jax.random import normal, uniform, exponential
from jax._src import core
import numpy as np
from jax._src import dtypes
from jax._src import prng
from jax._src import xla_bridge
from jax._src.api import jit, vmap
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import special as lax_special
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.random import _check_shape, _check_prng_key, _split, _key_impl, split
from jax._src import config
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

# ------------------------------------------------------------------------------
# JAX compatibility shim: older JAX versions may not expose
# dtypes.check_and_canonicalize_user_dtype. Define a fallback when missing.
# ------------------------------------------------------------------------------
if not hasattr(dtypes, "check_and_canonicalize_user_dtype"):
    try:
        _canonicalize = dtypes.canonicalize_dtype  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - extremely old JAX; best-effort fallback

        def _canonicalize(x):
            return x

    def check_and_canonicalize_user_dtype(dtype):
        # Older JAX separated "check" and "canonicalize"; for our usage where we
        # pass builtins like int/float or numpy dtypes, canonicalizing is
        # sufficient. If dtype is None, mirror JAX behavior of passing it through.
        return None if dtype is None else _canonicalize(dtype)

else:
    check_and_canonicalize_user_dtype = dtypes.check_and_canonicalize_user_dtype  # type: ignore[attr-defined]


RealArray = ArrayLike
IntegerArray = ArrayLike

DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]


def normal_approx_binomial(
    key: jax.Array, n: jax.Array, p: jax.Array, shape: tuple, dtype: jax.Array
) -> jax.Array:
    ntimesp = n * p
    draws = jnp.round(
        jnp.sqrt(ntimesp * (1 - p)) * jax.random.normal(key, shape) + ntimesp + 1 / 2
    )
    return lax.clamp(0, draws, n).astype(dtype)


def normal_approx_poisson(
    key: jax.Array, lam: jax.Array, shape: tuple, dtype: jax.Array
) -> jax.Array:
    draws = jnp.round(jnp.sqrt(lam) * jax.random.normal(key, shape) + lam + 1 / 2)
    return jnp.maximum(draws, 0).astype(dtype)


def faster_approx_multinomial(
    key: jax.Array,
    n: jax.Array,
    p: jax.Array,
    shape: tuple | None = None,
    dtype: jax.Array = jnp.float32,
) -> jax.Array:
    r"""
    Sample from a multinomial distribution.
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
        count = normal_approx_binomial(key, remainder, ratio, shape or (), dtype)
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


def faster_approx_poisson(
    key: jax.Array,
    lam: jax.Array,
    shape: tuple | None = None,
    *,
    max_k: int | None = None,
    percentile: float | None = None,
    normal_threshold: float = 10.0,
) -> jax.Array:
    """
    Draw samples from either a discretized Normal or a right-truncated
    Poisson(lambda), selected via a JAX-compatible conditional.

    - If ``lam >= normal_threshold``: sample from a Normal(lambda, lambda),
      discretized by rounding to the nearest nonnegative integer (no upper
      truncation).
    - Otherwise: sample from a right-truncated Poisson(lambda) on
      {0, 1, ..., K} via inverse CDF. Exactly one of ``max_k`` or
      ``percentile`` must be provided to determine K.

    Args:
        key: PRNG key.
        lam: Poisson rate (scalar JAX array).
        shape: Optional batch shape for the returned samples. Default None → ().
        max_k: Truncate Poisson support to {0, 1, ..., max_k} when using the
            truncated-Poisson path.
        percentile: When using the truncated-Poisson path and ``max_k`` is not
            provided, choose the smallest K such that CDF_Poisson(K; lam) >=
            ``percentile`` and truncate to {0, ..., K}.
        normal_threshold: Threshold at which to switch to the Normal
            approximation. Default 10.0.

    Returns:
        Integer array of samples with shape ``shape`` (or scalar if None).
    """
    if (max_k is None) == (percentile is None):
        raise ValueError("Provide exactly one of max_k or percentile.")

    if shape is None:
        shape = ()

    lam_scalar = lam.astype(jnp.float32)

    # Decide using a JAX-compatible conditional BEFORE any Poisson-specific work
    use_normal = lam_scalar >= jnp.asarray(normal_threshold, dtype=jnp.float32)

    def sample_normal(_):
        normal_samples = (
            jax.random.normal(key, shape=shape) * jnp.sqrt(lam_scalar) + lam_scalar
        )
        samples = jnp.maximum(jnp.round(normal_samples), 0.0)
        return samples.astype(jnp.int32)

    def sample_exact(_):
        # Determine truncation K
        if max_k is not None:
            K = int(max_k)
        else:
            # Find smallest K with CDF >= percentile using stable pmf recursion
            assert percentile is not None
            target = float(percentile)

            def cond_fun(state):
                k, pmf, cdf = state
                return jnp.logical_and(cdf < target, k < 100000)

            def body_fun(state):
                k, pmf, cdf = state
                next_k = k + 1
                next_pmf = pmf * lam_scalar / next_k
                next_cdf = cdf + next_pmf
                return (next_k, next_pmf, next_cdf)

            p0 = jnp.exp(-lam_scalar)
            k0, pmf0, cdf0 = (0, p0, p0)
            kf, _, _ = jax.lax.while_loop(cond_fun, body_fun, (k0, pmf0, cdf0))
            # If p0 already >= target, kf will be 0; otherwise it is smallest k s.t. CDF>=target
            K = int(kf)

        # Build pmf for k=0..K using stable recursion
        def scan_body(p_k, k_idx):
            next_p = p_k * lam_scalar / (k_idx + 1)
            return next_p, next_p

        p0 = jnp.exp(-lam_scalar)
        if K == 0:
            pmfs = jnp.array([p0])
        else:
            ks = jnp.arange(K)
            _, tail_pmfs = jax.lax.scan(scan_body, p0, ks)
            pmfs = jnp.concatenate([jnp.array([p0]), tail_pmfs])

        cdf = jnp.cumsum(pmfs)
        total_mass = cdf[-1]

        u = jax.random.uniform(key, shape=shape)
        thresh = u * total_mass

        def search_one(t):
            return jnp.searchsorted(cdf, t, side="left")

        idx = (
            jax.vmap(search_one)(thresh.ravel())
            if thresh.size
            else jnp.array([], dtype=jnp.int32)
        )
        idx = idx.reshape(thresh.shape)
        idx = jnp.minimum(idx, K)
        return idx.astype(jnp.int32)

    return jax.lax.cond(use_normal, sample_normal, sample_exact, operand=None)


def _isnan(x: ArrayLike) -> Array:
    return lax.ne(x, x)


def _stirling_approx_tail(k):
    stirling_tail_vals = jnp.array(
        [
            0.0810614667953272,
            0.0413406959554092,
            0.0276779256849983,
            0.02079067210376509,
            0.0166446911898211,
            0.0138761288230707,
            0.0118967099458917,
            0.0104112652619720,
            0.00925546218271273,
            0.00833056343336287,
        ],
        dtype=k.dtype,
    )
    use_tail_values = k <= 9
    k = lax.clamp(lax._const(k, 0.0), k, lax._const(k, 9.0))
    kp1sq = (k + 1) * (k + 1)
    approx = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1)
    k = jnp.floor(k)
    return lax.select(
        use_tail_values, stirling_tail_vals[jnp.asarray(k, dtype="int32")], approx
    )


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _binomial_inverse_cdf(key, count, prob, shape, dtype, max_iters):
    """
    Sample from binomial distribution using the inverse CDF method.
    This version computes the CDF incrementally, stopping as soon as the CDF exceeds the uniform sample.
    """
    if config.enable_checks.value:
        assert dtypes.issubdtype(prob.dtype, np.floating)

    def sample_one(key_single, count_single, prob_single):
        """Sample one binomial random variable using inverse CDF, step by step."""
        n_int = jnp.asarray(count_single, dtype=jnp.int32)

        # Handle edge cases
        edge_result = jnp.where(
            prob_single <= 0.0,
            0,
            jnp.where(prob_single >= 1.0, n_int, -1),  # -1 indicates no edge case
        )

        def compute_inverse_cdf():
            # Generate uniform random variable
            u = uniform(key_single, (), dtype=prob_single.dtype)

            # Compute PMF values using stable recursion
            log_1_minus_p = jnp.log1p(-prob_single)

            # Initial PMF: P(0) = (1-p)^n
            log_pmf_0 = count_single * log_1_minus_p
            pmf_0 = jnp.exp(log_pmf_0)

            def cond_fun(state):
                k, pmf, cdf, found, result = state
                # Continue if not found and k <= n
                return (~found) & (k <= n_int)

            def body_fun(state):
                k, pmf, cdf, found, result = state
                # For k=0, pmf is pmf_0, already set
                # For k>0, update pmf using recurrence
                next_pmf = jnp.where(
                    k == 0,
                    pmf,
                    pmf
                    * (
                        (count_single - k + 1)
                        / k
                        * prob_single
                        / (1 - prob_single + 1e-15)
                    ),
                )
                next_cdf = cdf + next_pmf
                # If not found and next_cdf >= u, set result to k
                this_found = (next_cdf >= u) & (~found)
                next_result = jnp.where(this_found, k, result)
                return (k + 1, next_pmf, next_cdf, found | this_found, next_result)

            # Initial state: k=0, pmf=pmf_0, cdf=pmf_0, found=False, result=0
            init_state = (
                jnp.array(0, dtype=jnp.int32),
                pmf_0,
                pmf_0,
                False,
                jnp.array(0, dtype=jnp.int32),
            )

            # Run the loop up to n_int+1 times (to cover all possible k)
            # Use while_loop for efficiency
            final_state = lax_control_flow.while_loop(cond_fun, body_fun, init_state)
            _, _, _, found, result = final_state

            # If not found after all, return n_int (should be rare, only if u > CDF(n))
            result = jnp.where(found, result, n_int)
            return result

        # Use edge case result if applicable, otherwise compute inverse CDF
        return jax.lax.cond(edge_result >= 0, lambda: edge_result, compute_inverse_cdf)

    # Vectorize the sampling function
    sample_fn = jax.vmap(sample_one, in_axes=(0, 0, 0))

    # Split keys and flatten arrays for vmap
    keys = split(key, count.size)
    keys_flat = keys.reshape(count.shape)
    count_flat = count.ravel()
    prob_flat = prob.ravel()
    keys_flat = keys_flat.ravel()

    # Sample and reshape
    samples_flat = sample_fn(keys_flat, count_flat, prob_flat)
    samples = samples_flat.reshape(count.shape)

    return samples.astype(dtype)


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _binomial_inversion(key, count, prob, shape, dtype, max_iters):
    if config.enable_checks.value:
        assert dtypes.issubdtype(prob.dtype, np.floating)

    log1minusprob = jnp.log1p(-prob)

    def body_fn(carry):
        i, num_geom, geom_sum, key = carry
        subkey, key = split(key)
        num_geom_out = lax.select(geom_sum <= count, num_geom + 1, num_geom)
        u = uniform(subkey, shape, prob.dtype)
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
def _btrs(key, count, prob, shape, dtype, max_iters):
    # transforman-rejection algorithm
    # https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
    stddev = jnp.sqrt(count * prob * (1 - prob))
    b = 1.15 + 2.53 * stddev
    a = -0.0873 + 0.0248 * b + 0.01 * prob
    c = count * prob + 0.5
    v_r = 0.92 - 4.2 / b
    r = prob / (1 - prob)
    alpha = (2.83 + 5.1 / b) * stddev
    m = jnp.floor((count + 1) * prob)

    def body_fn(carry):
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = split(key, 3)
        u = uniform(subkey_0, shape, prob.dtype)
        v = uniform(subkey_1, shape, prob.dtype)
        u = u - 0.5
        us = 0.5 - jnp.abs(u)
        accept1 = (us >= 0.07) & (v <= v_r)
        k = jnp.floor((2 * a / us + b) * u + c)
        reject = (k < 0) | (k > count)
        v = jnp.log(v * alpha / (a / (us * us) + b))
        ub = (
            (m + 0.5) * jnp.log((m + 1) / (r * (count - m + 1)))
            + (count + 1) * jnp.log((count - m + 1) / (count - k + 1))
            + (k + 0.5) * jnp.log(r * (count - k + 1) / (k + 1))
            + _stirling_approx_tail(m)
            + _stirling_approx_tail(count - m)
            - _stirling_approx_tail(k)
            - _stirling_approx_tail(count - k)
        )
        accept2 = v <= ub
        accept = accept1 | (~reject & accept2)
        k_out = lax.select(~accepted & ~reject, k, k_out)
        accepted |= accept
        return i + 1, k_out, accepted, key

    def cond_fn(carry):
        i, accepted = carry[0], carry[2]
        return (~accepted).any() & (i < max_iters)

    # k_init = lax.full_like(prob, -1, prob.dtype, shape)
    k_init = jnp.round(count * prob).astype(prob.dtype).reshape(shape)
    # key1, key2 = split(key)
    # k_init = normal_approx_binomial(key1, count, prob, shape, prob.dtype)
    carry = (0, k_init, jnp.full(shape, False, bool), key)
    return jnp.clip(
        lax_control_flow.while_loop(cond_fn, body_fn, carry)[1].astype(dtype), 0, count
    )


@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _binomial(key, count, prob, shape, dtype, max_rejections) -> Array:
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
    # use_inversion = count_nan_or_neg | (count * q <= 10.0)
    # TODO: might want to add an argument to choose which method to use
    use_inversion = False

    # consistent with np.random.binomial behavior for float count input
    count = jnp.floor(count)

    count_inv = lax.select(use_inversion, count, lax.full_like(count, 0.0))
    count_btrs = lax.select(use_inversion, lax.full_like(count, 1e4), count)
    q_btrs = lax.select(use_inversion, lax.full_like(q, 0.5), q)
    max_iters = max_rejections + 1
    samples = lax.select(
        use_inversion,
        _binomial_inverse_cdf(key, count_inv, q, shape, dtype, max_iters),
        _btrs(key, count_btrs, q_btrs, shape, dtype, max_iters),
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


def fast_approx_binomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    max_rejections: int = 2,
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
      max_rejections: the maximum number of rejections allowed.

    Returns:
      A random array with the specified dtype and with shape given by
      ``np.broadcast(n, p).shape``.
    """
    key, _ = _check_prng_key("binomial", key)
    check_arraylike("binomial", n, p)
    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            f"dtype argument to `binomial` must be a float dtype, got {dtype}"
        )
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _binomial(key, n, p, shape, dtype, max_rejections)


# Functions related to key reuse checking
# random_clone_p = core.Primitive("random_clone")
# dispatch.simple_impl(random_clone_p)
# random_clone_p.def_abstract_eval(lambda x: x)
# batching.defvectorized(random_clone_p)
# mlir.register_lowering(random_clone_p, lambda _, k: [k])


def fast_approx_multinomial(
    key: Array,
    n: RealArray,
    p: RealArray,
    *,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    unroll: int | bool = 1,
    max_rejections: int = 2,
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
      max_rejections: the maximum number of rejections allowed.

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
        count = fast_approx_binomial(
            key,
            remainder,
            ratio.clip(0, 1),
            dtype=remainder.dtype,
            max_rejections=max_rejections,
        )
        return remainder - count, count

    p = jnp.moveaxis(p, -1, 0)

    remaining_probs = lax_control_flow.cumsum(p, 0, reverse=True)
    ratios = p / jnp.where(remaining_probs == 0, 1, remaining_probs)
    ratios = ratios[:-1]

    keys = split(key, ratios.shape[0])
    remainder, counts = lax_control_flow.scan(f, n, (ratios, keys), unroll=unroll)
    # final remainder should be zero
    # Last set of counts is deterministic.
    # Not sure why JAX code normally draws a binomial for the last set of counts,
    # which I have confirmed is slower.
    counts = jnp.vstack([counts, remainder.reshape(1, -1)])
    return jnp.moveaxis(counts, 0, -1).astype(dtype)


@partial(jit, static_argnums=(2, 3, 4))
def _poisson_knuth(key, lam, shape, dtype, max_iters) -> Array:
    # Knuth's algorithm for generating Poisson random variates.
    # Reference:
    # https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables

    def body_fn(carry):
        i, k, rng, log_prod = carry
        rng, subkey = split(rng)
        k = lax.select(log_prod > -lam, k + 1, k)
        u = uniform(subkey, shape, np.float32)
        return i + 1, k, rng, log_prod + jnp.log(u)

    def cond_fn(carry):
        i, log_prod = carry[0], carry[3]
        return (log_prod > -lam).any() & (i < max_iters)

    k_init = lax.full_like(lam, 0, dtype, shape)
    log_rate_init = lax.full_like(lam, 0, np.float32, shape)
    k = lax_control_flow.while_loop(cond_fn, body_fn, (0, k_init, key, log_rate_init))[
        1
    ]
    return (k - 1).astype(dtype)


@partial(jit, static_argnums=(2, 3, 4))
def _poisson_rejection(key, lam, shape, dtype, max_iters) -> Array:
    # Transformed rejection due to Hormann.
    # Reference:
    # http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
    log_lam = lax.log(lam)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2)

    def body_fn(carry):
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = _split(key, 3)

        u = uniform(subkey_0, shape, lam.dtype) - 0.5
        v = uniform(subkey_1, shape, lam.dtype)
        u_shifted = 0.5 - abs(u)

        k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
        s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
        t = -lam + k * log_lam - lax_special.lgamma(k + 1)

        accept1 = (u_shifted >= 0.07) & (v <= v_r)
        reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
        accept2 = s <= t
        accept = accept1 | (~reject & accept2)

        k_out = lax.select(~accepted & ~reject, k, k_out)
        # k_out = lax.select(accept, k, k_out)
        accepted |= accept

        return i + 1, k_out, accepted, key

    def cond_fn(carry):
        i, k_out, accepted, key = carry
        return (~accepted).any() & (i < max_iters)

    # k_init = lax.full_like(lam, -1, lam.dtype, shape)
    key1, key2 = split(key)
    k_init = normal_approx_poisson(key1, lam, shape, lam.dtype)
    # k_init = jnp.round(lam).astype(lam.dtype).reshape(shape)
    accepted = lax.full_like(lam, False, np.dtype("bool"), shape)
    k = lax_control_flow.while_loop(cond_fn, body_fn, (0, k_init, accepted, key2))[1]
    return k.astype(dtype)


@partial(jit, static_argnums=(2, 3, 4))
def _poisson(key, lam, shape, dtype, max_rejections) -> Array:
    # The implementation matches TensorFlow and NumPy:
    # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_poisson_op.cc
    # https://github.com/numpy/numpy/blob/v1.18.3/numpy/random/src/distributions/distributions.c#L574
    # For lambda < 10, we use the Knuth algorithm; otherwise, we use transformed
    # rejection sampling.
    # use_knuth = _isnan(lam) | (lam < 10)
    use_knuth = False
    # TODO: might want to add an argument to choose which method to use
    # use_knuth = False
    lam_knuth = lax.select(use_knuth, lam, lax.full_like(lam, 0.0))
    # The acceptance probability for rejection sampling maxes out at 89% as
    # λ -> ∞, so pick some arbitrary large value.
    lam_rejection = lax.select(use_knuth, lax.full_like(lam, 1e5), lam)
    max_iters = max_rejections + 1
    result = lax.select(
        use_knuth,
        _poisson_knuth(key, lam_knuth, shape, dtype, max_iters),
        _poisson_rejection(key, lam_rejection, shape, dtype, max_iters),
    )
    result = jnp.clip(result, 0, jnp.inf)
    return lax.select(lam == 0, jnp.zeros_like(result), result)


def fast_approx_poisson(
    key: ArrayLike,
    lam: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeInt | None = None,
    max_rejections: int = 2,
) -> Array:
    r"""Sample Poisson random values with given shape and integer dtype.

    The values are distributed according to the probability mass function:

    .. math::
       f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    Where `k` is a non-negative integer and :math:`\lambda > 0`.

    Args:
      key: a PRNG key used as the random key.
      lam: rate parameter (mean of the distribution), must be >= 0. Must be broadcast-compatible with ``shape``
      shape: optional, a tuple of nonnegative integers representing the result
        shape. Default (None) produces a result shape equal to ``lam.shape``.
      dtype: optional, a integer dtype for the returned values (default int64 if
        jax_enable_x64 is true, otherwise int32).
      max_rejections: the maximum number of rejections allowed.

    Returns:
      A random array with the specified dtype and with shape given by ``shape`` if
      ``shape is not None, or else by ``lam.shape``.
    """
    key, _ = _check_prng_key("poisson", key)
    dtype = check_and_canonicalize_user_dtype(int if dtype is None else dtype)

    keys_dtype = typing.cast(prng.KeyTy, key.dtype)
    key_impl = keys_dtype._impl
    if key_impl is not prng.threefry_prng_impl:
        raise NotImplementedError(
            f"`poisson` is only implemented for the threefry2x32 RNG, not {key_impl}"
        )
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    else:
        shape = np.shape(lam)
    lam = jnp.broadcast_to(lam, shape)
    lam = lax.convert_element_type(lam, np.float32)
    return _poisson(key, lam, shape, dtype, max_rejections)


### JAX code follows


def _gamma_one(key: Array, alpha, max_rejections, log_space) -> Array:
    # Ref: A simple method for generating gamma variables, George Marsaglia and Wai Wan Tsang
    # The algorithm can also be founded in:
    # https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
    zero = lax._const(alpha, 0)
    one = lax._const(alpha, 1)
    two = lax._const(alpha, 2)
    minus_one = lax._const(alpha, -1)
    one_over_two = lax._const(alpha, 0.5)
    one_over_three = lax._const(alpha, 1.0 / 3.0)
    squeeze_const = lax._const(alpha, 0.0331)
    dtype = lax.dtype(alpha)

    zero = core.pvary(zero, tuple(core.typeof(alpha).vma))  # type: ignore[attr-defined]
    one = core.pvary(one, tuple(core.typeof(alpha).vma))  # type: ignore[attr-defined]
    minus_one = core.pvary(minus_one, tuple(core.typeof(alpha).vma))  # type: ignore[attr-defined]
    two = core.pvary(two, tuple(core.typeof(alpha).vma))  # type: ignore[attr-defined]

    # for alpha < 1, we boost alpha to alpha + 1 and get a sample according to
    #   Gamma(alpha) ~ Gamma(alpha+1) * Uniform()^(1 / alpha)
    # When alpha is very small, this boost can be problematic because it may result
    # in floating point underflow; for this reason we compute it in log space if
    # specified by the `log_space` argument:
    #   log[Gamma(alpha)] ~ log[Gamma(alpha + 1)] + log[Uniform()] / alpha
    # Note that log[Uniform()] ~ -Exponential(), but to avoid problems at x=0
    # exponential is computed in terms of log[1 - Uniform()]; we must account for this
    # so that log-space and non-log-space samples match.
    boost_mask = lax.ge(alpha, one)
    alpha_orig = alpha
    alpha = lax.select(boost_mask, alpha, lax.add(alpha, one))

    d = lax.sub(alpha, one_over_three)
    c = lax.div(one_over_three, lax.sqrt(d))

    def _cond_fn(state):
        n_rejections, _, X, V, U = state
        cond_reject = lax.bitwise_and(
            lax.ge(U, lax.sub(one, lax.mul(squeeze_const, lax.mul(X, X)))),
            lax.ge(
                lax.log(U),
                lax.add(
                    lax.mul(X, one_over_two),
                    lax.mul(d, lax.add(lax.sub(one, V), lax.log(V))),
                ),
            ),
        )
        return lax.bitwise_and(cond_reject, n_rejections < max_rejections)

    def _body_fn(state):
        n_rejections, key, _, _, _ = state

        def _next_kxv(kxv):
            key = kxv[0]
            key, subkey = _split(key)
            x = normal(subkey, (), dtype=dtype)
            v = lax.add(one, lax.mul(x, c))
            return key, x, v

        key, x_key, U_key = _split(key, 3)
        _, x, v = lax_control_flow.while_loop(
            lambda kxv: lax.le(kxv[2], zero), _next_kxv, (x_key, zero, minus_one)
        )
        X = lax.mul(x, x)
        V = lax.mul(lax.mul(v, v), v)
        U = uniform(U_key, (), dtype=dtype)
        return (n_rejections + 1, key, X, V, U)

    # initial state: n_rejections=0, key, X=zero, V=one, U=two
    key, subkey = _split(key)
    init_state = (0, key, zero, one, two)
    final_state = lax_control_flow.while_loop(_cond_fn, _body_fn, init_state)
    _, _, X, V, U = final_state

    # If we terminated due to max_rejections, we use the last sample (even if not accepted)
    # If we terminated due to acceptance, we use the accepted sample
    if log_space:
        log_samples = lax.neg(exponential(subkey, (), dtype=dtype))
        log_boost = lax.select(
            boost_mask | (log_samples == 0),
            zero,
            lax.mul(log_samples, lax.div(one, alpha_orig)),
        )
        return lax.add(lax.add(lax.log(d), lax.log(V)), log_boost)
    else:
        samples = 1 - uniform(subkey, (), dtype=dtype)
        boost = lax.select(boost_mask, one, lax.pow(samples, lax.div(one, alpha_orig)))
        return lax.mul(lax.mul(d, V), boost)


# TODO modify lax_special.random_gamma_grad to accept max_rejections
def _gamma_grad(sample, a, *, max_rejections, log_space):
    samples = jnp.reshape(sample, -1)
    alphas = jnp.reshape(a, -1)
    if log_space:
        # d[log(sample)] = d[sample] / sample
        # This requires computing exp(log_sample), which may be zero due to float roundoff.
        # In this case, correct it to smallest representable float.
        samples = lax.exp(samples)
        zero = lax._const(sample, 0)
        tiny = lax.full_like(samples, dtypes.finfo(samples.dtype).tiny)
        samples = lax.select(lax.eq(samples, zero), tiny, samples)

        def gamma_grad(alpha, sample):  # pyright: ignore[reportRedeclaration]  # noqa: F811
            return (
                lax_special.random_gamma_grad(alpha, sample, dtype=sample.dtype)
                / sample
            )
    else:

        def gamma_grad(alpha, sample):
            return lax_special.random_gamma_grad(alpha, sample, dtype=sample.dtype)

    if xla_bridge.get_backend().platform == "cpu":
        grads = lax_control_flow.map(lambda args: gamma_grad(*args), (alphas, samples))
    else:
        grads = vmap(gamma_grad)(alphas, samples)
    return grads.reshape(np.shape(a))


def _gamma_impl(key, a, *, log_space, max_rejections, use_vmap=False):
    # split key to match the shape of a
    a_shape = np.shape(a)
    split_count = math.prod(a_shape[key.ndim :])
    keys = key.flatten()
    keys = vmap(_split, in_axes=(0, None))(keys, split_count)
    keys = keys.flatten()
    alphas = a.flatten()

    if use_vmap and _key_impl(key) is prng.threefry_prng_impl:
        samples = vmap(
            partial(_gamma_one, max_rejections=max_rejections, log_space=log_space)
        )(
            keys,
            alphas,
        )
    else:
        samples = lax_control_flow.map(
            lambda args: _gamma_one(
                *args, max_rejections=max_rejections, log_space=log_space
            ),
            (keys, alphas),
        )

    return jnp.reshape(samples, a_shape)


def _gamma_batching_rule(batched_args, batch_dims, *, log_space, max_rejections):
    k, a = batched_args
    bk, ba = batch_dims
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims) if i is not None)
    k = batching.bdim_at_front(k, bk, size)
    a = batching.bdim_at_front(a, ba, size)
    return random_gamma_p.bind(
        k, a, log_space=log_space, max_rejections=max_rejections
    ), 0


random_gamma_p = core.Primitive("random_gamma")
random_gamma_p.def_impl(_gamma_impl)


def _random_gamma_abstract_eval(key, a, **_):
    core.standard_vma_rule("random_gamma", key, a)
    return a


random_gamma_p.def_abstract_eval(_random_gamma_abstract_eval)

ad.defjvp2(
    random_gamma_p,
    None,
    lambda tangent, ans, key, a, **kwds: tangent * _gamma_grad(ans, a, **kwds),
)
mlir.register_lowering(
    random_gamma_p,
    mlir.lower_fun(partial(_gamma_impl, use_vmap=True), multiple_results=False),
)
mlir.register_lowering(
    random_gamma_p,
    mlir.lower_fun(partial(_gamma_impl, use_vmap=True), multiple_results=False),
    platform="cpu",
)
batching.primitive_batchers[random_gamma_p] = _gamma_batching_rule


def fast_approx_gamma(
    key: ArrayLike,
    a: RealArray,
    max_rejections: int = 2,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array:
    r"""Sample Gamma random values with given shape and float dtype.

    The values are distributed according to the probability density function:

    .. math::
       f(x;a) \propto x^{a - 1} e^{-x}

    on the domain :math:`0 \le x < \infty`, with :math:`a > 0`.

    This is the standard gamma density, with a unit scale/rate parameter.
    Dividing the sample output by the rate is equivalent to sampling from
    *gamma(a, rate)*, and multiplying the sample output by the scale is equivalent
    to sampling from *gamma(a, scale)*.

    Args:
      key: a PRNG key used as the random key.
      a: a float or array of floats broadcast-compatible with ``shape``
        representing the parameter of the distribution.
      shape: optional, a tuple of nonnegative integers specifying the result
        shape. Must be broadcast-compatible with ``a``. The default (None)
        produces a result shape equal to ``a.shape``.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
      max_rejections: the maximum number of rejections allowed.

    Returns:
      A random array with the specified dtype and with shape given by ``shape`` if
      ``shape`` is not None, or else by ``a.shape``.

    See Also:
      loggamma : sample gamma values in log-space, which can provide improved
        accuracy for small values of ``a``.
    """
    key, _ = _check_prng_key("gamma", key)
    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            f"dtype argument to `gamma` must be a float dtype, got {dtype}"
        )
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _gamma(key, a, shape=shape, dtype=dtype, max_rejections=max_rejections)


def fast_approx_loggamma(
    key: ArrayLike,
    a: RealArray,
    max_rejections: int = 2,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array:
    """Sample log-gamma random values with given shape and float dtype.

    This function is implemented such that the following will hold for a
    dtype-appropriate tolerance::

      np.testing.assert_allclose(jnp.exp(loggamma(*args)), gamma(*args), rtol=rtol)

    The benefit of log-gamma is that for samples very close to zero (which occur frequently
    when `a << 1`) sampling in log space provides better precision.

    Args:
      key: a PRNG key used as the random key.
      a: a float or array of floats broadcast-compatible with ``shape``
        representing the parameter of the distribution.
      shape: optional, a tuple of nonnegative integers specifying the result
        shape. Must be broadcast-compatible with ``a``. The default (None)
        produces a result shape equal to ``a.shape``.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).

    Returns:
      A random array with the specified dtype and with shape given by ``shape`` if
      ``shape`` is not None, or else by ``a.shape``.

    See Also:
      gamma : standard gamma sampler.
    """
    key, _ = _check_prng_key("loggamma", key)
    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            f"dtype argument to `gamma` must be a float dtype, got {dtype}"
        )
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _gamma(
        key,
        a,
        max_rejections=max_rejections,
        shape=shape,
        dtype=dtype,
        log_space=True,
    )


@partial(jit, static_argnames=("shape", "dtype", "log_space", "max_rejections"))
def _gamma(key, a, max_rejections, shape, dtype, log_space=False) -> Array:
    if shape is None:
        shape = np.shape(a)
    else:
        _check_shape("gamma", shape, np.shape(a))

    a = lax.convert_element_type(a, dtype)
    if np.shape(a) != shape:
        a = jnp.broadcast_to(a, shape)
    key, (a,) = random_insert_pvary("gamma", key, a)
    return random_gamma_p.bind(
        key, a, log_space=log_space, max_rejections=max_rejections
    )


def random_insert_pvary(name, key, *args):
    # Older JAX versions may not define config._check_vma or may not expose a
    # .value attribute. Treat missing/falsey as disabled.
    _flag = getattr(config, "_check_vma", None)
    try:
        _enabled = bool(_flag.value)  # type: ignore[attr-defined]
    except Exception:
        _enabled = bool(_flag)
    if not _enabled:
        return key, args
    if not args:
        return key, args
    key_vma = core.typeof(key).vma  # type: ignore[attr-defined]
    out = []
    for a in args:
        arg_vma = (
            aval.vma
            if isinstance(aval := core.typeof(a), core.ShapedArray)
            else frozenset()
        )
        # If key is less varying than the args, then it's an error and user should
        # pvary at their level because it has key-reuse implications. They can
        # shard the keys passed to shard_map correctly so as to avoid key-reuse
        # getting correctly varying keys. But JAX shouldn't auto-pvary the key.
        if key_vma - arg_vma:
            a = core.pvary(a, tuple(k for k in key_vma if k not in arg_vma))
        if key_vma != core.typeof(a).vma:  # type: ignore[attr-defined]
            raise TypeError(
                f"{name} requires all arguments to have matching type. Got key type:"
                f" {core.typeof(key)} vs arg type: {core.typeof(a)}. Use"
                " jax.lax.pvary(...) to make them match. If your key is less varying"
                " than arg, watch out for key-reuse problems."
            )
        out.append(a)
    return key, out
