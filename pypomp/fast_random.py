import jax.numpy as jnp
from functools import partial
import math
from collections.abc import Sequence

import jax
import jax.random
from jax._src.lax import lax
from jax.random import normal, uniform, exponential
from jax._src import core
import numpy as np
from jax._src import dtypes
from jax._src.dtypes import check_and_canonicalize_user_dtype
from jax._src import prng
from jax._src import xla_bridge
from jax._src.api import jit, vmap
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import special as lax_special
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.random import _check_shape, _check_prng_key, _split, _key_impl
from jax._src import config

RealArray = ArrayLike
IntegerArray = ArrayLike
# TODO: Import or define these to match
# https://github.com/numpy/numpy/blob/main/numpy/typing/_dtype_like.py.
DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]


def normal_approx(
    key: jax.Array, n: jax.Array, p: jax.Array, shape: tuple
) -> jax.Array:
    ntimesp = n * p
    draws = jnp.round(
        jnp.sqrt(ntimesp * (1 - p)) * jax.random.normal(key, shape) + ntimesp + 1 / 2
    )
    return jnp.maximum(draws, 0)


def fast_approx_multinomial(
    key: jax.Array, n: jax.Array, p: jax.Array, shape: tuple | None = None
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


def fast_approx_poisson(
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

    zero = core.pvary(zero, tuple(core.typeof(alpha).vma))
    one = core.pvary(one, tuple(core.typeof(alpha).vma))
    minus_one = core.pvary(minus_one, tuple(core.typeof(alpha).vma))
    two = core.pvary(two, tuple(core.typeof(alpha).vma))

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
        count, _, X, V, U = state
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
        return lax.bitwise_and(cond_reject, count < max_rejections)

    def _body_fn(state):
        count, key, _, _, _ = state

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
        return (count + 1, key, X, V, U)

    # initial state: count=0, key, X=zero, V=one, U=two (arbitrary, will be replaced)
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

        def gamma_grad(alpha, sample):
            return lax_special.random_gamma_grad(alpha, sample) / sample
    else:
        gamma_grad = lax_special.random_gamma_grad
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
    max_rejections: int = 1,
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

    Returns:
      A random array with the specified dtype and with shape given by ``shape`` if
      ``shape`` is not None, or else by ``a.shape``.

    See Also:
      loggamma : sample gamma values in log-space, which can provide improved
        accuracy for small values of ``a``.
    """
    key, _ = _check_prng_key("gamma", key)
    dtype = dtypes.check_and_canonicalize_user_dtype(float if dtype is None else dtype)
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
    max_rejections: int = 1,
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
    dtype = dtypes.check_and_canonicalize_user_dtype(float if dtype is None else dtype)
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
    if not config._check_vma.value:
        return key, args
    if not args:
        return key, args
    key_vma = core.typeof(key).vma
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
        if key_vma != core.typeof(a).vma:
            raise TypeError(
                f"{name} requires all arguments to have matching type. Got key type:"
                f" {core.typeof(key)} vs arg type: {core.typeof(a)}. Use"
                " jax.lax.pvary(...) to make them match. If your key is less varying"
                " than arg, watch out for key-reuse problems."
            )
        out.append(a)
    return key, out
