import jax.numpy as jnp

import jax
import jax.random


def normal_approx(
    key: jax.Array, n: jax.Array, p: jax.Array, shape: tuple
) -> jax.Array:
    ntimesp = n * p
    draws = jnp.round(
        jnp.sqrt(ntimesp * (1 - p)) * jax.random.normal(key, shape) + ntimesp + 1 / 2
    )
    return jnp.maximum(draws, 0)


def fast_multinomial(
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


def fast_poisson(
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
