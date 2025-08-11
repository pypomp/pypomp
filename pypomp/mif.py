from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable, NamedTuple, Optional, Tuple, Any
from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _interp_covars
from .internal_functions import _geometric_cooling


class ParTrans(NamedTuple):
    is_custom: bool
    log_idx: Tuple[int, ...]
    logit_idx: Tuple[int, ...]
    to_est: Optional[Callable] = None
    from_est: Optional[Callable] = None


def parameter_trans(log=None, logit=None, *, to_est=None, from_est=None, bounds=None):
    if bounds is not None:
        raise ValueError(
            "Built-in logit only supports (0,1). For other bounds, provide custom to_est/from_est."
        )
    if (to_est is None) ^ (from_est is None):
        raise ValueError("to_est and from_est must be provided together or both None.")
    log = tuple(log or [])
    logit = tuple(logit or [])
    mode_custom = (to_est is not None and from_est is not None)
    return {
        "mode": "custom" if mode_custom else "builtin",
        "log_names": log,
        "logit_names": logit,
        "to_est": to_est,
        "from_est": from_est,
    }


def materialize_partrans(spec, paramnames):
    names = list(paramnames)
    if spec.get("mode", "builtin") == "custom":
        return ParTrans(
            is_custom=True,
            log_idx=(),
            logit_idx=(),
            to_est=spec["to_est"],
            from_est=spec["from_est"],
        )
    log_names = tuple(spec.get("log_names", ()))
    logit_names = tuple(spec.get("logit_names", ()))

    def idxs(lst):
        return tuple(names.index(n) for n in lst) if lst else ()

    return ParTrans(
        is_custom=False,
        log_idx=idxs(log_names),
        logit_idx=idxs(logit_names),
        to_est=None,
        from_est=None,
    )


_IDENTITY_PARTRANS = ParTrans(is_custom=False, log_idx=(), logit_idx=())


def _pt_forward(theta, pt: ParTrans):
    if pt.is_custom:
        return pt.to_est(theta)
    z = theta
    if len(pt.log_idx) > 0:
        li = jnp.array(pt.log_idx, dtype=jnp.int32)
        cols = jnp.take(z, li, axis=-1)
        z = z.at[..., li].set(jnp.log(cols))
    if len(pt.logit_idx) > 0:
        qi = jnp.array(pt.logit_idx, dtype=jnp.int32)
        cols = jnp.take(z, qi, axis=-1)
        y = jnp.clip(cols, 1e-12, 1 - 1e-12)
        logits = jnp.log(y) - jnp.log1p(-y)
        z = z.at[..., qi].set(logits)
    return z


def _pt_inverse(z, pt: ParTrans):
    if pt.is_custom:
        return pt.from_est(z)
    x = z
    if len(pt.log_idx) > 0:
        li = jnp.array(pt.log_idx, dtype=jnp.int32)
        cols = jnp.take(x, li, axis=-1)
        x = x.at[..., li].set(jnp.exp(cols))
    if len(pt.logit_idx) > 0:
        qi = jnp.array(pt.logit_idx, dtype=jnp.int32)
        cols = jnp.take(x, qi, axis=-1)
        vals = jax.nn.sigmoid(cols)
        x = x.at[..., qi].set(vals)
    return x


def _mif_internal(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializers: Callable,  # static
    rprocesses: Callable,      # static
    dmeasures: Callable,       # static
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    partrans: ParTrans,        # static
    M: int,                    # static
    a: float,
    J: int,                    # static
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logliks = jnp.zeros(M)
    params = jnp.zeros((M, J, theta.shape[-1]))
    params = jnp.concatenate([theta.reshape((1, J, theta.shape[-1])), params], axis=0)

    _perfilter_internal_2 = partial(
        _perfilter_internal,
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        rinitializers=rinitializers,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        a=a,
        partrans=partrans,
    )

    (params, logliks, key) = jax.lax.fori_loop(
        lower=0,
        upper=M,
        body_fun=_perfilter_internal_2,
        init_val=(params, logliks, key),
    )
    return logliks, params


# new-signature JIT/vmap
_jit_mif_internal_v1 = jit(_mif_internal, static_argnums=(4, 5, 6, 11, 12, 14))
_vmapped_mif_internal_v1 = jax.vmap(_mif_internal, in_axes=(1,) + (None,) * 15 + (0,))
_jv_mif_internal_v1 = jit(_vmapped_mif_internal_v1, static_argnums=(4, 5, 6, 11, 12, 14))


def _perfilter_internal(
    m: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    a: float,
    partrans: ParTrans,
):
    (params, logliks, key) = inputs
    thetas_nat0 = params[m]
    thetas = _pt_forward(thetas_nat0, partrans)
    loglik = 0.0

    key, subkey = jax.random.split(key)
    sigmas_init_cooled = _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
    thetas = thetas + sigmas_init_cooled * jax.random.normal(shape=thetas.shape, key=subkey)

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializers(_pt_inverse(thetas, partrans), keys, covars_t, t0)

    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)

    perfilter_helper_2 = partial(
        _perfilter_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys=ys,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        sigmas=sigmas,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        m=m,
        a=a,
        partrans=partrans,
    )

    (particlesF, thetas, loglik, norm_weights, counts, key) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=perfilter_helper_2,
        init_val=(particlesF, thetas, loglik, norm_weights, counts, key),
    )

    logliks = logliks.at[m + 1].set(-loglik)
    params = params.at[m + 1].set(_pt_inverse(thetas, partrans))
    return params, logliks, key


def _perfilter_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    m: int,
    a: float,
    partrans: ParTrans,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    (particlesF, thetas, loglik, norm_weights, counts, key) = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    sigmas_cooled = _geometric_cooling(nt=i + 1, m=m, ntimes=len(ys), a=a) * sigmas
    key, subkey = jax.random.split(key)
    thetas = thetas + sigmas_cooled * jnp.array(jax.random.normal(shape=thetas.shape, key=subkey))

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    thetas_nat = _pt_inverse(thetas, partrans)
    particlesP = rprocesses(particlesF, thetas_nat, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = jnp.nan_to_num(
        dmeasures(ys[i], particlesP, thetas_nat, covars_t, t2).squeeze(),
        nan=jnp.log(1e-18),
    )
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik += loglik_t

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights, thetas = jax.lax.cond(
        oddr > thresh,
        _resampler_thetas,
        _no_resampler_thetas,
        *(counts, particlesP, norm_weights, thetas, subkey),
    )

    return (particlesF, thetas, loglik, norm_weights, counts, key)


# -------- legacy wrapper (no partrans) --------
def _mif_internal_v0(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializers: Callable,  # static
    rprocesses: Callable,      # static
    dmeasures: Callable,       # static
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    M: int,                    # static
    a: float,
    J: int,                    # static
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _mif_internal(
        theta, t0, times, ys,
        rinitializers, rprocesses, dmeasures,
        sigmas, sigmas_init,
        ctimes, covars,
        _IDENTITY_PARTRANS,
        M, a, J, thresh, key,
    )


_jit_mif_internal_v0 = jit(_mif_internal_v0, static_argnums=(4, 5, 6, 11, 13))
_vmapped_mif_internal_v0 = jax.vmap(_mif_internal_v0, in_axes=(1,) + (None,) * 14 + (0,))
_jv_mif_internal_v0 = jit(_vmapped_mif_internal_v0, static_argnums=(4, 5, 6, 11, 13))


# -------- helpers to reorder kwargs into positional tuples --------
def _pack_v0_kwargs(kwargs: dict) -> tuple:
    return (
        kwargs["theta"],
        kwargs["t0"],
        kwargs["times"],
        kwargs["ys"],
        kwargs["rinitializers"],
        kwargs["rprocesses"],
        kwargs["dmeasures"],
        kwargs["sigmas"],
        kwargs["sigmas_init"],
        kwargs.get("ctimes", None),
        kwargs.get("covars", None),
        kwargs["M"],
        kwargs["a"],
        kwargs["J"],
        kwargs["thresh"],
        kwargs["key"],
    )


def _pack_v1_kwargs(kwargs: dict) -> tuple:
    partrans = kwargs.get("partrans", _IDENTITY_PARTRANS)
    if not isinstance(partrans, ParTrans):
        raise TypeError(
            "When calling with kwargs, 'partrans' must be a materialized ParTrans."
        )
    return (
        kwargs["theta"],
        kwargs["t0"],
        kwargs["times"],
        kwargs["ys"],
        kwargs["rinitializers"],
        kwargs["rprocesses"],
        kwargs["dmeasures"],
        kwargs["sigmas"],
        kwargs["sigmas_init"],
        kwargs.get("ctimes", None),
        kwargs.get("covars", None),
        partrans,
        kwargs["M"],
        kwargs["a"],
        kwargs["J"],
        kwargs["thresh"],
        kwargs["key"],
    )


# -------- public dispatchers (accept positional OR keyword args) --------
def _jv_mif_internal(*args, **kwargs):
    if kwargs:
        if "partrans" in kwargs and kwargs["partrans"] is not None:
            return _jv_mif_internal_v1(*_pack_v1_kwargs(kwargs))
        else:
            return _jv_mif_internal_v0(*_pack_v0_kwargs(kwargs))
    n = len(args)
    if n == 17:
        return _jv_mif_internal_v1(*args)
    elif n == 16:
        return _jv_mif_internal_v0(*args)
    else:
        raise TypeError(
            f"_jv_mif_internal expects 16 (legacy) or 17 (new) positional args, got {n}."
        )


def _jit_mif_internal(*args, **kwargs):
    if kwargs:
        if "partrans" in kwargs and kwargs["partrans"] is not None:
            return _jit_mif_internal_v1(*_pack_v1_kwargs(kwargs))
        else:
            return _jit_mif_internal_v0(*_pack_v0_kwargs(kwargs))
    n = len(args)
    if n == 17:
        return _jit_mif_internal_v1(*args)
    elif n == 16:
        return _jit_mif_internal_v0(*args)
    else:
        raise TypeError(
            f"_jit_mif_internal expects 16 (legacy) or 17 (new) positional args, got {n}."
        )