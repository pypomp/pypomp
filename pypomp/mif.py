from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _geometric_cooling
from .parameter_trans import ParTrans, _pt_forward, _pt_inverse
<<<<<<< Updated upstream
=======

_IDENTITY_PARTRANS = ParTrans(False, (), (), (), None, None)
>>>>>>> Stashed changes


def _mif_internal(
    theta: jax.Array,
    dt_array_extended: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
<<<<<<< Updated upstream
    partrans: ParTrans,
    M: int,
    a: float,
    J: int,
    thresh: float,
    key: jax.Array,
=======
    partrans: ParTrans = _IDENTITY_PARTRANS,
    M: int = 1,
    a: float = 1.0,
    J: int = 1,
    thresh: float = 0.0,
    key: jax.Array | None = None,
>>>>>>> Stashed changes
) -> tuple[jax.Array, jax.Array]:
    logliks = jnp.zeros(M)
    params = jnp.zeros((M, J, theta.shape[-1]))
    params = jnp.concatenate([theta.reshape((1, J, theta.shape[-1])), params], axis=0)

    _perfilter_internal_2 = partial(
        _perfilter_internal,
        dt_array_extended=dt_array_extended,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        rinitializers=rinitializers,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        accumvars=accumvars,
        covars_extended=covars_extended,
        partrans=partrans,
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


_jit_mif_internal = jit(_mif_internal, static_argnums=(6, 7, 8, 13, 14, 16))

_vmapped_mif_internal = jax.vmap(
    _mif_internal,
    in_axes=(1,) + (None,) * 17 + (0,),
)

_jv_mif_internal = jit(_vmapped_mif_internal, static_argnums=(6, 7, 8, 13, 14, 16))


def _perfilter_internal(
    m: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    dt_array_extended: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    partrans: ParTrans,
    thresh: float,
    a: float,
    partrans: ParTrans = _IDENTITY_PARTRANS,
):
    (params, logliks, key) = inputs
    thetas = _pt_forward(params[m], partrans)
    loglik = 0.0

    # Initial perturbation: same as the old version, cooling by observation-step count
    key, subkey = jax.random.split(key)
    sigmas_init_cooled = (
        _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
    )
    thetas = thetas + sigmas_init_cooled * jax.random.normal(
        shape=thetas.shape, key=subkey
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    thetas_nat0 = _pt_inverse(thetas, partrans)
    particlesF = rinitializers(thetas_nat0, keys, covars0, t0)

    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)

    # Total number of observation steps (same as the old version)
    n_obs = len(times)

    perfilter_helper_2 = partial(
        _perfilter_helper,
        dt_array_extended=dt_array_extended,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        sigmas=sigmas,
        accumvars=accumvars,
        covars_extended=covars_extended,
        partrans=partrans,
        thresh=thresh,
        m=m,
        a=a,
<<<<<<< Updated upstream
        n_obs=n_obs,  # >>> FIX: pass the observation-step count to helper for cooling
=======
        partrans=partrans,
>>>>>>> Stashed changes
    )

    # >>> FIX: carry obs_count and prev_obs in loop state
    init_state = (t0, particlesF, thetas, loglik, norm_weights, counts, key,
                  jnp.int32(0), jnp.bool_(True))  # obs_count=0, prev_obs=True

    (t, particlesF, thetas, loglik, norm_weights, counts, key,
     obs_count, prev_obs) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys_extended),
        body_fun=perfilter_helper_2,
        init_val=init_state,
    )

    logliks = logliks.at[m + 1].set(-loglik)
    thetas_nat_end = _pt_inverse(thetas, partrans)
    params = params.at[m + 1].set(thetas_nat_end)
    return params, logliks, key


def _perfilter_helper(
    i: int,
    inputs: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
        jax.Array, jax.Array
    ],
    dt_array_extended: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    partrans: ParTrans,
    thresh: float,
    m: int,
    a: float,
<<<<<<< Updated upstream
    n_obs: int,   # number of observation steps
) -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
        jax.Array, jax.Array
    ]:
    (t, particlesF, thetas, loglik, norm_weights, counts, key,
     obs_count, prev_obs) = inputs
=======
    partrans: ParTrans = _IDENTITY_PARTRANS,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    int,
]:
    (t, particlesF, thetas, loglik, norm_weights, counts, key, obs_idx) = inputs
>>>>>>> Stashed changes
    J = len(particlesF)

    # >>> FIX 1: perturb only at the first extended step of each observation interval; cooling is counted by observation steps
    is_obs_start = prev_obs  # the previous step is an observation point ⇒ current step is the first extended step of a new interval
    key, subkey = jax.random.split(key)
    def _perturb(thetas):
        sigmas_cooled = _geometric_cooling(nt=obs_count + 1, m=m, ntimes=n_obs, a=a) * sigmas
        noise = jax.random.normal(shape=thetas.shape, key=subkey)
        return thetas + sigmas_cooled * noise
    thetas = jax.lax.cond(is_obs_start, _perturb, lambda th: th, thetas)

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars_t = None if covars_extended is None else covars_extended[i]
    thetas_nat = _pt_inverse(thetas, partrans)
<<<<<<< Updated upstream

    # Propagate one step (Euler substep)
=======
>>>>>>> Stashed changes
    particlesP = rprocesses(
        particlesF, thetas_nat, keys, covars_t, t, dt_array_extended[i]
    )
    t = t + dt_array_extended[i]

<<<<<<< Updated upstream
    # At the end of the step: if this step is an observation point, do measurement, normalization, and (optional) resampling
    def _with_observation(body_inputs):
        loglik, norm_weights, counts, thetas, key = body_inputs
        covars_tt = None if covars_extended is None else covars_extended[i + 1]
        measurements = jnp.nan_to_num(
            dmeasures(ys_extended[i], particlesP, thetas_nat, covars_tt, t).squeeze(),
            nan=jnp.log(1e-18),
        )
=======
    def _with_observation(
        loglik, norm_weights, counts, thetas, key, obs_idx, dmeasures
    ):
        covars_t = None if covars_extended is None else covars_extended[i + 1]
        thetas_nat = _pt_inverse(thetas, partrans)
        measurements = jnp.nan_to_num(
            dmeasures(ys_extended[i], particlesP, thetas_nat, covars_t, t).squeeze(),
            nan=jnp.log(1e-18),
        )

>>>>>>> Stashed changes
        if len(measurements.shape) > 1:
            measurements = measurements.sum(axis=-1)

        weights = norm_weights + measurements
        norm_weights2, loglik_t = _normalize_weights(weights)
        loglik2 = loglik + loglik_t

        oddr = jnp.exp(jnp.max(norm_weights2)) / jnp.exp(jnp.min(norm_weights2))
        key2, subkey2 = jax.random.split(key)
        counts2, particlesF2, norm_weights3, thetas2 = jax.lax.cond(
            oddr > thresh,
            _resampler_thetas,
            _no_resampler_thetas,
            *(counts, particlesP, norm_weights2, thetas, subkey2),
        )

        # >>> FIX 2: safely zero-out accumvars (no jnp.where)
        if accumvars is not None:
            particlesF2 = particlesF2.at[:, accumvars].set(0.0)

        return (particlesF2, loglik2, norm_weights3, counts2, thetas2, key2)

    def _without_observation(body_inputs):
        loglik, norm_weights, counts, thetas, key = body_inputs
        return (particlesP, loglik, norm_weights, counts, thetas, key)

    particles, loglik, norm_weights, counts, thetas, key = jax.lax.cond(
        ys_observed[i],
        _with_observation,
        _without_observation,
        operand=(loglik, norm_weights, counts, thetas, key),
    )

<<<<<<< Updated upstream
    # Update the 'is start of a new interval' flag and the observation counter
    prev_obs = ys_observed[i]
    obs_count = obs_count + jnp.where(is_obs_start, jnp.int32(1), jnp.int32(0))

    return (t, particles, thetas, loglik, norm_weights, counts, key, obs_count, prev_obs)
=======
    return (t, particles, thetas, loglik, norm_weights, counts, key, obs_idx)
>>>>>>> Stashed changes
