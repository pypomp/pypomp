"""
pfilter_complete is an adaptation of pfilter that can return states
(i.e., values for each particle), filter mean, effective sample size.

This is designed for diagnostics and other data analysis investigations.
It is not designed for performance. Use pfilter within inference algorithms.

At some future point, pfilter_complete might be incorporated within
pfilter.

Currently, pfilter_complete is somewhat experimental, reflecting the
initial emphasis on methodology over data analysis for pypomp.
"""

from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from typing import Callable
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights
from .internal_functions import _interp_covars


@partial(jit, static_argnums=(4, 5, 6, 7))
def _pfilter_internal_complete2(
    theta: jax.Array,  # should be first for _line_search
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
   
    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializer(theta, keys, covars_t, t0)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    logliks_arr = jnp.zeros(len(ys))  # save loglik_t (conditional log-likelihood)
    particles_arr = jnp.zeros((len(ys), J, particlesF.shape[-1]))  # save states
    filter_mean_arr = jnp.zeros((len(ys), particlesF.shape[-1]))
    ess_arr = jnp.zeros(len(ys))
    traj_arr = jnp.zeros((len(ys), J))

    pfilter_helper_2 = partial(
        _pfilter_helper_complete2,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys=ys,
        theta=theta,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
    )
    particlesF, loglik, norm_weights, counts, key, logliks_arr, particles_arr, filter_mean_arr, ess_arr, traj_arr= jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=pfilter_helper_2,
        init_val=(particlesF, loglik, norm_weights, counts, key, logliks_arr, particles_arr,
                  filter_mean_arr, ess_arr, traj_arr),
    )

     # filt.t:
    b = jax.random.choice(key, jnp.arange(J), p=jnp.exp(norm_weights))
    filt_traj = jnp.zeros((len(ys), particlesF.shape[-1]))
    for t in range(len(ys) - 1, -1, -1):
        # extract the state value corresponding the b th particle at time t
        # and plug it into filt_traj
        filt_traj = filt_traj.at[t].set(particles_arr[t, b])
        # find the corresponding particle from the previous step from traj_arr
        # and update b
        b = traj_arr[t, b]
        b = jnp.int32(b)

    # return 1. negative loglikelihood, 
    # 2. mean log-likelihood, 
    # 3. conditional log-likelihood
    # 4. save states
    # 5. filter mean
    # 6. effective sample size
    # 7. filter_traj

    return (
        -loglik,
        loglik / len(ys),
        logliks_arr,
        particles_arr,
        filter_mean_arr,
        ess_arr,
        filt_traj,
    )


# Map over key
_vmapped_pfilter_internal_complete = jax.vmap(
    _pfilter_internal_complete2,
    in_axes=(None,) * 11 + (0,),
)

# Map over theta and key
_vmapped_pfilter_internal2_complete = jax.vmap(
    _pfilter_internal_complete2,
    in_axes=(0,) + (None,) * 10 + (0,),
)

def _pfilter_helper_complete2(
    i: int,
    inputs: tuple[jax.Array, float, jax.Array, jax.Array, jax.Array, 
                  jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    theta: jax.Array,
    rprocess: Callable,
    dmeasure: Callable,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, 
           jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    
    """
        Run particle filtering and store detailed outputs in self.results_history.

        This function runs the particle filter with J particles and (optionally)
        multiple theta replicates. Compared to the standard `pfilter`, this version
        saves additional filtering information including particles, filter means, ESS, 
        and filtered trajectories, etc. This is designed for diagnostics and other
        downstream data analysis investigations.

        Parameters
        ----------
        J : int
            Number of particles in the particle filter.
        key : jax.Array or None, optional
            JAX random key for reproducibility. If None, a new key is generated.
        theta : dict or list of dict, optional
            Parameter(s) of the POMP model. If None, `self.theta` is used.
            Can be a single parameter dict or a list of dicts.
        thresh : float, optional
            Threshold for resampling. Defaults to 0.
        reps : int, optional
            Number of replicates for each theta value. Defaults to 1.

        Returns
        -------
        None
            Results are stored in `self.results_history` as a dictionary with:
            - "negLogLiks": xarray.DataArray of negative log-likelihoods
            - "meanLogLiks": xarray.DataArray of mean conditional log-likelihoods
            - "condLogLiks": xarray.DataArray of full conditional log-likelihood arrays
            - "particles": xarray.DataArray of latent particle paths
            - "filter_mean": xarray.DataArray of filtered means at each time
            - "ess": xarray.DataArray of effective sample sizes
            - "filt_traj": xarray.DataArray of filtered trajectories
            - plus: theta, J, thresh, and key used.
    """
    
    (particlesF, loglik, norm_weights, counts, key, logliks_arr, particles_arr,
     filter_mean_arr, ess_arr, traj_arr) = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesP = rprocess(particlesF, theta, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t2)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik += loglik_t

    # save loglik_t (conditional log-likelihood) and particlesP (states) at time t2
    logliks_arr = logliks_arr.at[i].set(loglik_t)
    particles_arr = particles_arr.at[i].set(particlesP)

    # calculate filtering mean (filt.mean)
    # \sum (w_j * particle_j)
    filter_mean_t = (particlesP * jnp.exp(norm_weights[:, None])).sum(axis=0)
    filter_mean_arr = filter_mean_arr.at[i].set(filter_mean_t)

    # create filter.traj
    # variable "counts" records the source particle of particle i (1 - J) at
    # time t from the previous step
    traj_arr = traj_arr.at[i].set(counts)

    # calculate effective sample size (ess)
    ess_t = 1.0 / jnp.sum(jnp.exp(norm_weights) ** 2)
    ess_arr = ess_arr.at[i].set(ess_t)

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights = jax.lax.cond(
        oddr > thresh,
        _resampler,
        _no_resampler,
        *(counts, particlesP, norm_weights, subkey),
    )

    return (particlesF, loglik, norm_weights, counts, key, logliks_arr, particles_arr,
            filter_mean_arr, ess_arr, traj_arr)
