from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from .pfilter import _pfilter_internal
from .internal_functions import _normalize_weights

# from .internal_functions import _rinits_internal
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas

MONITORS = 1  # TODO: figure out what this is for and remove it if possible


# TODO: add external mif function
def mif(
    rinit,
    rproc,
    dmeas,
    ys,
    theta,
    sigmas,
    sigmas_init,
    covars,
    M,
    a,
    J,
    thresh=0,
    monitor=False,
    verbose=False,
    key=None,
):
    """
    Perform the iterated filtering (IF2) algorithm for a partially observed
    Markov process (POMP) model to estimate model parameters by maximizing
    the likelihood.

    Args:
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        dmeas (DMeas): Density evaluation for the measurement model.
        ys (array-like): The measurement array.
        theta (array-like): Initial parameters for the POMP model.
        sigmas (float): Perturbation factor for parameters.
        sigmas_init (float): Initial perturbation factor for parameters.
        covars (array-like): Covariates or None if not applicable.
        M (int): Number of algorithm iterations.
        a (float): Decay factor for sigmas.
        J (int): Number of particles.
        thresh (float): Resampling threshold.
        monitor (bool): Flag to monitor log-likelihood values.
        verbose (bool): Flag to print log-likelihood and parameter information.
        key (jax.random.PRNGKey): Random key for reproducibility.

    Raises:
        ValueError: If J is less than 1 or any required arguments are missing.

    Returns:
        tuple: Contains:
            - Array of negative log-likelihood values through iterations.
            - Array of parameters through iterations.
    """

    if J < 1:
        raise ValueError("J should be greater than 0.")
    missing_args = [
        arg
        for arg in [
            rinit,
            rproc,
            dmeas,
            ys,
            theta,
            sigmas,
            sigmas_init,
            covars,
            M,
            a,
            J,
            key,
        ]
        if arg is None
    ]
    if len(missing_args) > 0:
        raise ValueError(
            f"The following arguments are missing: {missing_args}. Please check your arguments and try again."
        )

    return _mif_internal(
        theta=theta,
        ys=ys,
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        dmeasure=dmeas.struct_pf,
        rinitializers=rinit.struct_per,
        rprocesses=rproc.struct_per,
        dmeasures=dmeas.struct_per,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        covars=covars,
        M=M,
        a=a,
        J=J,
        thresh=thresh,
        monitor=monitor,
        verbose=verbose,
        key=key,
    )


def _mif_internal(
    theta,
    ys,
    rinitializer,
    rprocess,
    dmeasure,
    rinitializers,
    rprocesses,
    dmeasures,
    sigmas,
    sigmas_init,
    covars,
    M,
    a,
    J,
    thresh,
    monitor,
    verbose,
    key,
):
    """
    Internal function for conducting the iterated filtering (IF2) algorithm.
    This is called in the '_fit_internal' function.

    Args:
        theta (array-like): Parameters involved in the POMP model.
        ys (array-like): The measurement array.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        rprocesses (function): Simulator for the perturbed process model.
        dmeasures (function): Density evaluation for the perturbed measurement
            model.
        sigmas (float): Perturbed factor.
        sigmas_init (float): Initial perturbed factor.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        M (int, optional): Algorithm Iteration. Defaults to 10.
        a (float, optional): Decay factor for sigmas. Defaults to 0.95.
        J (int, optional): The number of particles. Defaults to 100.
        thresh (float, optional): Threshold value to determine whether to
            resample particles. Defaults to 100.
        monitor (bool, optional): Boolean flag controlling whether to monitor
            the log-likelihood value. Defaults to False.
        verbose (bool, optional): Boolean flag controlling whether to print out
            the log-likehood and parameter information. Defaults to False.

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations.
        - An array of parameters through the iterations.
    """
    logliks = []
    params = []

    ndim = theta.ndim
    thetas = jnp.tile(theta, (J,) + (1,) * ndim)
    params.append(thetas)

    if monitor:
        key, subkey = jax.random.split(key=key)
        loglik = jnp.mean(
            jnp.array(
                [
                    _pfilter_internal(
                        thetas.mean(0),
                        ys,
                        J,
                        rinitializer,
                        rprocess,
                        dmeasure,
                        covars=covars,
                        thresh=thresh,
                        key=subkey,
                    )
                    for i in range(MONITORS)
                ]
            )
        )
        logliks.append(loglik)

    for m in tqdm(range(M)):
        # TODO: Cool sigmas between time-iterations.
        key, *subkeys = jax.random.split(key=key, num=3)
        sigmas = a * sigmas
        sigmas_init = a * sigmas_init
        thetas += sigmas_init * jax.random.normal(shape=thetas.shape, key=subkeys[0])
        loglik_ext, thetas = _perfilter_internal(
            thetas,
            ys,
            J,
            sigmas,
            rinitializers,
            rprocesses,
            dmeasures,
            ndim=ndim,
            covars=covars,
            thresh=thresh,
            key=subkeys[1],
        )

        params.append(thetas)

        if monitor:
            key, subkey = jax.random.split(key=key)
            loglik = jnp.mean(
                jnp.array(
                    [
                        _pfilter_internal(
                            thetas.mean(0),
                            ys,
                            J,
                            rinitializer,
                            rprocess,
                            dmeasure,
                            covars=covars,
                            thresh=thresh,
                            key=subkey,
                        )
                        for i in range(MONITORS)
                    ]
                )
            )

            logliks.append(loglik)

            if verbose:
                print(loglik)
                print(thetas.mean(0))

    return jnp.array(logliks), jnp.array(params)


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal(
    theta,
    ys,
    J,
    sigmas,
    rinitializers,
    rprocesses,
    dmeasures,
    ndim,
    covars,
    thresh,
    key,
):
    """
    Internal functions for perturbed particle filtering algorithm, which calls
    function 'perfilter_helper' iteratively.
    """
    loglik = 0
    key, subkey = jax.random.split(key)
    thetas = theta + sigmas * jax.random.normal(
        shape=(J,) + theta.shape[-ndim:], key=subkey
    )
    key, keys = _keys_helper(key=key, J=J, covars=covars)
    # Took this if statement from _perfilter_helper, but I'm not sure why it's needed.
    if covars is not None:
        particlesF = rinitializers(thetas, keys, covars)
    else:
        particlesF = rinitializers(thetas, keys, None)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    # if key is None:
    # key = jax.random.PRNGKey(np.random.choice(int(1e18)))
    perfilter_helper_2 = partial(
        _perfilter_helper, rprocesses=rprocesses, dmeasures=dmeasures
    )
    (
        particlesF,
        thetas,
        sigmas,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
    ) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=perfilter_helper_2,
        init_val=[
            particlesF,
            thetas,
            sigmas,
            covars,
            loglik,
            norm_weights,
            counts,
            ys,
            thresh,
            key,
        ],
    )

    return -loglik, thetas


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal_mean(
    theta,
    ys,
    J,
    sigmas,
    rinitializers,
    rprocesses,
    dmeasures,
    ndim,
    covars,
    thresh,
    key,
):
    """
    Internal functions for calculating the mean result using perturbed particle
    filtering algorithm across the measurements.
    """
    value, thetas = _perfilter_internal(
        theta,
        ys,
        J,
        sigmas,
        rinitializers,
        rprocesses,
        dmeasures,
        ndim,
        covars,
        thresh,
        key,
    )
    return value / len(ys), thetas


def _perfilter_helper(t, inputs, rprocesses, dmeasures):
    """
    Helper functions for perturbed particle filtering algorithm, which conducts
    a single iteration of filtering and is called in function
    'perfilter_internal'.
    """
    (
        particlesF,
        thetas,
        sigmas,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
    ) = inputs
    J = len(particlesF)

    key, keys = _keys_helper(key=key, J=J, covars=covars)

    key, subkey = jax.random.split(key)
    thetas += sigmas * jnp.array(jax.random.normal(shape=thetas.shape, key=subkey))

    # Get prediction particles
    # r processes: particleF and thetas are both vectorized (J times)
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars)
    else:
        particlesP = rprocesses(particlesF, thetas, keys, None)

    measurements = jnp.nan_to_num(
        dmeasures(ys[t], particlesP, thetas, covars).squeeze(), nan=jnp.log(1e-18)
    )  # shape (Np,)

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
        counts,
        particlesP,
        norm_weights,
        thetas,
        subkey,
    )

    return [
        particlesF,
        thetas,
        sigmas,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
    ]
