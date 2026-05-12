import jax
from .structs import PompStruct
from ..core.algorithms.pfilter import _vmapped_pfilter_internal2


def pfilter(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    thresh: float,
    keys: jax.Array,
    CLL: bool = False,
    ESS: bool = False,
    filter_mean: bool = False,
    prediction_mean: bool = False,
) -> dict[str, jax.Array]:
    """
    This is a pure functional implementation of the particle filter, intended for
    users who need to compose it within custom JAX loops or higher-order
    functions. For a more user-friendly (but non-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.pfilter`.

    This implementation leverages JAX to efficiently vectorize the algorithm across
    multiple parameter sets simultaneously.

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, n_params).
        J (int): Number of particles.
        thresh (float): Resampling threshold.
        keys (jax.Array): Random keys. Shape (n_reps, reps, ...).
        CLL (bool): Compute conditional log-likelihoods.
        ESS (bool): Compute effective sample size.
        filter_mean (bool): Compute filtered mean.
        prediction_mean (bool): Compute prediction mean.

    Returns:
        dict[str, jax.Array]: A dictionary containing the results of the particle filter.
        The following entries are always present:
        - `logLik`: The log-likelihood estimate.
        The following entries are present if their corresponding flags are set to True:
        - `CLL`: Conditional log-likelihoods at each time point.
        - `ESS`: Effective sample size at each time point.
        - `filter_mean`: Filtered state means at each time point.
        - `prediction_mean`: Predicted state means at each time point.
    """
    results = _vmapped_pfilter_internal2(
        thetas_array,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys,
        J,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        thresh,
        keys,
        CLL,
        ESS,
        filter_mean,
        prediction_mean,
        False,
    )
    results["logLik"] = -results.pop("neg_loglik")
    return results
