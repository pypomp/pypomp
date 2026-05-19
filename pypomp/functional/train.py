import jax
from .structs import PompStruct
from ..core.algorithms.train import _vmapped_train_internal


def train(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    optimizer: str,
    M: int,
    eta: jax.Array,
    c: float,
    max_ls_itn: int,
    thresh: float,
    scale: bool,
    ls: bool,
    alpha: float | jax.Array,
    keys: jax.Array,
    alpha_cooling: float,
    n_monitors: int,
    clip_norm: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """
    This is a pure functional implementation of the optimization algorithm, intended
    for users who need to compose it within custom JAX loops or higher-order
    functions. For a more user-friendly (but non-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.train`.

    This function performs Maximum Likelihood Estimation (MLE) by treating the particle filter
    as a differentiable computational graph. It computes gradients of the log-likelihood
    with respect to the parameters via reverse-mode automatic differentiation (using JAX),
    and updates the parameters using optimizers (e.g., Adam, SGD).

    This implementation leverages JAX to efficiently vectorize the algorithm across
    multiple initial parameter sets simultaneously.

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, n_params).
        J (int): Number of particles.
        optimizer (str): Optimizer choice.
        M (int): Number of iterations.
        eta (jax.Array): Learning rates array. Shape (M, n_params).
        c (float): Armijo condition constant.
        max_ls_itn (int): Max line search iterations.
        thresh (float): Resampling threshold.
        scale (bool): Whether to scale direction.
        ls (bool): Whether to use line search.
        alpha (float | jax.Array): Alpha parameter.
        keys (jax.Array): Random keys. Shape (n_reps, ...).
        alpha_cooling (float): Alpha cooling factor.
        n_monitors (int): Number of monitors.
        clip_norm (float | None): Gradient clipping norm.
        beta1 (float): Exponential decay rate for first moment estimates.
        beta2 (float): Exponential decay rate for second moment estimates.
        epsilon (float): Small constant for numerical stability.

    Returns:
        tuple[jax.Array, jax.Array]:
            Negative logLik history: Shape (n_reps, M)
            Theta history: Shape (n_reps, M+1, n_params)
    """
    return _vmapped_train_internal(
        thetas_array,
        struct.ys,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        J,
        optimizer,
        M,
        eta,
        c,
        max_ls_itn,
        thresh,
        scale,
        ls,
        alpha,
        keys,
        alpha_cooling,
        n_monitors,
        clip_norm,
        beta1,
        beta2,
        epsilon,
    )
