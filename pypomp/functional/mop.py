import warnings

import jax

from ..core.algorithms.contexts import MopContext
from ..core.algorithms.mop import _vmapped_mop_internal
from .structs import PompStruct


def _warn_dpop_experimental(stacklevel: int) -> None:
    """Warn that DPOP (``pop`` or ``dpop=True``) is experimental."""
    warnings.warn(
        "DPOP (pop, or dpop=True in train) is experimental and its API and "
        "behavior are subject to change.",
        category=FutureWarning,
        stacklevel=stacklevel + 1,
    )


def _mop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
    dpop: bool,
) -> jax.Array:
    """Shared core of :func:`mop` and :func:`pop`."""
    context = MopContext.from_struct(struct, J=J, alpha=alpha, dpop=dpop)
    return _vmapped_mop_internal(thetas_array, keys, context)


def mop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
) -> jax.Array:
    """MOP differentiable particle filter log-likelihood objective.

    A pure functional implementation of the Measurement Off-Parameter (MOP)
    differentiable particle filter (Tan et al. 2024 [1]_), intended for composition
    within custom JAX loops or higher-order functions.

    Unlike the standard particle filter (:func:`~pypomp.functional.pfilter`), the MOP
    objective is designed to be fully differentiable with respect to the model
    parameters using automatic differentiation.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.
    thetas_array : jax.Array
        Array of parameters of shape ``(n_reps, n_params)`` on the
        estimation scale, aligned with the canonical order of
        ``struct.param_names``.
    J : int
        Number of particles.
    alpha : float
        Alpha parameter for MOP.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.

    Returns
    -------
    jax.Array
        Negative MOP log-likelihood estimates.

    Notes
    -----
    Because :func:`mop` internally transforms parameters from the estimation
    scale to the natural scale using ``struct.par_trans.from_est`` within its
    differentiable computation graph, gradients computed via automatic
    differentiation (e.g. :func:`jax.grad`) are with respect to the estimation
    parameters.

    Callers are responsible for transforming initial parameters to the
    estimation scale before the optimization loop (e.g. via
    :meth:`~pypomp.ParTrans._transform_array` with ``direction="to_est"``) and
    transforming optimized estimates back to the natural scale
    (``direction="from_est"``) when the loop concludes.

    See Also
    --------
    pypomp.functional.pop : MOP with the process score added to the weights.
    pypomp.Pomp.train : High-level OOP training interface.
    pypomp.functional.align_params : Prepare parameter arrays.

    References
    ----------
    .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
       for Partially Observed Markov Processes using Automatic Differentiation."
       *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
    """
    return _mop(struct, thetas_array, J, alpha, keys, False)


def pop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
) -> jax.Array:
    """POP differentiable particle filter log-likelihood objective (experimental).

    The Process Off-Parameter (POP) objective is :func:`mop` with the score of
    the process log-density added to the particle weights.  It gives useful
    gradients for process models whose sample paths are not differentiable in
    the parameters, such as discrete-state models.  Its value equals
    :func:`mop`'s for the same keys; only the gradient differs.

    The model must have a state named ``_logw``, in which ``rproc``
    accumulates the log-density of its sampled transitions (see Notes).

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.
    thetas_array : jax.Array
        Array of parameters of shape ``(n_reps, n_params)`` on the
        estimation scale, aligned with the canonical order of
        ``struct.param_names``.
    J : int
        Number of particles.
    alpha : float
        Alpha parameter for MOP.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.

    Returns
    -------
    jax.Array
        Negative log-likelihood estimates.

    Notes
    -----
    ``_logw`` is reset to zero at every observation time, so it need not be
    listed in ``accumvars``.  For the gradient to be correct:

    - ``rproc`` adds to ``_logw`` the log-density of every random draw whose
      distribution depends on the parameters and that has no pathwise
      gradient, such as Poisson, binomial and multinomial counts.  A
      surrogate with the same gradient in the parameters is enough; the
      ``*_logw`` functions in :mod:`pypomp.random` compute one.
    - Those draws carry no gradient of their own.  The discrete samplers in
      :mod:`pypomp.random` already satisfy this; for other samplers, wrap the
      draw in :func:`jax.lax.stop_gradient`, or its gradient is counted twice.
    - The model's gradients are finite.  NaN gradients are not masked.

    See :ref:`dpop-rproc` for a template.  Parameters are on the estimation
    scale, as for :func:`mop`.

    See Also
    --------
    pypomp.functional.mop : The same objective without the process score.
    pypomp.Pomp.train : Pass ``dpop=True`` to train with POP.
    """
    _warn_dpop_experimental(stacklevel=2)
    return _mop(struct, thetas_array, J, alpha, keys, True)
