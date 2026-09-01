from collections.abc import Callable
from typing import NamedTuple

import jax

from ..core.par_trans import ParTrans


class PompStruct(NamedTuple):
    """Lightweight immutable JAX PyTree containing a compiled POMP model.

    Packs the static data arrays and pre-compiled simulator callables for
    a POMP model into a single NamedTuple that can be passed through JAX
    JIT/vmap/grad boundaries.  Obtain an instance from an existing
    :class:`~pypomp.Pomp` object via :meth:`~pypomp.Pomp.to_struct`.

    Attributes
    ----------
    ys : jax.Array
        Observation array of shape ``(n_times, n_obs)``.
    dt_array_extended : jax.Array
        Integration step sizes, extended to include the step from ``t0`` to ``t1``.
    nstep_array : jax.Array
        Number of integration steps per observation interval.
    t0 : float
        Initial time.
    times : jax.Array
        Observation times of shape ``(n_times,)``.
    covars_extended : jax.Array or None
        Covariate array interpolated onto the integration grid, or
        ``None`` if no covariates are used.
    accumvars : tuple of int or None
        Indices of accumulator state variables, or ``None``.
    rinit_pf : callable
        Compiled initial state simulator for the particle filter.
    rproc_pf : callable
        Compiled state transition simulator for the particle filter.
    dmeas_pf : callable or None
        Compiled measurement log-density for the particle filter.
    rinit_per : callable
        Compiled initial state simulator for the IF2 perturbation loop.
    rproc_per : callable
        Compiled state transition simulator for the IF2 perturbation loop.
    dmeas_per : callable or None
        Compiled measurement log-density for the IF2 perturbation loop.
    rmeas_pf : callable or None
        Compiled measurement simulator for :func:`simulate`.
    dprior_pf : callable or None
        Compiled prior log-density simulator.
    par_trans : ParTrans
        Parameter transformation object.
    param_names : list of str
        Canonical parameter name ordering.
    y_names : list of str
        Observation column names, in the order matching the columns of
        ``ys`` (i.e. ``ys[:, i]`` corresponds to ``y_names[i]``).

    See Also
    --------
    pypomp.Pomp.to_struct : Construct a PompStruct from a Pomp model.
    """

    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_extended: jax.Array | None
    accumvars: tuple[int, ...] | None
    rinit_pf: Callable
    rproc_pf: Callable
    dmeas_pf: Callable | None
    rinit_per: Callable
    rproc_per: Callable
    dmeas_per: Callable | None
    rmeas_pf: Callable | None
    dprior_pf: Callable | None
    par_trans: ParTrans
    param_names: list[str]
    y_names: list[str]


def pomp_struct_flatten(struct: PompStruct):
    # Dynamic children (JAX arrays)
    children = (
        struct.ys,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.times,
        struct.covars_extended,
    )
    # Static auxiliary data (non-arrays)
    aux_data = (
        struct.t0,
        struct.accumvars,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.rinit_per,
        struct.rproc_per,
        struct.dmeas_per,
        struct.rmeas_pf,
        struct.dprior_pf,
        struct.par_trans,
        struct.param_names,
        struct.y_names,
    )
    return children, aux_data


def pomp_struct_unflatten(aux_data, children):
    ys, dt_array_extended, nstep_array, times, covars_extended = children
    (
        t0,
        accumvars,
        rinit_pf,
        rproc_pf,
        dmeas_pf,
        rinit_per,
        rproc_per,
        dmeas_per,
        rmeas_pf,
        dprior_pf,
        par_trans,
        param_names,
        y_names,
    ) = aux_data
    return PompStruct(
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        covars_extended=covars_extended,
        accumvars=accumvars,
        rinit_pf=rinit_pf,
        rproc_pf=rproc_pf,
        dmeas_pf=dmeas_pf,
        rinit_per=rinit_per,
        rproc_per=rproc_per,
        dmeas_per=dmeas_per,
        rmeas_pf=rmeas_pf,
        dprior_pf=dprior_pf,
        par_trans=par_trans,
        param_names=param_names,
        y_names=y_names,
    )


jax.tree_util.register_pytree_node(
    PompStruct, pomp_struct_flatten, pomp_struct_unflatten
)


class PanelPompStruct(NamedTuple):
    """
    A lightweight, immutable JAX PyTree holding the static data and compiled
    simulator functions for a PanelPOMP model.

    This object contains all the plumbing necessary to evaluate the core
    JAX algorithms for panel models (like panel_mif, panel_train) purely functionally.
    """

    ys_per_unit: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_per_unit: jax.Array | None
    accumvars: tuple[int, ...] | None
    rinit_pf: Callable
    rproc_pf: Callable
    dmeas_pf: Callable | None
    rinit_per: Callable
    rproc_per: Callable
    dmeas_per: Callable | None
    rmeas_pf: Callable | None
    par_trans: ParTrans
    param_names: list[str]
    shared_param_names: list[str]
    unit_param_names: list[str]
    unit_param_permutations: jax.Array
    unit_names: list[str]


def panel_pomp_struct_flatten(struct: PanelPompStruct):
    # Dynamic children (JAX arrays)
    children = (
        struct.ys_per_unit,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.times,
        struct.covars_per_unit,
        struct.unit_param_permutations,
    )
    # Static auxiliary data (non-arrays)
    aux_data = (
        struct.t0,
        struct.accumvars,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.rinit_per,
        struct.rproc_per,
        struct.dmeas_per,
        struct.rmeas_pf,
        struct.par_trans,
        struct.param_names,
        struct.shared_param_names,
        struct.unit_param_names,
        struct.unit_names,
    )
    return children, aux_data


def panel_pomp_struct_unflatten(aux_data, children):
    (
        ys_per_unit,
        dt_array_extended,
        nstep_array,
        times,
        covars_per_unit,
        unit_param_permutations,
    ) = children
    (
        t0,
        accumvars,
        rinit_pf,
        rproc_pf,
        dmeas_pf,
        rinit_per,
        rproc_per,
        dmeas_per,
        rmeas_pf,
        par_trans,
        param_names,
        shared_param_names,
        unit_param_names,
        unit_names,
    ) = aux_data
    return PanelPompStruct(
        ys_per_unit=ys_per_unit,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        covars_per_unit=covars_per_unit,
        accumvars=accumvars,
        rinit_pf=rinit_pf,
        rproc_pf=rproc_pf,
        dmeas_pf=dmeas_pf,
        rinit_per=rinit_per,
        rproc_per=rproc_per,
        dmeas_per=dmeas_per,
        rmeas_pf=rmeas_pf,
        par_trans=par_trans,
        param_names=param_names,
        shared_param_names=shared_param_names,
        unit_param_names=unit_param_names,
        unit_param_permutations=unit_param_permutations,
        unit_names=unit_names,
    )


jax.tree_util.register_pytree_node(
    PanelPompStruct, panel_pomp_struct_flatten, panel_pomp_struct_unflatten
)


def resolve_dprior(dprior: Callable | None, struct: PompStruct) -> Callable:
    """Resolve prior Callable at functional entry points."""
    from ..core.model_mechanics import _DPrior, _flat_dprior

    if dprior is not None:
        return _DPrior(
            mechanics=dprior,
            statenames=[],
            param_names=struct.param_names,
            covar_names=[],
            par_trans=struct.par_trans,
            validate_logic=False,
        ).mechanics

    if struct.dprior_pf is not None:
        return struct.dprior_pf

    return _DPrior(
        mechanics=_flat_dprior,
        statenames=[],
        param_names=struct.param_names,
        covar_names=[],
        par_trans=struct.par_trans,
        validate_logic=False,
    ).mechanics
