import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
from ..core.par_trans import ParTrans


class PompStruct(NamedTuple):
    """
    A lightweight, immutable JAX PyTree holding the static data and compiled
    simulator functions for a POMP model.

    This object contains all the plumbing necessary to evaluate the core
    JAX algorithms (like pfilter, mif) purely functionally.
    """

    ys: jnp.ndarray
    dt_array_extended: jnp.ndarray
    nstep_array: jnp.ndarray
    t0: float
    times: jnp.ndarray
    covars_extended: jnp.ndarray | None
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
        struct.par_trans,
        struct.param_names,
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
        par_trans,
        param_names,
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
        par_trans=par_trans,
        param_names=param_names,
    )


jax.tree_util.register_pytree_node(
    PompStruct, pomp_struct_flatten, pomp_struct_unflatten
)
