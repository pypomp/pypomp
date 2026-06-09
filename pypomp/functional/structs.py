import jax.numpy as jnp
from typing import NamedTuple, Callable


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
