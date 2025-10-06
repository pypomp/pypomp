"""
This file contains the classes for components that define the model structure.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable


def _time_interp(
    rproc: Callable,  # potentially vmap'd
    nstep_fixed: int | None,
    dt_fixed: float | None,
) -> Callable:
    vsplit = jax.vmap(
        jax.random.split, (0, None)
    )  # handle multiple keys from vmap'd rproc

    def _interp_helper(
        i: int,
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, int],
        covars_extended: jax.Array,
        dt_array_extended: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
        X_, theta_, keys, t, t_idx = inputs  # keys is a (J,) array when rproc is vmap'd
        covars_t = covars_extended[t_idx] if covars_extended is not None else None
        dt = dt_fixed if dt_fixed is not None else dt_array_extended[t_idx]
        vkeys = vsplit(keys, 2)
        X_ = rproc(X_, theta_, vkeys[:, 0], covars_t, t, dt)
        t = t + dt
        t_idx = t_idx + 1
        return (X_, theta_, vkeys[:, 1], t, t_idx)

    def _rproc_interp(
        X_: jax.Array,
        theta_: jax.Array,
        keys: jax.Array,
        covars_extended: jax.Array,
        dt_array_extended: jax.Array,
        t: float,
        t_idx: int,
        nstep_dynamic: int,
        accumvars: tuple[int, ...] | None,
    ) -> tuple[jax.Array, int]:
        X_ = jnp.where(accumvars is not None, X_.at[:, accumvars].set(0), X_)

        nstep = nstep_fixed if nstep_fixed is not None else nstep_dynamic
        interp_helper2 = partial(
            _interp_helper,
            covars_extended=covars_extended,
            dt_array_extended=dt_array_extended,
        )
        X_, theta_, keys, t, t_idx = jax.lax.fori_loop(
            lower=0,
            upper=nstep,
            body_fun=interp_helper2,
            init_val=(X_, theta_, keys, t, t_idx),
        )
        return X_, t_idx

    return _rproc_interp


class RInit:
    def __init__(self, struct: Callable, t0: float):
        """
        Initializes the RInit class with the required function structure for simulating
        the initial state distribution of a POMP model.

        Args:
            struct (Callable): A function with a specific structure where the
                first four arguments must be 'theta_', 'key', 'covars', and 't0',
                in that order. The function must return a JAX array of shape (dim(X),)
                where dim(X) is the dimension of the state vector.
            t0 (float): The initial time point for the simulation.

        Note:
            While this function can check that the arguments of struct are in the
            correct order, it cannot check that the output is correct. The user must
            ensure that struct returns a JAX array of the correct shape.
        """
        for i, arg in enumerate(["theta_", "key", "covars", "t0"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        self.t0 = float(t0)
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (None, 0, None, None))
        self.struct_per = jax.vmap(struct, (0, 0, None, None))
        self.original_func = struct


class RProc:
    def __init__(
        self,
        struct: Callable,
        nstep: int | None = None,
        dt: float | None = None,
        accumvars: tuple[int, ...] | None = None,
    ):
        """
        Initializes the RProc class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape (dim(X),) JAX array.

        Args:
            struct (callable): A function with a specific structure where the
                first six arguments must be 'X_', 'theta_', 'key', 'covars', 't', and
                'dt', in that order.
            nstep (int, optional): The number of steps used for the fixedstep method.
                Must be None if dt is provided.
            dt (float, optional): The time step used for the time_helper method.
                Must be None if nstep is provided.
            accumvars (tuple, optional): A tuple of integers specifying the indices of
                the state variables that are accumulated. These will be set to 0 at the
                beginning of each observation interval.
        """
        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t", "dt"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        if dt is not None and nstep is not None:
            raise ValueError("Only nstep or dt can be provided, not both")

        self.struct = struct
        self.struct_pf = jax.vmap(struct, (0, None, 0, None, None, None))
        self.struct_per = jax.vmap(struct, (0, 0, 0, None, None, None))

        self.struct_interp = _time_interp(
            struct,
            nstep_fixed=nstep,
            dt_fixed=dt,
        )
        self.struct_pf_interp = _time_interp(
            jax.vmap(struct, (0, None, 0, None, None, None)),
            nstep_fixed=nstep,
            dt_fixed=dt,
        )
        self.struct_per_interp = _time_interp(
            jax.vmap(struct, (0, 0, 0, None, None, None)),
            nstep_fixed=nstep,
            dt_fixed=dt,
        )
        self.nstep = int(nstep) if nstep is not None else None
        self.dt = float(dt) if dt is not None else None
        self.accumvars = accumvars
        self.original_func = struct


class DMeas:
    def __init__(self, struct: Callable):
        """
        Initializes the DMeas class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape () JAX array.

        Args:
            struct (function): A function with a specific structure where the
                first four arguments must be 'Y_', 'X_', 'theta_', 'covars', and 't',
                in that order.
        """
        for i, arg in enumerate(["Y_", "X_", "theta_", "covars", "t"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (None, 0, None, None, None))
        self.struct_per = jax.vmap(struct, (None, 0, 0, None, None))
        self.original_func = struct


class RMeas:
    def __init__(self, struct: Callable, ydim: int):
        """
        Initializes the RMeas class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape () JAX array.

        Args:
            struct (function): A function with a specific structure where the
                first four arguments must be 'X_', 'theta_', 'key', 'covars', and 't',
                in that order.
            ydim (int): The dimension of Y. This currently needs to be known in advance
                to run simulate().
        """

        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (0, None, 0, None, None))
        self.struct_per = jax.vmap(struct, (0, 0, 0, None, None))
        self.ydim = ydim
        self.original_func = struct
