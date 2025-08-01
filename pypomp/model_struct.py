"""
This file contains the classes for components that define the model structure.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
from pypomp.internal_functions import (
    _interp_covars,
    _num_fixedstep_steps,
    _num_euler_steps,
)

# given the "step_type", "dt", "nstep" and "accumvars", we aim to create a rproc that evolves state over a time interval
def _time_interp(
    rproc: Callable,  # potentially vmap'd
    step_type: str,
    dt: float | None,
    nstep: int | None,
    accumvars: tuple[int, ...] | None, # Which state variable dimensions should be reset to zero at the start of each time step (e.g., accumulated log-likelihood).
) -> Callable:
    vsplit = jax.vmap(
        jax.random.split, (0, None)
    )  # handle multiple keys from vmap'd rproc
    # if keys has shape (J,), vsplit(J,2) splits each key using split(key, 2) and generates a new key array (J, 2).
    
    # choose to use euler or fixedstep
    num_step_func = None
    match step_type:
        case "fixedstep":
            num_step_func = _num_fixedstep_steps
        case "euler":
            num_step_func = _num_euler_steps
    if num_step_func is None:
        raise ValueError("step_type must be either 'fixedstep' or 'euler'")

    def _interp_helper( # one-step function for rproc() function
        i: int,
        inputs: tuple[jax.Array, jax.Array, jax.Array, float],
        ctimes: jax.Array,
        covars: jax.Array,
        dt: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, float]:
        X_, theta_, keys, t = inputs  # keys is a (J,) array when rproc is vmap'd
        covars_t = _interp_covars(t, ctimes, covars) # the interpolated covars at time t
        vkeys = vsplit(keys, 2) # split the keys to (J, 2)
        X_ = rproc(X_, theta_, vkeys[:, 0], covars_t, t, dt)
        t = t + dt
        return (X_, theta_, vkeys[:, 1], t)

    def _rproc_interp( # looping _interp_helper() to move from t1 to t2
        X_: jax.Array,
        theta_: jax.Array,
        keys: jax.Array,
        ctimes: jax.Array,
        covars: jax.Array,
        t1: float,
        t2: float,
        dt: float | None,
        nstep: int | None,
        accumvars: tuple[int, ...] | None,
        num_step_func: Callable,
    ) -> jax.Array:
        X_ = jnp.where(accumvars is not None, X_.at[:, accumvars].set(0), X_) # set X_[:, accumvars] = 0 for accumulative variables
        nstep2, dt2 = num_step_func(t1, t2, dt=dt, nstep=nstep)
        interp_helper2 = partial(_interp_helper, ctimes=ctimes, covars=covars, dt=dt2) # use _interp_helper with ctimes, covars and dt to be fixed
        X_, theta_, keys, t = jax.lax.fori_loop(
            lower=0,
            upper=nstep2,
            body_fun=interp_helper2,
            init_val=(X_, theta_, keys, t1),
        ) # fori_loop to run nstep2 times
        return X_

    return partial(
        _rproc_interp,
        dt=dt,
        nstep=nstep,
        accumvars=accumvars,
        num_step_func=num_step_func,
    )  


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
        self.struct = struct # the initializer functions
        self.struct_pf = jax.vmap(struct, (None, 0, None, None)) # rinit for _pf
        self.struct_per = jax.vmap(struct, (0, 0, None, None)) # rinit for _per
        self.original_func = struct


class RProc:
    def __init__(
        self,
        struct: Callable,
        step_type: str = "fixedstep",
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
            step_type (str, optional): Method to describe how the process evolves over
                time. Possible choices are 'fixedstep' and 'euler'.
            nstep (int, optional): The number of steps used for the fixedstep method.
                Required if step_type is 'fixedstep'. Must be None if step_type is 'euler'.
            dt (float, optional): The time step used for the time_helper method.
                Required if step_type is 'euler'. Must be None if step_type is 'fixedstep'.
            accumvars (tuple, optional): A tuple of integers specifying the indices of the state variables that are accumulated.
        """
        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t", "dt"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        if step_type == "euler" and dt is None:
            raise ValueError("dt must be specified if step_type is 'euler'")
        if step_type == "fixedstep" and nstep is None:
            raise ValueError("nstep must be specified if step_type is 'fixedstep'")

        self.struct = _time_interp(
            struct, step_type=step_type, nstep=nstep, dt=dt, accumvars=accumvars
        )
        self.struct_pf = _time_interp(
            jax.vmap(struct, (0, None, 0, None, None, None)), # struct = rproc(X_, theta_, keys, covars, t, dt), not vectorize theta
            step_type=step_type,
            nstep=nstep,
            dt=dt,
            accumvars=accumvars,
        )
        self.struct_per = _time_interp(
            jax.vmap(struct, (0, 0, 0, None, None, None)), # vectorize theta
            step_type=step_type,
            nstep=nstep,
            dt=dt,
            accumvars=accumvars,
        )
        self.nstep = int(nstep) if nstep is not None else None
        self.dt = float(dt) if dt is not None else None
        self.step_type = step_type
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
