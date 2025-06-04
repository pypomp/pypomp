"""
This file contains the classes for components that define the model structure.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional
from pypomp.internal_functions import interp_covars


# TODO make _euler more general (e.g., rename and add support for other time step schemes, add parameter transformations)
def _euler(rproc: callable, dt: float, accumvars: tuple[int, ...]) -> callable:
    def _euler_helper(
        i: int, inputs: tuple, ctimes: jax.Array, covars: jax.Array, dt: float
    ) -> tuple[jax.Array, jax.Array, jax.Array, float]:
        X_, theta_, key, t = inputs
        covars_t = interp_covars(t, ctimes, covars)
        X_ = rproc(X_, theta_, key, covars_t, t, dt)
        t = t + dt
        return (X_, theta_, key, t)

    def _num_euler_steps(t1: float, t2: float, dt: float) -> tuple[int, float]:
        tol = jnp.sqrt(jnp.finfo(float).eps)
        nstep, dt2 = jax.lax.cond(
            t1 >= t2,
            lambda t1, t2, dt, tol: (0, 0.0),
            lambda t1, t2, dt, tol: jax.lax.cond(
                t1 + dt >= t2,
                lambda t1, t2, dt, tol: (1, t2 - t1),
                lambda t1, t2, dt, tol: (
                    jnp.ceil((t2 - t1) / dt / (1 + tol)).astype(int),
                    (t2 - t1) / jnp.ceil((t2 - t1) / dt / (1 + tol)).astype(int),
                ),
                *(t1, t2, dt, tol),
            ),
            *(t1, t2, dt, tol),
        )
        return nstep, dt2

    def _rproc_euler(
        X_: jax.Array,
        theta_: jax.Array,
        key: jax.Array,
        ctimes: jax.Array,
        covars: jax.Array,
        t1: float,
        t2: float,
        dt: float,
        accumvars: tuple[int, ...],
    ) -> jax.Array:
        X_ = X_.at[:, accumvars].set(0)
        nstep, dt2 = _num_euler_steps(t1, t2, dt=dt)
        euler_helper2 = partial(_euler_helper, ctimes=ctimes, covars=covars, dt=dt2)
        X_, theta_, key, t = jax.lax.fori_loop(
            lower=0,
            upper=nstep,
            body_fun=euler_helper2,
            init_val=(X_, theta_, key, t1),
        )
        return X_

    return partial(_rproc_euler, dt=dt, accumvars=accumvars)


class RInit:
    def __init__(self, struct: callable, t0: Optional[float] = None):
        """
        Initializes the RInit class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape (dim(X),) JAX
        array.

        Args:
            struct (function): A function with a specific structure where the
                first three arguments must be 'theta_', 'key', 'covars', and 't0'.
        """
        for i, arg in enumerate(["theta_", "key", "covars", "t0"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        self.t0 = t0
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (None, 0, None, None))
        self.struct_per = jax.vmap(struct, (0, 0, None, None))

        # @property
        # def t0(self):
        #     return self.t0

        # @t0.setter
        # def t0(self, t0):
        #     raise AttributeError("t0 cannot be set.")


class RProc:
    def __init__(
        self,
        struct: callable,
        time_helper: Optional[str] = None,
        dt: Optional[float] = None,
        accumvars: Optional[tuple[int, ...]] = None,
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
            time_helper (str, optional): Method to describe how the process evolves over time. Currently only 'euler' is supported. Defaults to None.
            dt (float, optional): The time step used for the time_helper method.
                Required if time_helper is 'euler'.
        """
        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t", "dt"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        if time_helper == "euler":
            self.struct = _euler(struct, dt=dt, accumvars=accumvars)
            self.struct_pf = _euler(
                jax.vmap(struct, (0, None, 0, None, None, None)),
                dt=dt,
                accumvars=accumvars,
            )
            self.struct_per = _euler(
                jax.vmap(struct, (0, 0, 0, None, None, None)),
                dt=dt,
                accumvars=accumvars,
            )
        else:
            print("time_helper must be 'euler' (CHANGE THIS TO AN ERROR LATER)")


class DMeas:
    def __init__(self, struct: callable):
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


class RMeas:
    def __init__(self, struct: callable, ydim: int):
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
