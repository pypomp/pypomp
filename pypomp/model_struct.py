"""
This file contains the classes for components that define the model structure.
"""

import jax


def euler(rproc, dt):
    def euler_helper(i, inputs):
        X_, theta_, key, covars, t = inputs
        X_ = rproc(X_, theta_, key, covars, t)
        t = t + dt
        return (X_, theta_, key, covars, t)

    def rproc_euler(X_, theta_, key, covars, t):
        X_, theta_, key, covars, t = jax.lax.fori_loop(
            lower=0,
            upper=int(1 / dt),  # TODO check this is correct
            body_fun=euler_helper,
            init_val=(X_, theta_, key, covars, t),
        )
        return X_, t

    return rproc_euler


class RInit:
    def __init__(self, struct):
        """
        Initializes the RInit class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape (dim(X),) JAX
        array.

        Args:
            struct (function): A function with a specific structure where the
                first three arguments must be 'theta_', 'key', and 'covars'.
        """
        for i, arg in enumerate(["theta_", "key", "covars"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (None, 0, None))
        self.struct_per = jax.vmap(struct, (0, 0, None))


class RProc:
    def __init__(self, struct, time_helper=None, dt=None):
        """
        Initializes the RProc class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape (dim(X),) JAX array.

        Args:
            struct (function): A function with a specific structure where the
                first four arguments must be 'X_', 'theta_', 'key', 'covars', and 't',
                in that order.
        """
        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        if time_helper == "euler":
            struct = euler(struct, dt=dt)

        self.struct = struct
        self.struct_pf = jax.vmap(struct, (0, None, 0, None))
        self.struct_per = jax.vmap(struct, (0, 0, 0, None))


class DMeas:
    def __init__(self, struct):
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
        self.struct_pf = jax.vmap(struct, (None, 0, None, None))
        self.struct_per = jax.vmap(struct, (None, 0, 0, None))


class RMeas:
    def __init__(self, struct):
        """
        Initializes the RMeas class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape () JAX array.

        Args:
            struct (function): A function with a specific structure where the
                first four arguments must be 'X_', 'theta_', 'key', 'covars', and 't',
                in that order.
        """

        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (0, None, 0, None))
        self.struct_per = jax.vmap(struct, (0, 0, 0, None))
