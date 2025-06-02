"""
This file contains the classes for components that define the model structure.
"""

import jax


def euler(rproc, dt):
    def euler_helper(i, inputs):
        X_, theta_, key, covars, t = inputs
        X_ = rproc(
            X_, theta_, key, covars, t
        )  # TODO consider applying vmap here so t isn't copied unnecessarily
        t = t + dt
        return (X_, theta_, key, covars, t)

    def rproc_euler(X_, theta_, key, covars, t):
        X_, theta_, key, covars, t = jax.lax.fori_loop(
            lower=0,
            upper=14,  # int(1 / dt),  # TODO FIX THIS
            body_fun=euler_helper,
            init_val=(X_, theta_, key, covars, t),
        )
        return X_, t

    return rproc_euler


class RInit:
    def __init__(self, struct, t0=None):
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

        vm_pf = jax.vmap(struct, (None, 0, None, None))
        vm_per = jax.vmap(struct, (0, 0, None, None))

        def struct_t0(theta_, key, covars):
            X_ = struct(theta_, key, covars, t0)
            return X_, t0

        def struct_pf(theta_, key, covars):
            X_ = vm_pf(theta_, key, covars, t0)
            return X_, t0

        def struct_per(theta_, key, covars):
            X_ = vm_per(theta_, key, covars, t0)
            return X_, t0

        self.t0 = t0
        self.struct = struct_t0
        self.struct_pf = struct_pf
        self.struct_per = struct_per

        # @property
        # def t0(self):
        #     return self.t0

        # @t0.setter
        # def t0(self, t0):
        #     raise AttributeError("t0 cannot be set.")


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
            self.struct = euler(struct, dt=dt)
            self.struct_pf = euler(jax.vmap(struct, (0, None, 0, None, None)), dt=dt)
            self.struct_per = euler(jax.vmap(struct, (0, 0, 0, None, None)), dt=dt)
        else:
            print("time_helper must be 'euler' (REPLACE THIS WITH AN ERROR LATER")


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
        self.struct_pf = jax.vmap(struct, (None, 0, None, None, None))
        self.struct_per = jax.vmap(struct, (None, 0, 0, None, None))


class RMeas:
    def __init__(self, struct, ydim):
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
