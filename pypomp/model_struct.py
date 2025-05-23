"""
This file contains the classes for components that define the model structure.
"""

import jax


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

        Raises:
            ValueError: If the first argument of the function is not 'theta_'.
            ValueError: If the second argument of the function is not 'key'.
            ValueError: If the third argument of the function is not 'covars'.
        """
        for i, arg in enumerate(["theta_", "key", "covars"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (None, 0, None))
        self.struct_per = jax.vmap(struct, (0, 0, None))


class RProc:
    def __init__(self, struct):
        """
        Initializes the RProc class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape (dim(X),) JAX array.

        Args:
            struct (function): A function with a specific structure where the
                first four arguments must be 'X_', 'theta_', 'key', and
                'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'X_'.
            ValueError: If the second argument of the function is not 'theta_'.
            ValueError: If the third argument of the function is not 'key'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """
        for i, arg in enumerate(["X_", "theta_", "key", "covars"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
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
                first four arguments must be 'Y_', 'X_', 'theta_', and 'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'Y_'.
            ValueError: If the second argument of the function is not 'X_'.
            ValueError: If the third argument of the function is not 'theta_'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """
        for i, arg in enumerate(["Y_", "X_", "theta_", "covars"]):
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
                first four arguments must be 'X_', 'theta_', 'key', and 'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'X_'.
            ValueError: If the second argument of the function is not 'theta_'.
            ValueError: If the third argument of the function is not 'key'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """

        for i, arg in enumerate(["X_", "theta_", "key", "covars"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (0, None, 0, None))
        self.struct_per = jax.vmap(struct, (0, 0, 0, None))
