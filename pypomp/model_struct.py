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
                first three arguments must be 'params', 'key', and 'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'params'.
            ValueError: If the second argument of the function is not 'key'.
            ValueError: If the third argument of the function is not 'covars'.
        """

        if struct.__code__.co_varnames[0] != "params":
            raise ValueError("The first argument of struct must be 'params'")
        if struct.__code__.co_varnames[1] != "key":
            raise ValueError("The second argument of struct must be 'key'")
        if struct.__code__.co_varnames[2] != "covars":
            raise ValueError("The third argument of struct must be 'covars'")
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
                first four arguments must be 'state', 'params', 'key', and
                'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'state'.
            ValueError: If the second argument of the function is not 'params'.
            ValueError: If the third argument of the function is not 'key'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """

        if struct.__code__.co_varnames[0] != "state":
            raise ValueError("The first argument of struct must be 'state'")
        if struct.__code__.co_varnames[1] != "params":
            raise ValueError("The second argument of struct must be 'params'")
        if struct.__code__.co_varnames[2] != "key":
            raise ValueError("The third argument of struct must be 'key'")
        if struct.__code__.co_varnames[3] != "covars":
            raise ValueError("The fourth argument of struct must be 'covars'")
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
                first three arguments must be 'y', 'state', 'params', and 'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'y'.
            ValueError: If the second argument of the function is not 'state'.
            ValueError: If the third argument of the function is not 'params'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """

        if struct.__code__.co_varnames[0] != "y":
            raise ValueError("The first argument of struct must be 'y'")
        if struct.__code__.co_varnames[1] != "state":
            raise ValueError("The second argument of struct must be 'state'")
        if struct.__code__.co_varnames[2] != "params":
            raise ValueError("The third argument of struct must be 'params'")
        if struct.__code__.co_varnames[3] != "covars":
            raise ValueError("The fourth argument of struct must be 'covars'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (None, 0, None))
        self.struct_per = jax.vmap(struct, (None, 0, 0))
