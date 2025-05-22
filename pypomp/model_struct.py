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

        if struct.__code__.co_varnames[0] != "theta_":
            raise ValueError("The first argument of struct must be 'theta_'")
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
                first four arguments must be 'X_', 'theta_', 'key', and
                'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'X_'.
            ValueError: If the second argument of the function is not 'theta_'.
            ValueError: If the third argument of the function is not 'key'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """

        if struct.__code__.co_varnames[0] != "X_":
            raise ValueError("The first argument of struct must be 'X_'")
        if struct.__code__.co_varnames[1] != "theta_":
            raise ValueError("The second argument of struct must be 'theta_'")
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
                first four arguments must be 'Y_', 'X_', 'theta_', and 'covars'.

        Raises:
            ValueError: If the first argument of the function is not 'Y_'.
            ValueError: If the second argument of the function is not 'X_'.
            ValueError: If the third argument of the function is not 'theta_'.
            ValueError: If the fourth argument of the function is not 'covars'.
        """

        if struct.__code__.co_varnames[0] != "Y_":
            raise ValueError("The first argument of struct must be 'Y_'")
        if struct.__code__.co_varnames[1] != "X_":
            raise ValueError("The second argument of struct must be 'X_'")
        if struct.__code__.co_varnames[2] != "theta_":
            raise ValueError("The third argument of struct must be 'theta_'")
        if struct.__code__.co_varnames[3] != "covars":
            raise ValueError("The fourth argument of struct must be 'covars'")
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

        if struct.__code__.co_varnames[0] != "X_":
            raise ValueError("The first argument of struct must be 'X_'")
        if struct.__code__.co_varnames[1] != "theta_":
            raise ValueError("The second argument of struct must be 'theta_'")
        if struct.__code__.co_varnames[2] != "key":
            raise ValueError("The third argument of struct must be 'key'")
        if struct.__code__.co_varnames[3] != "covars":
            raise ValueError("The fourth argument of struct must be 'covars'")
        self.struct = struct
        self.struct_pf = jax.vmap(struct, (0, None, 0, None))
        self.struct_per = jax.vmap(struct, (0, 0, 0, None))
