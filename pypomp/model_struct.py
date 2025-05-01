
class rinit:
    def __init__(self, struct):
        """
        Initializes the rinit class with the required function structure.
        While this function can check that the arguments of struct are in the 
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape (J, dim(X)) JAX 
        array.

        Args:
            struct (function): A function with a specific structure where the 
                first three arguments must be 'params', 'J', and 'covars'. 

        Raises:
            ValueError: If the first argument of the function is not 'params'.
            ValueError: If the second argument of the function is not 'J'.
            ValueError: If the third argument of the function is not 'covars'.
        """

        if struct.__code__.co_varnames[0] != "params":
            raise ValueError(
                "The first argument of struct must be 'params'"
            )
        if struct.__code__.co_varnames[1] != "J":
            raise ValueError(
                "The first argument of struct must be 'J'"
            )
        if struct.__code__.co_varnames[2] != "covars":
            raise ValueError(
                "The first argument of struct must be 'covars'"
            )
        self.struct = struct

class rproc:
    def __init__(self, struct):
        """
        Initializes the rproc class with the required function structure.
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
            raise ValueError(
                "The first argument of struct must be 'state'"
            )
        if struct.__code__.co_varnames[1] != "params":
            raise ValueError(
                "The second argument of struct must be 'params'"
            )
        if struct.__code__.co_varnames[2] != "key":
            raise ValueError(
                "The third argument of struct must be 'key'"
            )
        if struct.__code__.co_varnames[3] != "covars":
            raise ValueError(
                "The fourth argument of struct must be 'covars'"
            )
        self.struct = struct

class dmeas:
    def __init__(self, struct):
        """
        Initializes the dmeas class with the required function structure.
        While this function can check that the arguments of struct are in the 
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a shape () JAX array.

        Args:
            struct (function): A function with a specific structure where the 
                first three arguments must be 'y', 'state', and 'params'.

        Raises:
            ValueError: If the first argument of the function is not 'y'.
            ValueError: If the second argument of the function is not 'state'.
            ValueError: If the third argument of the function is not 'params'.
        """

        if struct.__code__.co_varnames[0] != "y":
            raise ValueError(
                "The first argument of struct must be 'y'"
            )
        if struct.__code__.co_varnames[1] != "state":
            raise ValueError(
                "The second argument of struct must be 'state'"
            )
        if struct.__code__.co_varnames[2] != "params":
            raise ValueError(
                "The third argument of struct must be 'params'"
            )
        self.struct = struct
