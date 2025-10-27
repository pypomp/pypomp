"""
This file contains the classes for components that define the model structure.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Literal
from .ParTrans_class import ParTrans


def _create_dict_wrapper(
    user_func: Callable,
    param_names: list[str],
    statenames: list[str],
    covar_names: list[str],
    function_type: Literal["RProc", "DMeas", "RInit", "RMeas"],
    par_trans: ParTrans,
):
    """
    Create a wrapper that converts arrays to/from dicts for user functions.

    Args:
        user_func: The user-defined function
        param_names: List of parameter names in canonical order
        statenames: List of state variable names in canonical order
        covar_names: List of covariate names in canonical order
        function_type: The type of function to wrap
        par_trans: Parameter transformation object
    """
    if function_type == "RProc":
        # RProc case: X_dict, theta_dict -> state_dict
        def wrapped(X_array, theta_array, key, covars, t, dt, should_trans):  # pyright: ignore[reportRedeclaration]
            theta_dict = {name: theta_array[i] for i, name in enumerate(param_names)}
            theta_dict_trans = (
                par_trans.from_est(theta_dict) if should_trans else theta_dict
            )
            X_dict = {name: X_array[i] for i, name in enumerate(statenames)}
            covars_dict = {name: covars[i] for i, name in enumerate(covar_names)}
            result_dict = user_func(X_dict, theta_dict_trans, key, covars_dict, t, dt)
            result_array = jnp.array(
                [result_dict[name] for name in statenames]
            ).reshape(-1)
            return result_array
    elif function_type == "DMeas":
        # DMeas case: X_dict, theta_dict -> scalar
        def wrapped(Y_array, X_array, theta_array, covars, t, should_trans):  # pyright: ignore[reportRedeclaration]
            theta_dict = {name: theta_array[i] for i, name in enumerate(param_names)}
            theta_dict_trans = (
                par_trans.from_est(theta_dict) if should_trans else theta_dict
            )
            X_dict = {name: X_array[i] for i, name in enumerate(statenames)}
            covars_dict = {name: covars[i] for i, name in enumerate(covar_names)}
            return user_func(Y_array, X_dict, theta_dict_trans, covars_dict, t)
    elif function_type == "RInit":
        # RInit case: theta_dict -> state_dict
        def wrapped(theta_array, key, covars, t0, should_trans):  # pyright: ignore[reportRedeclaration]
            theta_dict = {name: theta_array[i] for i, name in enumerate(param_names)}
            theta_dict_trans = (
                par_trans.from_est(theta_dict) if should_trans else theta_dict
            )
            covars_dict = {name: covars[i] for i, name in enumerate(covar_names)}
            result_dict = user_func(theta_dict_trans, key, covars_dict, t0)
            result_array = jnp.array(
                [result_dict[name] for name in statenames]
            ).reshape(-1)
            return result_array
    elif function_type == "RMeas":
        # RMeas case: X_dict, theta_dict -> observation_array
        def wrapped(X_array, theta_array, key, covars, t, should_trans):
            theta_dict = {name: theta_array[i] for i, name in enumerate(param_names)}
            theta_dict_trans = (
                par_trans.from_est(theta_dict) if should_trans else theta_dict
            )
            X_dict = {name: X_array[i] for i, name in enumerate(statenames)}
            covars_dict = {name: covars[i] for i, name in enumerate(covar_names)}
            return user_func(X_dict, theta_dict_trans, key, covars_dict, t)
    else:
        raise ValueError(f"Invalid function type: {function_type}")

    return wrapped


def _time_interp(
    rproc: Callable,  # potentially vmap'd
    nstep_fixed: int | None,
    max_steps_bound: int | None,
) -> Callable:
    vsplit = jax.vmap(
        jax.random.split, (0, None)
    )  # handle multiple keys from vmap'd rproc

    def _interp_body(
        i: int,
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, int],
        covars_extended: jax.Array,
        dt_array_extended: jax.Array,
        should_trans: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
        # keys is a (J,) array when rproc is vmap'd
        X_, theta_, keys, t, t_idx = inputs
        covars_t = covars_extended[t_idx] if covars_extended is not None else None
        dt = dt_array_extended[t_idx]
        vkeys = vsplit(keys, 2)
        X_ = rproc(X_, theta_, vkeys[:, 0], covars_t, t, dt, should_trans)
        t = t + dt
        t_idx = t_idx + 1
        return (X_, theta_, vkeys[:, 1], t, t_idx)

    def _rproc_interp(
        X_: jax.Array,
        theta_: jax.Array,
        keys: jax.Array,
        covars_extended: jax.Array,
        dt_array_extended: jax.Array,
        t: jax.Array,
        t_idx: int,
        nstep_dynamic: int,
        accumvars: tuple[int, ...] | None,
        should_trans: bool,
    ) -> tuple[jax.Array, int]:
        # Reset accumulated variables at the start of each observation interval
        if accumvars is not None:
            X_ = X_.at[:, accumvars].set(0)

        nstep = nstep_fixed if nstep_fixed is not None else nstep_dynamic
        interp_body2 = partial(
            _interp_body,
            covars_extended=covars_extended,
            dt_array_extended=dt_array_extended,
            should_trans=should_trans,
        )
        X_, theta_, keys, t, t_idx = jax.lax.fori_loop(
            lower=0,
            upper=nstep,
            body_fun=interp_body2,
            init_val=(X_, theta_, keys, t, t_idx),
        )
        return X_, t_idx

    return _rproc_interp


class RInit:
    def __init__(
        self,
        struct: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
    ):
        """
        Initializes the RInit class with the required function structure for simulating
        the initial state distribution of a POMP model.

        Args:
            struct (Callable): A function with a specific structure where the
                first four arguments must be 'theta_', 'key', 'covars', and 't0',
                in that order. The function must return a dict mapping state names to values.
            statenames (list[str]): List of state variable names in canonical order.
            param_names (list[str]): List of parameter names in canonical order.
            covar_names (list[str]): List of covariate names in canonical order.
            par_trans (ParTrans): Parameter transformation object.
        Note:
            While this function can check that the arguments of struct are in the
            correct order, it cannot check that the output is correct. The user must
            ensure that struct returns a dict with keys matching statenames.
        """
        for i, arg in enumerate(["theta_", "key", "covars", "t0"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a list of strings")
        if not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in param_names
        ):
            raise ValueError("param_names must be a list of strings")

        self.statenames = statenames
        self.param_names = param_names
        self.covar_names = covar_names

        # Create wrapped function that converts arrays to/from dicts
        wrapped_struct = _create_dict_wrapper(
            struct, param_names, statenames, covar_names, "RInit", par_trans
        )

        self.struct = wrapped_struct
        self.struct_pf = jax.vmap(wrapped_struct, (None, 0, None, None, None))
        self.struct_per = jax.vmap(wrapped_struct, (0, 0, None, None, None))
        self.original_func = struct

    def __eq__(self, other):
        """Check equality based on meaningful attributes."""
        if not isinstance(other, RInit):
            return False
        return (
            self.statenames == other.statenames
            and self.param_names == other.param_names
            and self.original_func == other.original_func
        )


class RProc:
    def __init__(
        self,
        struct: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
        nstep: int | None = None,
        dt: float | None = None,
        accumvars: tuple[int, ...] | None = None,
    ):
        """
        Initializes the RProc class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a dict with keys matching statenames.

        Args:
            struct (callable): A function with a specific structure where the
                first six arguments must be 'X_', 'theta_', 'key', 'covars', 't', and
                'dt', in that order. The function must return a dict mapping state names to values.
            statenames (list[str]): List of state variable names in canonical order.
            param_names (list[str]): List of parameter names in canonical order.
            covar_names (list[str]): List of covariate names in canonical order.
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

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a list of strings")
        if not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in param_names
        ):
            raise ValueError("param_names must be a list of strings")

        if dt is not None and nstep is not None:
            raise ValueError("Only nstep or dt can be provided, not both")

        self.statenames = statenames
        self.param_names = param_names
        self.covar_names = covar_names

        # Create wrapped function that converts arrays to/from dicts
        wrapped_struct = _create_dict_wrapper(
            struct, param_names, statenames, covar_names, "RProc", par_trans
        )

        self.struct = wrapped_struct
        self.struct_pf = jax.vmap(wrapped_struct, (0, None, 0, None, None, None, None))
        self.struct_per = jax.vmap(wrapped_struct, (0, 0, 0, None, None, None, None))

        self.struct_interp = _time_interp(
            wrapped_struct,
            nstep_fixed=nstep,
            max_steps_bound=None,
        )
        self.struct_pf_interp = _time_interp(
            jax.vmap(wrapped_struct, (0, None, 0, None, None, None, None)),
            nstep_fixed=nstep,
            max_steps_bound=None,
        )
        self.struct_per_interp = _time_interp(
            jax.vmap(wrapped_struct, (0, 0, 0, None, None, None, None)),
            nstep_fixed=nstep,
            max_steps_bound=None,
        )
        self.nstep = int(nstep) if nstep is not None else None
        self.dt = float(dt) if dt is not None else None
        self.accumvars = accumvars
        self._max_steps_bound = None
        self.original_func = struct

    def __eq__(self, other):
        """Check equality based on meaningful attributes."""
        if not isinstance(other, RProc):
            return False
        return (
            self.statenames == other.statenames
            and self.param_names == other.param_names
            and self.original_func == other.original_func
            and self.nstep == other.nstep
            and self.dt == other.dt
            and self.accumvars == other.accumvars
        )

    def rebuild_interp(
        self, nstep_array: jax.Array | None, max_steps_bound: int | None
    ) -> None:
        """
        Set the maximum number of sub-steps allowed within any observation interval, and
        use a fixed nstep if nstep_array contains only one value. Rebuilds interpolator
        functions to honor this bound and set the nstep attribute.
        """
        if nstep_array is not None and jnp.min(nstep_array) == jnp.max(nstep_array):
            self.nstep = int(jnp.min(nstep_array))

        self._max_steps_bound = (
            int(max_steps_bound) if max_steps_bound is not None else None
        )
        self.struct_interp = _time_interp(
            self.struct,
            nstep_fixed=self.nstep,
            max_steps_bound=self._max_steps_bound,
        )
        self.struct_pf_interp = _time_interp(
            self.struct_pf,
            nstep_fixed=self.nstep,
            max_steps_bound=self._max_steps_bound,
        )
        self.struct_per_interp = _time_interp(
            self.struct_per,
            nstep_fixed=self.nstep,
            max_steps_bound=self._max_steps_bound,
        )


class DMeas:
    def __init__(
        self,
        struct: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
    ):
        """
        Initializes the DMeas class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a scalar log-likelihood.

        Args:
            struct (function): A function with a specific structure where the
                first five arguments must be 'Y_', 'X_', 'theta_', 'covars', and 't',
                in that order. The function must return a scalar log-likelihood.
            statenames (list[str]): List of state variable names in canonical order.
            param_names (list[str]): List of parameter names in canonical order.
            covar_names (list[str]): List of covariate names in canonical order.
        """
        for i, arg in enumerate(["Y_", "X_", "theta_", "covars", "t"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a list of strings")
        if not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in param_names
        ):
            raise ValueError("param_names must be a list of strings")

        self.statenames = statenames
        self.param_names = param_names
        self.covar_names = covar_names
        # Create wrapped function that converts arrays to/from dicts
        wrapped_struct = _create_dict_wrapper(
            struct, param_names, statenames, covar_names, "DMeas", par_trans
        )

        self.struct = wrapped_struct
        self.struct_pf = jax.vmap(wrapped_struct, (None, 0, None, None, None, None))
        self.struct_per = jax.vmap(wrapped_struct, (None, 0, 0, None, None, None))
        self.original_func = struct

    def __eq__(self, other):
        """Check equality based on meaningful attributes."""
        if not isinstance(other, DMeas):
            return False
        return (
            self.statenames == other.statenames
            and self.param_names == other.param_names
            and self.original_func == other.original_func
        )


class RMeas:
    def __init__(
        self,
        struct: Callable,
        ydim: int,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
    ):
        """
        Initializes the RMeas class with the required function structure.
        While this function can check that the arguments of struct are in the
        correct order, it cannot check that the output is correct. In this case,
        the user must make sure that struct returns a JAX array of shape (ydim,).

        Args:
            struct (function): A function with a specific structure where the
                first five arguments must be 'X_', 'theta_', 'key', 'covars', and 't',
                in that order. The function must return a JAX array of shape (ydim,).
            ydim (int): The dimension of Y. This currently needs to be known in advance
                to run simulate().
            statenames (list[str]): List of state variable names in canonical order.
            param_names (list[str]): List of parameter names in canonical order.
            covar_names (list[str]): List of covariate names in canonical order.
        """

        for i, arg in enumerate(["X_", "theta_", "key", "covars", "t"]):
            if struct.__code__.co_varnames[i] != arg:
                raise ValueError(f"Argument {i + 1} of struct must be '{arg}'")

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a list of strings")
        if not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in param_names
        ):
            raise ValueError("param_names must be a list of strings")

        self.statenames = statenames
        self.param_names = param_names
        self.covar_names = covar_names
        # Create wrapped function that converts arrays to/from dicts
        wrapped_struct = _create_dict_wrapper(
            struct, param_names, statenames, covar_names, "RMeas", par_trans
        )

        self.struct = wrapped_struct
        self.struct_pf = jax.vmap(wrapped_struct, (0, None, 0, None, None, None))
        self.struct_per = jax.vmap(wrapped_struct, (0, 0, 0, None, None, None))
        self.ydim = ydim
        self.original_func = struct

    def __eq__(self, other):
        """Check equality based on meaningful attributes."""
        if not isinstance(other, RMeas):
            return False
        return (
            self.statenames == other.statenames
            and self.param_names == other.param_names
            and self.original_func == other.original_func
            and self.ydim == other.ydim
        )
