"""
This file contains the classes for components that define the model structure.
"""

import inspect
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .ParTrans_class import ParTrans
from typing import Annotated, Callable, get_origin, get_args, get_type_hints
from .types import (
    StateDict,
    ParamDict,
    CovarDict,
    TimeFloat,
    StepSizeFloat,
    RNGKey,
    ObservationDict,
    InitialTimeFloat,
)

# --- Type Inspection Utilities ---


def _get_annotation_tag(annotation) -> str | None:
    """Extract tag from Annotated[base, tag] or return None."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if len(args) >= 2:
            return args[1]
    return None


_TYPE_MAP = {
    "X_": StateDict,
    "theta_": ParamDict,
    "key": RNGKey,
    "covars": CovarDict,
    "t": TimeFloat,
    "dt": StepSizeFloat,
    "Y_": ObservationDict,
    "t0": InitialTimeFloat,
}
_TAG_TO_INTERNAL = {
    tag: internal_key
    for internal_key, type_val in _TYPE_MAP.items()
    for tag in (_get_annotation_tag(type_val),)
    if tag is not None
}


def _align_by_type(user_func: Callable, internal_order: list[str]) -> dict[str, str]:
    """Map internal parameter names to user parameter names via type annotations."""
    try:
        type_hints = get_type_hints(user_func, include_extras=True)
    except Exception:
        sig = inspect.signature(user_func)
        type_hints = {
            n: p.annotation
            for n, p in sig.parameters.items()
            if p.annotation != inspect.Parameter.empty
        }

    name_mapping = {}

    # 1. Match by Annotated Tag
    for param_name, user_type in type_hints.items():
        tag = _get_annotation_tag(user_type)
        if tag in _TAG_TO_INTERNAL:
            internal = _TAG_TO_INTERNAL[tag]
            if internal in name_mapping:
                raise ValueError(f"Multiple parameters annotated with tag {tag!r}.")
            name_mapping[internal] = param_name

    # 2. Match by Name or Underlying Type (Fallback)
    for internal in internal_order:
        if internal in name_mapping:
            continue

        # Check explicit name match first (common convention)
        sig = inspect.signature(user_func)
        if internal in sig.parameters:
            name_mapping[internal] = internal
            continue

        # Check rough type match
        target_type = _TYPE_MAP.get(internal)
        if target_type:
            # (Simple strict equality check for simplicity, expanded logic can go here)
            for pname, ptype in type_hints.items():
                if ptype == target_type and pname not in name_mapping.values():
                    name_mapping[internal] = pname
                    break

    missing = [k for k in internal_order if k not in name_mapping]
    if missing:
        raise ValueError(
            f"Could not map arguments for: {missing}. Use pypomp.types or exact names."
        )
    return name_mapping


# --- Validation Utilities ---


def _get_dummies(statenames, param_names, covar_names, y_names):
    """Generate dummy data for validation."""
    return {
        "X_": {n: 0.1 for n in statenames},
        "theta_": {n: 0.1 for n in param_names},
        "covars": {n: 0.1 for n in covar_names},
        "Y_": {n: 0.1 for n in (y_names or [])},
        "t": 0.0,
        "t0": 0.0,
        "dt": 0.1,
        "key": jax.random.key(0),
    }


def _validate_call(user_func, name_mapping, dummies, output_validator):
    """Generic validator that runs the function once."""
    kwargs = {
        user_name: dummies[internal] for internal, user_name in name_mapping.items()
    }

    try:
        result = user_func(**kwargs)
    except (AttributeError, TypeError) as e:
        raise TypeError(
            f"Error running '{user_func.__name__}': {e}.\n"
            "HINT: Check that you are treating inputs as dicts (not arrays) "
            "and that argument order/types are correct."
        ) from e

    output_validator(result)


# --- Base Component Class ---


class _ModelComponent:
    """Base class handling initialization, signature alignment, and validation."""

    # Subclasses must define these:
    internal_names: list[str]
    vmap_axes_pf: tuple
    vmap_axes_per: tuple

    def __init__(
        self,
        struct: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
        y_names: list[str] | None = None,
        validate_logic: bool = True,
    ):
        self.statenames = statenames
        self.param_names = param_names
        self.covar_names = covar_names
        self.par_trans = par_trans
        self.y_names = y_names or []
        self.original_func = struct

        # 1. Validation of list inputs
        for name, lst in [("statenames", statenames), ("param_names", param_names)]:
            if not isinstance(lst, list) or not all(isinstance(s, str) for s in lst):
                raise ValueError(f"{name} must be a list of strings")

        # 2. Align Arguments
        self.name_mapping = _align_by_type(struct, self.internal_names)

        # 3. Validate Logic (Dry Run)
        if validate_logic:
            dummies = _get_dummies(statenames, param_names, covar_names, y_names)
            _validate_call(struct, self.name_mapping, dummies, self._validate_output)

        # 4. Create Wrappers
        self.struct = self._make_wrapper(struct)
        self.struct_pf = jax.vmap(self.struct, self.vmap_axes_pf)
        self.struct_per = jax.vmap(self.struct, self.vmap_axes_per)

    def _validate_output(self, result):
        raise NotImplementedError

    def _make_wrapper(self, user_func):
        raise NotImplementedError

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return (
            self.statenames == other.statenames
            and self.param_names == other.param_names
            and self.original_func == other.original_func
        )


# --- Concrete Implementations ---


class RInit(_ModelComponent):
    """
    Defines the initialization process for the state variables at time t0.

    Args:
        struct (Callable): The user-defined initialization function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.

    User Function Structure
    -----------------------
    The `struct` function receives parameters, a PRNG key, covariates, and the initial time.
    It must return a dictionary mapping state names to their initial values.

    **Argument Binding:**
    You can define the function arguments in two ways:

    1. **By Name:** Use the exact names `theta_`, `key`, `covars`, and `t0`.
    2. **By Type:** Use the type hints from `pypomp.types` (recommended).

    **Template:**

    .. code-block:: python

        from pypomp.types import ParamDict, RNGKey, CovarDict, InitialTimeFloat

        def rinit(
            params: ParamDict,
            key: RNGKey,
            covars: CovarDict,
            t0: InitialTimeFloat
        ) -> dict:
            \"\"\"
            Returns initial state dictionary.
            \"\"\"
            # Access parameters by name
            S_0 = params['S_0']

            # Return dict with ALL state variables
            return {'S': S_0, 'I': 1.0, 'R': 0.0}
    """

    internal_names = ["theta_", "key", "covars", "t0"]
    vmap_axes_pf = (None, 0, None, None, None)
    vmap_axes_per = (0, 0, None, None, None)

    def _validate_output(self, result):
        if not isinstance(result, dict):
            raise TypeError(f"RInit must return a dict, got {type(result)}")
        missing = set(self.statenames) - set(result.keys())
        if missing:
            raise ValueError(f"RInit output missing state keys: {missing}")

    def _make_wrapper(self, user_func):
        # Capture variables in closure
        pnames, snames, cnames = self.param_names, self.statenames, self.covar_names
        mapping, trans = self.name_mapping, self.par_trans

        def wrapped(theta_arr, key, covars, t0, should_trans):
            theta_dict = {n: theta_arr[i] for i, n in enumerate(pnames)}
            if should_trans:
                theta_dict = trans.from_est(theta_dict)
            covars_dict = {n: covars[i] for i, n in enumerate(cnames)}

            res = user_func(
                **{
                    mapping["theta_"]: theta_dict,
                    mapping["key"]: key,
                    mapping["covars"]: covars_dict,
                    mapping["t0"]: t0,
                }
            )
            return jnp.array([res[n] for n in snames]).reshape(-1)

        return wrapped


class RProc(_ModelComponent):
    """
    Defines the process model (state transitions) of the system.

    Args:
        struct (Callable): The user-defined stepping function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.
        nstep (int, optional): Number of steps per observation interval.
        dt (float, optional): Fixed time step size (mutually exclusive with nstep).
        accumvars (tuple[int, ...], optional): Indices of states to zero-out at each observation.

    User Function Structure
    -----------------------
    The `struct` function performs a **single Euler step**. It receives the current state,
    parameters, PRNG key, covariates, current time, and step size.

    **Argument Binding:** You can define the function arguments in two ways:

    1. **By Name:** `X_`, `theta_`, `key`, `covars`, `t`, `dt`.
    2. **By Type:** `StateDict`, `ParamDict`, `RNGKey`, `CovarDict`, `TimeFloat`, `StepSizeFloat`.

    **Template:**

    .. code-block:: python

        import jax.random as random
        from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat, StepSizeFloat

        def rproc(
            state: StateDict,
            params: ParamDict,
            key: RNGKey,
            covars: CovarDict,
            t: TimeFloat,
            dt: StepSizeFloat
        ) -> dict:
            \"\"\"
            Returns the new state after time step `dt`.
            \"\"\"
            rate = params['beta'] * state['I']
            n_events = random.poisson(key, rate * dt)

            new_S = state['S'] - n_events
            new_I = state['I'] + n_events

            return {'S': new_S, 'I': new_I}
    """

    internal_names = ["X_", "theta_", "key", "covars", "t", "dt"]
    vmap_axes_pf = (0, None, 0, None, None, None, None)
    vmap_axes_per = (0, 0, 0, None, None, None, None)

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
        validate_logic: bool = True,
        nstep_array: np.ndarray | None = None,
        max_steps_bound: int | None = None,
    ):
        if dt is not None and nstep is not None:
            raise ValueError("Only nstep or dt can be provided, not both")

        super().__init__(
            struct,
            statenames,
            param_names,
            covar_names,
            par_trans,
            validate_logic=validate_logic,
        )

        self.nstep = int(nstep) if nstep is not None else None
        self.dt = float(dt) if dt is not None else None
        self.accumvars = accumvars
        self._max_steps_bound = None

        # Setup interpolation wrappers
        if nstep_array is not None:
            nstep_arr = np.asarray(nstep_array)
            all_nstep_same = np.min(nstep_arr) == np.max(nstep_arr)
            # If nstep is the same for all intervals (which can happen even if derived
            # from dt), use it for the interpolated functions.
            if all_nstep_same:
                self.nstep = int(np.min(nstep_arr))

        # _max_steps_bound might allow train to work if the step size is dynamic
        # but bounded. This is not currently implemented.
        self._max_steps_bound = int(max_steps_bound) if max_steps_bound else None

        # If nstep is given, interpolated functions use it in order to have a fixed
        # number of steps. This is necessary for train to work.
        self.struct_interp = _time_interp(
            self.struct, self.nstep, self._max_steps_bound
        )
        self.struct_pf_interp = _time_interp(
            self.struct_pf, self.nstep, self._max_steps_bound
        )
        self.struct_per_interp = _time_interp(
            self.struct_per, self.nstep, self._max_steps_bound
        )

    def _validate_output(self, result):
        if not isinstance(result, dict):
            raise TypeError(f"RProc must return a dict, got {type(result)}")
        missing = set(self.statenames) - set(result.keys())
        if missing:
            raise ValueError(f"RProc output missing state keys: {missing}")

    def _make_wrapper(self, user_func):
        pnames, snames, cnames = self.param_names, self.statenames, self.covar_names
        mapping, trans = self.name_mapping, self.par_trans

        def wrapped(X_arr, theta_arr, key, covars, t, dt, should_trans):
            X_dict = {n: X_arr[i] for i, n in enumerate(snames)}
            theta_dict = {n: theta_arr[i] for i, n in enumerate(pnames)}
            if should_trans:
                theta_dict = trans.from_est(theta_dict)
            covars_dict = {n: covars[i] for i, n in enumerate(cnames)}

            res = user_func(
                **{
                    mapping["X_"]: X_dict,
                    mapping["theta_"]: theta_dict,
                    mapping["key"]: key,
                    mapping["covars"]: covars_dict,
                    mapping["t"]: t,
                    mapping["dt"]: dt,
                }
            )
            return jnp.array([res[n] for n in snames]).reshape(-1)

        return wrapped

    def __eq__(self, other):
        return super().__eq__(other) and (
            self.nstep == other.nstep
            and self.dt == other.dt
            and self.accumvars == other.accumvars
        )


class DMeas(_ModelComponent):
    """
    Defines the measurement density (likelihood) model.

    Args:
        struct (Callable): The user-defined density function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.
        y_names (list[str], optional): List of observation names.

    User Function Structure
    -----------------------
    The `struct` function calculates the log-likelihood of the data given the state.
    **Output:** Must return a **scalar** (float or 0-d JAX array).

    **Argument Binding:** You can define the function arguments in two ways:

    1. **By Name:** `Y_`, `X_`, `theta_`, `covars`, `t`.
    2. **By Type:** `ObservationDict`, `StateDict`, `ParamDict`, `CovarDict`, `TimeFloat`.

    **Template:**

    .. code-block:: python

        import jax.scipy.stats as stats
        from pypomp.types import ObservationDict, StateDict, ParamDict, CovarDict, TimeFloat

        def dmeas(
            data: ObservationDict,
            state: StateDict,
            params: ParamDict,
            covars: CovarDict,
            t: TimeFloat
        ) -> float:
            \"\"\"
            Returns scalar log-likelihood.
            \"\"\"
            # Expected cases based on state
            mu = state['I'] * params['rho']

            # Log-likelihood of observed data
            lik = stats.poisson.logpmf(data['cases'], mu)

            return lik
    """

    internal_names = ["Y_", "X_", "theta_", "covars", "t"]
    vmap_axes_pf = (None, 0, None, None, None, None)
    vmap_axes_per = (None, 0, 0, None, None, None)

    def _validate_output(self, result):
        # Allow Python number OR JAX scalar (0-d array)
        is_jax_scalar = (
            hasattr(result, "shape") or hasattr(result, "__jax_array__")
        ) and jnp.ndim(result) == 0
        if not (isinstance(result, (int, float, np.number)) or is_jax_scalar):
            raise TypeError(
                f"DMeas must return a scalar (float or 0-d array). Got {type(result)} with shape {getattr(result, 'shape', 'N/A')}"
            )

    def _make_wrapper(self, user_func):
        pnames, snames, cnames, ynames = (
            self.param_names,
            self.statenames,
            self.covar_names,
            self.y_names,
        )
        mapping, trans = self.name_mapping, self.par_trans

        def wrapped(Y_arr, X_arr, theta_arr, covars, t, should_trans):
            Y_dict = {n: Y_arr[i] for i, n in enumerate(ynames)}
            X_dict = {n: X_arr[i] for i, n in enumerate(snames)}
            theta_dict = {n: theta_arr[i] for i, n in enumerate(pnames)}
            if should_trans:
                theta_dict = trans.from_est(theta_dict)
            covars_dict = {n: covars[i] for i, n in enumerate(cnames)}

            return user_func(
                **{
                    mapping["Y_"]: Y_dict,
                    mapping["X_"]: X_dict,
                    mapping["theta_"]: theta_dict,
                    mapping["covars"]: covars_dict,
                    mapping["t"]: t,
                }
            )

        return wrapped


class RMeas(_ModelComponent):
    """
    Defines the measurement simulation model (observation process).

    Args:
        struct (Callable): The user-defined simulation function.
        ydim (int): Dimension of the observation vector.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.

    User Function Structure
    -----------------------
    The `struct` function simulates a single observation vector from the current state.
    **Output:** Must return a 1D **JAX Array** (not a dictionary).

    **Argument Binding:** You can define the function arguments in two ways:

    1. **By Name:** `X_`, `theta_`, `key`, `covars`, `t`.
    2. **By Type:** `StateDict`, `ParamDict`, `RNGKey`, `CovarDict`, `TimeFloat`.

    **Template:**

    .. code-block:: python

        import jax.numpy as jnp
        import jax.random as random
        from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat

        def rmeas(
            state: StateDict,
            params: ParamDict,
            key: RNGKey,
            covars: CovarDict,
            t: TimeFloat
        ) -> jax.Array:
            \"\"\"
            Returns simulated data array of shape (ydim,).
            \"\"\"
            mu = state['I'] * params['rho']
            sim_cases = random.poisson(key, mu)

            # Return array, e.g., [cases, deaths]
            return jnp.array([sim_cases])
    """

    internal_names = ["X_", "theta_", "key", "covars", "t"]
    vmap_axes_pf = (0, None, 0, None, None, None)
    vmap_axes_per = (0, 0, 0, None, None, None)

    def __init__(self, struct, ydim, *args, **kwargs):
        self.ydim = ydim
        super().__init__(struct, *args, **kwargs)

    def _validate_output(self, result):
        if not hasattr(result, "shape"):  # Duck type check for array
            raise TypeError(f"RMeas must return a JAX array, got {type(result)}")

    def _make_wrapper(self, user_func):
        pnames, snames, cnames = self.param_names, self.statenames, self.covar_names
        mapping, trans = self.name_mapping, self.par_trans

        def wrapped(X_arr, theta_arr, key, covars, t, should_trans):
            X_dict = {n: X_arr[i] for i, n in enumerate(snames)}
            theta_dict = {n: theta_arr[i] for i, n in enumerate(pnames)}
            if should_trans:
                theta_dict = trans.from_est(theta_dict)
            covars_dict = {n: covars[i] for i, n in enumerate(cnames)}

            return user_func(
                **{
                    mapping["X_"]: X_dict,
                    mapping["theta_"]: theta_dict,
                    mapping["key"]: key,
                    mapping["covars"]: covars_dict,
                    mapping["t"]: t,
                }
            )

        return wrapped


# --- Interpolation Helper
def _time_interp(rproc, nstep_fixed, max_steps_bound):
    vsplit = jax.vmap(jax.random.split, (0, None))

    def _interp_body(
        i, inputs, theta_, covars_extended, dt_array_extended, should_trans
    ):
        X_, keys, t, t_idx = inputs
        covars_t = covars_extended[t_idx] if covars_extended is not None else None
        dt = dt_array_extended[t_idx]
        vkeys = vsplit(keys, 2)
        X_ = rproc(X_, theta_, vkeys[:, 0], covars_t, t, dt, should_trans)
        return (X_, vkeys[:, 1], t + dt, t_idx + 1)

    def _rproc_interp(
        X_,
        theta_,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep_dynamic,
        accumvars,
        should_trans,
    ):
        if accumvars is not None and len(accumvars) > 0:
            X_ = X_.at[:, accumvars].set(0)
        nstep = nstep_fixed if nstep_fixed is not None else nstep_dynamic

        final = jax.lax.fori_loop(
            0,
            nstep,
            partial(
                _interp_body,
                theta_=theta_,
                covars_extended=covars_extended,
                dt_array_extended=dt_array_extended,
                should_trans=should_trans,
            ),
            (X_, keys, t, t_idx),
        )
        return final[0], final[3]  # Return X_ and new t_idx

    return _rproc_interp
