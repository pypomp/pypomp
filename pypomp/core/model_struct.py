"""
This file contains the classes for components that define the model structure.
"""

import inspect
from collections.abc import Callable
from functools import partial
from typing import (
    Annotated,
    Any,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import jax
import jax.numpy as jnp
import numpy as np

from pypomp.types import (
    CovarDict,
    InitialTimeFloat,
    ObservationDict,
    ParamDict,
    RNGKey,
    StateDict,
    StepSizeFloat,
    TimeFloat,
)

from .par_trans import ParTrans

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


_DUMMY_J = 3  # Particle count used when validating a vectorized component.


def _get_dummies(
    statenames: list[str],
    param_names: list[str],
    covar_names: list[str],
    y_names: list[str] | None,
    vectorized: bool = False,
) -> dict[str, Any]:
    """Generate dummy data for validation.

    When ``vectorized`` is True the state entries are given a leading particle
    axis, matching the convention for manually vectorized components. Parameters
    and covariates stay scalar because they are shared across particles.
    """
    states: dict[str, Any] = (
        {n: jnp.full((_DUMMY_J,), 0.1) for n in statenames}
        if vectorized
        else {n: 0.1 for n in statenames}
    )
    return {
        "X_": states,
        "theta_": {n: 0.1 for n in param_names},
        "covars": {n: 0.1 for n in covar_names},
        "Y_": {n: 0.1 for n in (y_names or [])},
        "t": 0.0,
        "t0": 0.0,
        "dt": 0.1,
        "key": jax.random.key(0),
    }


def _validate_call(
    user_func: Callable,
    name_mapping: dict[str, str],
    dummies: dict[str, dict[str, float] | float | jax.Array],
    output_validator: Callable,
):
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

    # Components that support manual vectorization override this (see _RProc).
    _is_vectorized: bool = False

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
            dummies = _get_dummies(
                statenames, param_names, covar_names, y_names, self._is_vectorized
            )
            _validate_call(struct, self.name_mapping, dummies, self._validate_output)

        # 4. Create Wrappers
        self._build_structs(struct)

    def _build_structs(self, struct: Callable) -> None:
        """Create the internal callables, adding a particle axis via vmap."""
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


# --- Public Decorators ---

_F = TypeVar("_F", bound=Callable[..., Any])


def vectorized(func: _F) -> _F:
    """Mark an ``rproc`` as already vectorized over the particle axis.

    By default pypomp applies :func:`jax.vmap` to ``rproc`` so that the user
    writes the dynamics for a *single* particle. Decorating ``rproc`` with
    ``@vectorized`` disables that vmap: the function is then called once per
    Euler step with every state entry as a ``(J,)`` array, and is responsible
    for handling all ``J`` particles itself.

    This exists purely as a CPU performance escape hatch. XLA's CPU backend does
    not vectorize per-particle random number generation across the particle
    batch, so a manually vectorized ``rproc`` can be several times faster on CPU.
    On GPU the default vmapped path already maps particles onto threads, so the
    decorator is not expected to help there.

    Contract for a vectorized ``rproc``:

    - Each value in the state dict is a ``(J,)`` array; the returned dict must
      hold ``(J,)`` arrays (or scalars, which are broadcast).
    - Parameters and covariates are shared across particles and stay scalar,
      except under ``mif``, where parameters are perturbed per particle and
      arrive as ``(J,)`` arrays. Writing elementwise code handles both.
    - ``key`` is a *single* PRNG key for the whole step, not one per particle.
      Draw for all particles at once, e.g. ``jax.random.normal(key, (J,))``.
    - Infer ``J`` from the state (``X_["S"].shape[0]``) rather than hardcoding
      it, so the function still composes when pypomp maps over replicates.

    Example
    -------

    .. code-block:: python

        from pypomp import vectorized

        @vectorized
        def rproc(X_, theta_, key, covars, t, dt):
            S, I = X_["S"], X_["I"]
            J = S.shape[0]
            dw = jax.random.normal(key, (J,)) * jnp.sqrt(dt)
            infections = theta_["beta"] * S * I * dt + dw
            return {"S": S - infections, "I": I + infections}
    """
    func._pypomp_vectorized = True  # pyright: ignore[reportFunctionMemberAccess]
    return func


# --- Concrete Implementations ---


class _RInit(_ModelComponent):
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
            raise TypeError(f"rinit function must return a dict, got {type(result)}")
        missing = set(self.statenames) - set(result.keys())
        if missing:
            raise ValueError(f"rinit function output missing state keys: {missing}")

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


class _RProc(_ModelComponent):
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

        from pypomp.random import fast_poisson
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
            n_events = fast_poisson(key, rate * dt)

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
        vectorized: bool | None = None,
    ):
        if dt is not None and nstep is not None:
            raise ValueError("Only nstep or dt can be provided, not both")

        # Set before super().__init__() because validation and wrapper creation
        # both depend on it.
        if vectorized is None:
            vectorized = bool(getattr(struct, "_pypomp_vectorized", False))
        self._is_vectorized = vectorized

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
        # The vectorized path threads the state through the sub-step loop as a
        # dict of columns, so it uses the dict-based wrappers directly.
        pf_step = self._struct_pf_dict if self._is_vectorized else self.struct_pf
        per_step = self._struct_per_dict if self._is_vectorized else self.struct_per
        base_step = self._struct_pf_dict if self._is_vectorized else self.struct
        interp_args = (self.nstep, self._max_steps_bound, self._is_vectorized)
        self.struct_interp = _time_interp(base_step, *interp_args, self.statenames)
        self.struct_pf_interp = _time_interp(pf_step, *interp_args, self.statenames)
        self.struct_per_interp = _time_interp(per_step, *interp_args, self.statenames)

    def _validate_output(self, result):
        if not isinstance(result, dict):
            raise TypeError(f"rproc function must return a dict, got {type(result)}")
        missing = set(self.statenames) - set(result.keys())
        if missing:
            raise ValueError(f"rproc function output missing state keys: {missing}")

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

    def _make_wrapper_vectorized(self, user_func, theta_batched: bool):
        """Wrapper for an rproc that already handles the particle axis itself."""
        pnames, snames, cnames = self.param_names, self.statenames, self.covar_names
        mapping, trans = self.name_mapping, self.par_trans

        def wrapped(X_dict, theta_arr, key, covars, t, dt, should_trans):
            J = X_dict[snames[0]].shape[0]
            if theta_batched:
                theta_dict = {n: theta_arr[:, i] for i, n in enumerate(pnames)}
            else:
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
            # broadcast_to allows returning a scalar for a state that happens to
            # be constant across particles, and keeps the fori_loop carry shapes
            # stable from one sub-step to the next.
            return {n: jnp.broadcast_to(jnp.asarray(res[n]), (J,)) for n in snames}

        return wrapped

    @staticmethod
    def _array_adapter(dict_func, snames: list[str]):
        """Give a dict-based vectorized wrapper the standard array signature."""

        def wrapped(X_arr, theta_arr, key, covars, t, dt, should_trans):
            X_dict = {n: X_arr[:, i] for i, n in enumerate(snames)}
            res = dict_func(X_dict, theta_arr, key, covars, t, dt, should_trans)
            return jnp.stack([res[n] for n in snames], axis=-1)

        return wrapped

    def _build_structs(self, struct: Callable) -> None:
        if not self._is_vectorized:
            super()._build_structs(struct)
            return
        # The user function maps over particles itself, so no vmap is applied.
        self._struct_pf_dict = self._make_wrapper_vectorized(
            struct, theta_batched=False
        )
        self._struct_per_dict = self._make_wrapper_vectorized(
            struct, theta_batched=True
        )
        self.struct = self._array_adapter(self._struct_pf_dict, self.statenames)
        self.struct_pf = self.struct
        self.struct_per = self._array_adapter(self._struct_per_dict, self.statenames)

    def __eq__(self, other):
        return super().__eq__(other) and (
            self.nstep == other.nstep
            and self.dt == other.dt
            and self.accumvars == other.accumvars
            and self._is_vectorized == other._is_vectorized
        )


class _DMeas(_ModelComponent):
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
                f"dmeas function must return a scalar (float or 0-d array). Got {type(result)} with shape {getattr(result, 'shape', 'N/A')}"
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


class _RMeas(_ModelComponent):
    """
    Defines the measurement simulation model (observation process).

    Args:
        struct (Callable): The user-defined simulation function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.
        y_names (list[str], optional): List of observation names.

    User Function Structure
    -----------------------
    The `struct` function simulates observations from the current state.
    **Output:** Must return a **dictionary** mapping observation names to their simulated values.

    **Argument Binding:** You can define the function arguments in two ways:

    1. **By Name:** `X_`, `theta_`, `key`, `covars`, `t`.
    2. **By Type:** `StateDict`, `ParamDict`, `RNGKey`, `CovarDict`, `TimeFloat`.

    **Template:**

    .. code-block:: python

        import jax.numpy as jnp
        from pypomp.random import fast_poisson
        from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat

        def rmeas(
            state: StateDict,
            params: ParamDict,
            key: RNGKey,
            covars: CovarDict,
            t: TimeFloat
        ) -> dict:
            \"\"\"
            Returns simulated observation dictionary.
            \"\"\"
            mu = state['I'] * params['rho']
            sim_cases = fast_poisson(key, mu)

            # Return dictionary of simulated observations
            return {'cases': sim_cases}
    """

    internal_names = ["X_", "theta_", "key", "covars", "t"]
    vmap_axes_pf = (0, None, 0, None, None, None)
    vmap_axes_per = (0, 0, 0, None, None, None)

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
        self.ydim = len(y_names) if y_names is not None else 0

        super().__init__(
            struct,
            statenames,
            param_names,
            covar_names,
            par_trans,
            y_names=y_names,
            validate_logic=validate_logic,
        )

    def _validate_output(self, result):
        if not isinstance(result, dict):
            raise TypeError(f"rmeas function must return a dict, got {type(result)}")
        missing = set(self.y_names) - set(result.keys())
        if missing:
            raise ValueError(
                f"rmeas function output missing observation keys: {missing}"
            )

    def _make_wrapper(self, user_func):
        pnames, snames, cnames, ynames = (
            self.param_names,
            self.statenames,
            self.covar_names,
            self.y_names,
        )
        mapping, trans = self.name_mapping, self.par_trans

        def wrapped(X_arr, theta_arr, key, covars, t, should_trans):
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
                }
            )
            return jnp.array([res[n] for n in ynames]).reshape(-1)

        return wrapped


class _DPrior(_ModelComponent):
    """
    Defines the prior log-density model for parameter values.

    Args:
        struct (Callable): The user-defined prior function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.

    User Function Structure
    -----------------------
    The `struct` function calculates the log-prior density given parameter values.
    **Output:** Must return a **scalar** (float or 0-d JAX array).

    **Argument Binding:** You can define the function arguments in two ways:

    1. **By Name:** Use the exact parameter name `theta_`.
    2. **By Type:** Label the argument with `ParamDict` from `pypomp.types`.

    **Template:**

    .. code-block:: python

        import jax.scipy.stats as stats
        from pypomp.types import ParamDict

        def dprior(params: ParamDict) -> float:
            \"\"\"
            Returns scalar log-prior density.
            \"\"\"
            return stats.norm.logpdf(params['beta'], loc=1.0, scale=0.5)
    """

    internal_names = ["theta_"]
    vmap_axes_pf = (None, None)
    vmap_axes_per = (0, None)

    def _validate_output(self, result):
        is_jax_scalar = (
            hasattr(result, "shape") or hasattr(result, "__jax_array__")
        ) and jnp.ndim(result) == 0
        if not (isinstance(result, (int, float, np.number)) or is_jax_scalar):
            raise TypeError(
                f"dprior function must return a scalar (float or 0-d array). Got {type(result)} with shape {getattr(result, 'shape', 'N/A')}"
            )

    def _make_wrapper(self, user_func):
        pnames = self.param_names
        mapping, trans = self.name_mapping, self.par_trans

        def _from_est_array(arr):
            d_in = {n: arr[i] for i, n in enumerate(pnames)}
            d_out = trans.from_est(d_in)
            return jnp.stack([d_out[n] for n in pnames])

        def wrapped(theta_arr, should_trans=False):
            theta_dict = {n: theta_arr[i] for i, n in enumerate(pnames)}
            if should_trans and len(pnames) > 0:
                theta_dict = trans.from_est(theta_dict)
                J = jax.jacobian(_from_est_array)(theta_arr)
                log_det_J = jnp.linalg.slogdet(J)[1]
            else:
                log_det_J = 0.0

            log_p = user_func(**{mapping["theta_"]: theta_dict})
            return log_p + log_det_J

        return wrapped


def _flat_dprior(params: ParamDict) -> float:
    """Flat improper prior -- always returns 0.0."""
    return 0.0


# --- Interpolation Helper
def _time_interp(
    rproc,
    nstep_fixed,
    max_steps_bound,
    vectorized: bool = False,
    statenames: list[str] | None = None,
):
    if vectorized and not statenames:
        raise ValueError("statenames are required to build a vectorized rproc")
    snames: list[str] = list(statenames) if statenames else []
    vsplit = jax.vmap(jax.random.split, (0, None))

    def _interp_body(
        i, inputs, theta_, covars_extended, dt_array_extended, should_trans
    ):
        X_, keys, t, t_idx = inputs
        covars_t = covars_extended[t_idx] if covars_extended is not None else None
        dt = dt_array_extended[t_idx]
        if vectorized:
            # A single key per step; the rproc draws for all particles at once.
            next_key, subkey = jax.random.split(keys)
            X_ = rproc(X_, theta_, subkey, covars_t, t, dt, should_trans)
            return (X_, next_key, t + dt, t_idx + 1)
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
        if vectorized:
            if jnp.ndim(keys) != 0:
                # Callers hand over one key per particle; a vectorized rproc
                # needs only one, so the rest are discarded.
                keys = keys[0]
            # Carry the state as separate columns so that it is not sliced and
            # restacked on every sub-step.
            X_ = {n: X_[:, i] for i, n in enumerate(snames)}
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
        X_out = final[0]
        if vectorized:
            X_out = jnp.stack([X_out[n] for n in snames], axis=-1)
        return X_out, final[3]  # Return X_ and new t_idx

    return _rproc_interp
