"""
This file contains the classes for components that define the model mechanics.
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


def _get_annotation_tag(annotation: Any) -> str | None:
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


# --- Base Component Class ---

_DUMMY_J = 3  # Particle count used when validating a vectorized component.


class _ModelComponent:
    """Base class handling initialization, signature alignment, and validation."""

    # Subclasses must define these:
    internal_names: list[str]
    vmap_axes_pf: tuple[int | None, ...]
    vmap_axes_per: tuple[int | None, ...]

    # Components that support manual vectorization override this (see _RProc).
    _is_vectorized: bool = False

    statenames: list[str]
    param_names: list[str]
    covar_names: list[str]
    par_trans: ParTrans
    y_names: list[str]
    original_func: Callable
    name_mapping: dict[str, str]
    mechanics: Callable
    mechanics_pf: Callable
    mechanics_per: Callable

    def __init__(
        self,
        mechanics: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
        y_names: list[str] | None = None,
        validate_logic: bool = True,
    ) -> None:
        self.statenames = statenames
        self.param_names = param_names
        self.covar_names = covar_names
        self.par_trans = par_trans
        self.y_names = y_names or []
        self.original_func = mechanics

        # 1. Validation of list inputs
        for name, lst in [
            ("statenames", statenames),
            ("param_names", param_names),
            ("covar_names", covar_names),
        ]:
            if not isinstance(lst, list) or not all(isinstance(s, str) for s in lst):
                raise ValueError(f"{name} must be a list of strings")

        # 2. Align Arguments
        self.name_mapping = _align_by_type(mechanics, self.internal_names)

        # 3. Validate Logic (Dry Run)
        if validate_logic:
            self._validate_call()

        # 4. Create Wrappers
        self._build_mechanics()

    def _build_mechanics(self) -> None:
        """Create the internal callables, adding a particle axis via vmap."""
        self.mechanics = self._make_wrapper()
        self.mechanics_pf = jax.vmap(self.mechanics, self.vmap_axes_pf)
        self.mechanics_per = jax.vmap(self.mechanics, self.vmap_axes_per)

    def _get_dummies(self) -> dict[str, Any]:
        """Generate dummy data for validation."""
        states: dict[str, Any] = (
            {n: jnp.full((_DUMMY_J,), 0.1) for n in self.statenames}
            if self._is_vectorized
            else {n: 0.1 for n in self.statenames}
        )
        return {
            "X_": states,
            "theta_": {n: 0.1 for n in self.param_names},
            "covars": {n: 0.1 for n in self.covar_names},
            "Y_": {n: 0.1 for n in self.y_names},
            "t": 0.0,
            "t0": 0.0,
            "dt": 0.1,
            "key": jax.random.key(0),
        }

    def _validate_call(self) -> None:
        """Generic validator that runs the user function once with dummy data."""
        dummies = self._get_dummies()
        try:
            result = self._call_original(**dummies)
        except (AttributeError, TypeError) as e:
            raise TypeError(
                f"Error running '{self.original_func.__name__}': {e}.\n"
                "HINT: Check that you are treating inputs as dicts (not arrays) "
                "and that argument order/types are correct."
            ) from e

        self._validate_output(result)

    def _prepare_theta(
        self, theta_arr: jax.Array, should_trans: bool = False, batched: bool = False
    ) -> dict[str, Any]:
        """Convert theta array to dict and optionally transform from estimation scale."""
        pnames, trans = self.param_names, self.par_trans
        theta_dict: dict[str, Any] = (
            {n: theta_arr[:, i] for i, n in enumerate(pnames)}
            if batched
            else {n: theta_arr[i] for i, n in enumerate(pnames)}
        )
        if should_trans and len(pnames) > 0:
            theta_dict = trans.from_est(theta_dict)
        return theta_dict

    def _unpack_covars(self, covars: jax.Array | None) -> dict[str, Any]:
        """Convert covariate array to dict."""
        if covars is None:
            return {}
        return {n: covars[i] for i, n in enumerate(self.covar_names)}

    def _unpack_states(self, X_arr: jax.Array) -> dict[str, Any]:
        """Convert state array to dict."""
        return {n: X_arr[i] for i, n in enumerate(self.statenames)}

    def _unpack_obs(self, Y_arr: jax.Array) -> dict[str, Any]:
        """Convert observation array to dict."""
        return {n: Y_arr[i] for i, n in enumerate(self.y_names)}

    def _pack_states(self, d: dict[str, Any]) -> jax.Array:
        """Pack state dictionary values into a 1D array."""
        return jnp.array([d[n] for n in self.statenames]).reshape(-1)

    def _pack_obs(self, d: dict[str, Any]) -> jax.Array:
        """Pack observation dictionary values into a 1D array."""
        return jnp.array([d[n] for n in self.y_names]).reshape(-1)

    def _call_original(self, **internal_kwargs: Any) -> Any:
        """Call user-supplied mechanics function using mapped argument names."""
        mapping = self.name_mapping
        return self.original_func(
            **{mapping[k]: v for k, v in internal_kwargs.items() if k in mapping}
        )

    def _validate_dict_output(
        self,
        result: Any,
        required_keys: list[str],
        component_name: str,
        key_label: str = "state",
    ) -> None:
        """Validate that result is a dict containing all required keys."""
        if not isinstance(result, dict):
            raise TypeError(
                f"{component_name} function must return a dict, got {type(result)}"
            )
        missing = set(required_keys) - set(result.keys())
        if missing:
            raise ValueError(
                f"{component_name} function output missing {key_label} keys: {missing}"
            )

    def _validate_scalar_output(self, result: Any, component_name: str) -> None:
        """Validate that result is a scalar (Python number or 0-d JAX array)."""
        is_jax_scalar = (
            hasattr(result, "shape") or hasattr(result, "__jax_array__")
        ) and jnp.ndim(result) == 0
        if not (isinstance(result, (int, float, np.number)) or is_jax_scalar):
            raise TypeError(
                f"{component_name} function must return a scalar (float or 0-d array). "
                f"Got {type(result)} with shape {getattr(result, 'shape', 'N/A')}"
            )

    def _validate_output(self, result: Any) -> None:
        """Validates the output of the user's mechanics function."""
        raise NotImplementedError

    def _make_wrapper(self) -> Callable:
        """Wraps the user's mechanics function with the internal API."""
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return (
            self.statenames == other.statenames  # type: ignore[attr-defined]
            and self.param_names == other.param_names  # type: ignore[attr-defined]
            and self.original_func == other.original_func  # type: ignore[attr-defined]
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

    This exists as a performance optimization. XLA's CPU backend does
    not vectorize per-particle random number generation across the particle
    batch, so a manually vectorized ``rproc`` can be several times faster on CPU.
    On GPU, carrying state as separate columns (which both default vmap and
    ``@vectorized`` now use) provides substantial speedups at large effective batch sizes
    (e.g., when vmapping over many replicates and particles) where the device is compute-bound.

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
        mechanics (Callable): The user-defined initialization function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.

    User Function Structure
    -----------------------
    The `mechanics` function receives parameters, a PRNG key, covariates, and the initial time.
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

    def _validate_output(self, result: Any) -> None:
        self._validate_dict_output(result, self.statenames, "rinit", "state")

    def _make_wrapper(self) -> Callable:
        def wrapped(
            theta_arr: jax.Array,
            key: jax.Array,
            covars: jax.Array | None,
            t0: float,
            should_trans: bool,
        ) -> jax.Array:
            res = self._call_original(
                theta_=self._prepare_theta(theta_arr, should_trans),
                key=key,
                covars=self._unpack_covars(covars),
                t0=t0,
            )
            return self._pack_states(res)

        return wrapped


class _RProc(_ModelComponent):
    """
    Defines the process model (state transitions) of the system.

    Args:
        mechanics (Callable): The user-defined stepping function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.
        nstep (int, optional): Number of steps per observation interval.
        dt (float, optional): Fixed time step size (mutually exclusive with nstep).
        accumvars (tuple[int, ...], optional): Indices of states to zero-out at each observation.

    User Function Structure
    -----------------------
    The `mechanics` function performs a **single Euler step**. It receives the current state,
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

    _is_vectorized: bool
    nstep: int | None
    dt: float | None
    accumvars: tuple[int, ...] | None
    mechanics_pf_interp: Callable
    mechanics_per_interp: Callable

    internal_names = ["X_", "theta_", "key", "covars", "t", "dt"]
    vmap_axes_pf = (0, None, 0, None, None, None, None)
    vmap_axes_per = (0, 0, 0, None, None, None, None)

    def __init__(
        self,
        mechanics: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
        nstep: int | None = None,
        dt: float | None = None,
        accumvars: tuple[int, ...] | None = None,
        validate_logic: bool = True,
        nstep_array: np.ndarray | None = None,
        vectorized: bool | None = None,
    ) -> None:
        if dt is not None and nstep is not None:
            raise ValueError("Only nstep or dt can be provided, not both")

        # Set before super().__init__() because validation and wrapper creation
        # both depend on it.
        if vectorized is None:
            vectorized = bool(getattr(mechanics, "_pypomp_vectorized", False))
        self._is_vectorized = vectorized

        self.nstep = int(nstep) if nstep is not None else None
        self.dt = float(dt) if dt is not None else None
        self.accumvars = accumvars

        # If nstep is given, interpolated functions use it in order to have a fixed
        # number of steps. This is necessary for train to work.
        if nstep_array is not None:
            nstep_arr = np.asarray(nstep_array)
            all_nstep_same = np.min(nstep_arr) == np.max(nstep_arr)
            # If nstep is the same for all intervals (which can happen even if derived
            # from dt), use it for the interpolated functions.
            if all_nstep_same:
                self.nstep = int(np.min(nstep_arr))

        super().__init__(
            mechanics,
            statenames,
            param_names,
            covar_names,
            par_trans,
            validate_logic=validate_logic,
        )

    def _validate_output(self, result: Any) -> None:
        self._validate_dict_output(result, self.statenames, "rproc", "state")

    def _build_step_fns(
        self,
    ) -> tuple[
        tuple[Callable, Callable],
        tuple[Callable, Callable],
    ]:
        """Build the (step_fn, prepare_theta) pairs for pf and per axes."""
        snames = self.statenames
        prepare_pf = partial(self._prepare_theta, batched=False)
        prepare_per = partial(self._prepare_theta, batched=True)

        if self._is_vectorized:

            def step_vec(X_dict, theta_dict, key, covars_t, t, dt):
                J = X_dict[snames[0]].shape[0]
                covars_dict = self._unpack_covars(covars_t)
                res = self._call_original(
                    X_=X_dict,
                    theta_=theta_dict,
                    key=key,
                    covars=covars_dict,
                    t=t,
                    dt=dt,
                )
                return {n: jnp.broadcast_to(jnp.asarray(res[n]), (J,)) for n in snames}

            return (step_vec, prepare_pf), (step_vec, prepare_per)

        def single_step(X_dict, theta_dict, key, covars_dict, t, dt):
            res = self._call_original(
                X_=X_dict,
                theta_=theta_dict,
                key=key,
                covars=covars_dict,
                t=t,
                dt=dt,
            )
            return {n: res[n] for n in snames}

        vmap_pf = jax.vmap(single_step, in_axes=(0, None, 0, None, None, None))
        vmap_per = jax.vmap(single_step, in_axes=(0, 0, 0, None, None, None))

        def step_pf(X_dict, theta_dict, key, covars_t, t, dt):
            J = X_dict[snames[0]].shape[0]
            step_keys = jax.random.split(key, J)
            covars_dict = self._unpack_covars(covars_t)
            return vmap_pf(X_dict, theta_dict, step_keys, covars_dict, t, dt)

        def step_per(X_dict, theta_dict, key, covars_t, t, dt):
            J = X_dict[snames[0]].shape[0]
            step_keys = jax.random.split(key, J)
            covars_dict = self._unpack_covars(covars_t)
            return vmap_per(X_dict, theta_dict, step_keys, covars_dict, t, dt)

        return (step_pf, prepare_pf), (step_per, prepare_per)

    def _build_mechanics(self) -> None:
        """Both vectorized and default paths thread state through sub-step loop."""
        (step_pf, prepare_pf), (step_per, prepare_per) = self._build_step_fns()
        self.mechanics_pf_interp = _time_interp(
            step_pf,
            nstep_fixed=self.nstep,
            statenames=self.statenames,
            prepare_theta=prepare_pf,
        )
        self.mechanics_per_interp = _time_interp(
            step_per,
            nstep_fixed=self.nstep,
            statenames=self.statenames,
            prepare_theta=prepare_per,
        )

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and isinstance(other, _RProc)
            and self.nstep == other.nstep
            and self.dt == other.dt
            and self.accumvars == other.accumvars
            and self._is_vectorized == other._is_vectorized
        )


class _DMeas(_ModelComponent):
    """
    Defines the measurement density (likelihood) model.

    Args:
        mechanics (Callable): The user-defined density function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.
        y_names (list[str], optional): List of observation names.

    User Function Structure
    -----------------------
    The `mechanics` function calculates the log-likelihood of the data given the state.
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

    def _validate_output(self, result: Any) -> None:
        self._validate_scalar_output(result, "dmeas")

    def _make_wrapper(self) -> Callable:
        def wrapped(
            Y_arr: jax.Array,
            X_arr: jax.Array,
            theta_arr: jax.Array,
            covars: jax.Array | None,
            t: float,
            should_trans: bool,
        ) -> jax.Array | float:
            return self._call_original(
                Y_=self._unpack_obs(Y_arr),
                X_=self._unpack_states(X_arr),
                theta_=self._prepare_theta(theta_arr, should_trans),
                covars=self._unpack_covars(covars),
                t=t,
            )

        return wrapped


class _RMeas(_ModelComponent):
    """
    Defines the measurement simulation model (observation process).

    Args:
        mechanics (Callable): The user-defined simulation function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.
        y_names (list[str], optional): List of observation names.

    User Function Structure
    -----------------------
    The `mechanics` function simulates observations from the current state.
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

    ydim: int

    internal_names = ["X_", "theta_", "key", "covars", "t"]
    vmap_axes_pf = (0, None, 0, None, None, None)
    vmap_axes_per = (0, 0, 0, None, None, None)

    def __init__(
        self,
        mechanics: Callable,
        statenames: list[str],
        param_names: list[str],
        covar_names: list[str],
        par_trans: ParTrans,
        y_names: list[str] | None = None,
        validate_logic: bool = True,
    ) -> None:
        self.ydim = len(y_names) if y_names is not None else 0

        super().__init__(
            mechanics,
            statenames,
            param_names,
            covar_names,
            par_trans,
            y_names=y_names,
            validate_logic=validate_logic,
        )

    def _validate_output(self, result: Any) -> None:
        self._validate_dict_output(result, self.y_names, "rmeas", "observation")

    def _make_wrapper(self) -> Callable:
        def wrapped(
            X_arr: jax.Array,
            theta_arr: jax.Array,
            key: jax.Array,
            covars: jax.Array | None,
            t: float,
            should_trans: bool,
        ) -> jax.Array:
            res = self._call_original(
                X_=self._unpack_states(X_arr),
                theta_=self._prepare_theta(theta_arr, should_trans),
                key=key,
                covars=self._unpack_covars(covars),
                t=t,
            )
            return self._pack_obs(res)

        return wrapped


class _DPrior(_ModelComponent):
    """
    Defines the prior log-density model for parameter values.

    Args:
        mechanics (Callable): The user-defined prior function.
        statenames (list[str]): List of state variable names.
        param_names (list[str]): List of parameter names.
        covar_names (list[str]): List of covariate names.
        par_trans (ParTrans): Parameter transformation object.

    User Function Structure
    -----------------------
    The `mechanics` function calculates the log-prior density given parameter values.
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

    def _validate_output(self, result: Any) -> None:
        self._validate_scalar_output(result, "dprior")

    def _make_wrapper(self) -> Callable:
        pnames, trans = self.param_names, self.par_trans

        def _from_est_array(arr: jax.Array) -> jax.Array:
            d_in: dict[str, Any] = {n: arr[i] for i, n in enumerate(pnames)}
            d_out = trans.from_est(d_in)
            return jnp.stack([d_out[n] for n in pnames])

        def wrapped(
            theta_arr: jax.Array, should_trans: bool = False
        ) -> jax.Array | float:
            theta_dict = self._prepare_theta(theta_arr, should_trans)
            if should_trans and len(pnames) > 0:
                J = jax.jacobian(_from_est_array)(theta_arr)
                log_det_J = jnp.linalg.slogdet(J)[1]
            else:
                log_det_J = 0.0

            log_p = self._call_original(theta_=theta_dict)
            return log_p + log_det_J

        return wrapped


def _flat_dprior(params: ParamDict) -> float:
    """Flat improper prior -- always returns 0.0."""
    return 0.0


# --- Interpolation Helper
def _time_interp(
    step_fn: Callable,
    nstep_fixed: int | None,
    statenames: list[str],
    prepare_theta: Callable | None = None,
) -> Callable:
    """Constructs an interpolated version of rproc.

    Args:
        step_fn (Callable): Single Euler sub-step simulation callable.
        nstep_fixed (int | None): Fixed number of sub-steps per observation
            interval, or None if specified dynamically at runtime.
        statenames (list[str]): List of state variable names.
        prepare_theta (Callable | None): Optional callable to transform or prepare
            parameters once per observation interval. Defaults to None.

    Returns:
        Callable: Interpolated version of rproc.
    """
    if not statenames:
        raise ValueError("statenames are required to build rproc interp")
    snames: list[str] = list(statenames)

    def _interp_step(
        i: int | jax.Array,
        inputs: tuple[dict[str, jax.Array], jax.Array, jax.Array, int],
        theta_: dict[str, jax.Array],
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, int]:
        X_dict, keys, t, t_idx = inputs
        covars_t = covars_extended[t_idx] if covars_extended is not None else None
        dt = dt_array_extended[t_idx]
        next_key, subkey = jax.random.split(keys)
        X_dict = step_fn(X_dict, theta_, subkey, covars_t, t, dt)
        return (X_dict, next_key, t + dt, t_idx + 1)

    def _rproc_interp(
        X_: jax.Array,
        theta_: dict[str, jax.Array],
        keys: jax.Array,
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
        t: float,
        t_idx: int,
        nstep_dynamic: int,
        accumvars: tuple[int, ...] | None,
        should_trans: bool,
    ) -> tuple[jax.Array, int]:
        """Interpolated version of rproc."""
        if accumvars is not None and len(accumvars) > 0:
            X_ = X_.at[:, accumvars].set(0)
        if jnp.ndim(keys) != 0:
            # Callers may hand over one key per particle; extract a single scalar seed key.
            keys = keys[0]
        # Carry the state as separate columns so that it is not sliced and
        # restacked on every sub-step.
        X_dict = {n: X_[:, i] for i, n in enumerate(snames)}
        nstep = nstep_fixed if nstep_fixed is not None else nstep_dynamic

        if prepare_theta is not None:
            theta_ = prepare_theta(theta_, should_trans)

        final = jax.lax.fori_loop(
            0,
            nstep,
            partial(
                _interp_step,
                theta_=theta_,
                covars_extended=covars_extended,
                dt_array_extended=dt_array_extended,
            ),
            (X_dict, keys, t, t_idx),
        )
        X_dict_out = final[0]
        X_out = jnp.stack([X_dict_out[n] for n in snames], axis=-1)
        return X_out, final[3]  # Return state and new t_idx

    return _rproc_interp
