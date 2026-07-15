"""
Object-oriented interface for defining and fitting POMP models.

The :class:`Pomp` class is the primary entry point for single-unit POMP
modelling in Pypomp.  It wraps user-supplied simulator and density
functions, validates their signatures, and exposes a high-level API for
particle filtering, iterated filtering, and gradient-based training.
"""

import importlib
import cloudpickle
from copy import deepcopy
from typing import Callable, Any, cast

import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import warnings

from .metadata import ModelMetadata
from .model_struct import _RInit, _RProc, _DMeas, _RMeas
from .algorithms.helpers import _calc_ys_covars
from .par_trans import ParTrans
from .results import ResultsHistory
from .parameters import PompParameters
from pypomp.functional.structs import PompStruct

from .estimation_mixin import PompEstimationMixin
from .analysis_mixin import PompAnalysisMixin


class Pomp(PompEstimationMixin, PompAnalysisMixin):
    """
    Define and fit a partially observed Markov process (POMP) model.

    A POMP model describes a time series whose latent state evolves according
    to a Markov process that is only partially observable through noisy
    measurements.  This class encapsulates the four model components — initial
    state distribution (``rinit``), state transition (``rproc``), measurement
    density (``dmeas``), and measurement simulator (``rmeas``) — and exposes
    methods for simulation, particle filtering, iterated filtering, and
    gradient-based training.

    .. important::

        The ``rinit``, ``rproc``, ``dmeas``, and ``rmeas`` arguments expect
        user-defined functions with **specific argument names and type hints**.
        The ``Pomp`` object raises an error at construction time if these
        functions do not conform to the specification.

        - **rinit**: See :ref:`rinit-tutorial`.
        - **rproc**: See :ref:`rproc-tutorial`.
        - **dmeas**: See :ref:`dmeas-tutorial`.
        - **rmeas**: See :ref:`rmeas-tutorial`.

    Parameters
    ----------
    ys : pd.DataFrame
        Measurement data frame.  The index must contain the observation times
        as numeric values.
    theta : PompParameters
        Initial parameter set(s).  Pass a :class:`~pypomp.PompParameters`
        object with multiple parameter sets to run estimation methods in parallel over
        multiple starting points.
    statenames : list of str
        Names of all latent state variables in the process model.
    t0 : float
        Initial time for the model, typically just before the first
        observation.
    rinit : callable
        Initial state simulator.  See :ref:`rinit-tutorial` for the required
        signature.
    rproc : callable
        State transition simulator for a single time step.  See
        :ref:`rproc-tutorial` for the required signature.
    dmeas : callable or None, optional
        Measurement log-density function.  Required for particle filtering
        and iterated filtering.  See :ref:`dmeas-tutorial`.
    rmeas : callable or None, optional
        Measurement simulator.  Required for :meth:`simulate`.  See
        :ref:`rmeas-tutorial`.
    par_trans : ParTrans or None, optional
        Parameter transformation object mapping between the natural
        parameter space and the estimation space.  Defaults to the
        identity transformation.
    covars : pd.DataFrame or None, optional
        Time-varying covariate data frame.  The index must contain numeric
        covariate times.  Interpolated to the integration grid at runtime.
    nstep : int or None, optional
        Number of Euler integration steps between consecutive observations.
        Mutually exclusive with ``dt``.
    dt : float or None, optional
        Fixed integration step size.  Mutually exclusive with ``nstep``.
    accumvars : list of str or None, optional
        Names of accumulator state variables (e.g. incidence counters) that
        are reset to zero at the start of each observation interval.
    validate_logic : bool, optional
        Whether to validate model component function signatures and logic
        at construction time.  Defaults to ``True``.
    order : str, optional
        Covariate interpolation method: ``"linear"`` (default) or
        ``"constant"`` (left-step).

    Examples
    --------
    Build a minimal SIR-like POMP model:

    >>> import pandas as pd
    >>> import jax
    >>> import pypomp as pp
    >>> from pypomp.types import StateDict, ParamDict, TimeFloat, StepSizeFloat, RNGKey, ObservationDict
    >>>
    >>> def my_rinit(theta_: ParamDict, t0: TimeFloat, key: RNGKey) -> StateDict:
    ...     return {"S": 990.0, "I": 10.0, "R": 0.0}
    >>>
    >>> def my_rproc(X_: StateDict, theta_: ParamDict, t: TimeFloat, dt: StepSizeFloat, key: RNGKey) -> StateDict:
    ...     return X_  # identity placeholder
    >>>
    >>> def my_dmeas(Y_: ObservationDict, X_: StateDict, theta_: ParamDict, t: TimeFloat) -> float:
    ...     import jax.numpy as jnp
    ...     return jnp.array(0.0)
    >>>
    >>> ys = pd.DataFrame({"cases": [10, 12, 15]}, index=[1.0, 2.0, 3.0])
    >>> theta = pp.PompParameters({"beta": 0.5, "gamma": 0.1})
    >>> model = pp.Pomp(
    ...     ys=ys, theta=theta, statenames=["S", "I", "R"], t0=0.0,
    ...     rinit=my_rinit, rproc=my_rproc, dmeas=my_dmeas, dt=0.1,
    ... )

    See Also
    --------
    pypomp.panel.PanelPomp : Multi-unit panel extension of this class.
    pypomp.core.parameters.PompParameters : Parameter container for single-unit models.
    """

    ys: pd.DataFrame
    """The measurement data frame with observation times as the index."""

    _theta: PompParameters | None
    """Internal storage for model parameters in canonical order."""

    canonical_param_names: list[str]
    """Ordered list of parameter names used throughout the model."""

    statenames: list[str]
    """Names of all latent state variables in the process model."""

    t0: float
    """Initial time for the model (typically before the first observation)."""

    rinit: _RInit
    """Simulator for the initial state distribution."""

    rproc: _RProc
    """Process model simulator handling state transitions between observation times."""

    dmeas: _DMeas | None
    """Measurement density used to evaluate the likelihood of observations."""

    rmeas: _RMeas | None
    """Measurement simulator used to generate synthetic observations."""

    par_trans: ParTrans
    """Parameter transformation object mapping between natural and estimation spaces."""

    covars: pd.DataFrame | None
    """Time-varying covariates for the model, if applicable."""

    _covars_extended: np.ndarray | None
    """Internal covariate array interpolated/aligned to the integration grid."""

    _nstep_array: np.ndarray
    """Number of integration steps between successive observation times."""

    _dt_array_extended: np.ndarray
    """Time step sizes for each integration step over the full time grid."""

    _max_steps_per_interval: int
    """Maximum number of integration steps between any two observation times."""

    accumvars: list[str] | None
    """Names of accumulator state variables that are reset at each observation time."""

    _accumvars_indices: tuple[int, ...] | None
    """Indices of accumulator state variables within the full state vector."""

    results_history: ResultsHistory
    """A :class:`~pypomp.core.results.ResultsHistory` object storing the history of results from :meth:`pfilter`, :meth:`mif`, and :meth:`train` calls."""

    fresh_key: jax.Array | None
    """Running a method that accepts a JAX PRNG key will store a fresh, unused key here."""

    metadata: ModelMetadata
    """Environment and version metadata initialized when this instance was built."""

    def __init__(
        self,
        ys: pd.DataFrame,
        theta: PompParameters,
        statenames: tuple[str, ...] | list[str],
        t0: float,
        rinit: Callable,
        rproc: Callable,
        dmeas: Callable | None = None,
        rmeas: Callable | None = None,
        par_trans: ParTrans | None = None,
        nstep: int | None = None,
        dt: float | None = None,
        accumvars: tuple[str, ...] | list[str] | None = None,
        covars: pd.DataFrame | None = None,
        validate_logic: bool = True,
        order: str = "linear",
    ):
        if not isinstance(ys, pd.DataFrame):
            raise TypeError("ys must be a pandas DataFrame")
        if covars is not None and not isinstance(covars, pd.DataFrame):
            raise TypeError("covars must be a pandas DataFrame or None")

        if not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters instance")
        self._theta = theta

        # Extract parameter names from first theta dict
        self.canonical_param_names = self._theta.get_param_names()

        # If statenames not provided, we need to infer them
        if statenames is None:
            raise ValueError(
                "statenames must be provided as a list of state variable names"
            )

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a tuple or list of strings")

        if accumvars is not None:
            if not all(isinstance(name, str) for name in accumvars):
                raise ValueError("accumvars must be a tuple or list of strings")
            if not all(name in statenames for name in accumvars):
                raise ValueError("all accumvars must be in statenames")
            self._accumvars_indices = tuple(
                tuple(statenames).index(name) for name in accumvars
            )
        else:
            self._accumvars_indices = None

        self.statenames = list(statenames)
        self.accumvars = list(accumvars) if accumvars is not None else None
        self.ys = ys
        self.covars = covars
        self.t0 = float(t0)
        self.results_history = ResultsHistory()
        self.fresh_key = None
        self.metadata = ModelMetadata()

        if covars is not None:
            self.covar_names = list(covars.columns)
        else:
            self.covar_names = []

        self.par_trans = par_trans or ParTrans()
        self.rinit = _RInit(
            struct=rinit,
            statenames=self.statenames,
            param_names=self.canonical_param_names,
            covar_names=self.covar_names,
            par_trans=self.par_trans,
            validate_logic=validate_logic,
        )

        if dmeas is not None:
            self.dmeas = _DMeas(
                struct=dmeas,
                statenames=self.statenames,
                param_names=self.canonical_param_names,
                covar_names=self.covar_names,
                par_trans=self.par_trans,
                y_names=list(self.ys.columns),
                validate_logic=validate_logic,
            )
        else:
            self.dmeas = None

        if rmeas is not None:
            self.rmeas = _RMeas(
                struct=rmeas,
                statenames=self.statenames,
                param_names=self.canonical_param_names,
                covar_names=self.covar_names,
                par_trans=self.par_trans,
                y_names=list(self.ys.columns),
                validate_logic=validate_logic,
            )
        else:
            self.rmeas = None

        if self.dmeas is None and self.rmeas is None:
            raise ValueError("You must supply at least one of dmeas or rmeas")

        (
            self._covars_extended,
            self._dt_array_extended,
            self._nstep_array,
            self._max_steps_per_interval,
        ) = _calc_ys_covars(
            t0=self.t0,
            times=np.array(self.ys.index),
            ctimes=np.array(self.covars.index) if self.covars is not None else None,
            covars=np.array(self.covars) if self.covars is not None else None,
            dt=dt,
            nstep=nstep,
            order=order,
        )

        self.rproc = _RProc(
            struct=rproc,
            statenames=self.statenames,
            param_names=self.canonical_param_names,
            covar_names=self.covar_names,
            par_trans=self.par_trans,
            nstep=nstep,
            dt=dt,
            accumvars=self._accumvars_indices,
            validate_logic=validate_logic,
            nstep_array=self._nstep_array,
            max_steps_bound=self._max_steps_per_interval,
        )

    @property
    def theta(self) -> PompParameters:
        """The current parameter set for the model.

        Returns
        -------
        PompParameters
            The active :class:`~pypomp.PompParameters` object.

        Raises
        ------
        ValueError
            If ``theta`` has not been set.
        """
        if self._theta is None:
            raise ValueError("Model parameters have not been set (theta is None).")
        return self._theta

    @theta.setter
    def theta(self, value: PompParameters | None):
        if value is not None and not isinstance(value, PompParameters):
            raise TypeError("theta must be a PompParameters instance")
        self._theta = value

    def _prepare_theta_input(
        self,
        theta: PompParameters | None,
    ) -> PompParameters:
        """
        Prepare the theta input for the method.
        """
        if theta is None:
            return self.theta
        if not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters object or None")
        if set(theta.get_param_names()) != set(self.canonical_param_names):
            raise ValueError(
                "theta parameter names must match canonical_param_names up to reordering"
            )
        return theta

    def _update_fresh_key(
        self, key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Updates the fresh_key attribute and returns a new key along with the old key.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple containing the new key and the old key.
                The old key is the key that was used to update the fresh_key attribute.
                The new key is the key that should be used for the next method call.
        """
        old_key = self.fresh_key if key is None else key
        if old_key is None:
            raise ValueError(
                "Both the key argument and the fresh_key attribute are None. At least one key must be given."
            )
        self.fresh_key, new_key = jax.random.split(old_key)
        return new_key, old_key

    def to_struct(self) -> PompStruct:
        """Export the model to a lightweight JAX-compatible struct.

        Packs the static data arrays and compiled simulator callables into a
        :class:`~pypomp.functional.PompStruct` NamedTuple suitable for use
        with the functional API (``pypomp.functional``).

        Returns
        -------
        PompStruct
            The compiled structural representation of the model.

        See Also
        --------
        pypomp.functional.pfilter : Functional particle filter.
        pypomp.functional.mif : Functional iterated filter.
        """
        return PompStruct(
            ys=jnp.array(self.ys),
            dt_array_extended=jnp.array(self._dt_array_extended),
            nstep_array=jnp.array(self._nstep_array),
            t0=self.t0,
            times=jnp.array(self.ys.index),
            covars_extended=jnp.array(self._covars_extended)
            if self._covars_extended is not None
            else None,
            accumvars=self.rproc.accumvars,
            rinit_pf=self.rinit.struct_pf,
            rproc_pf=self.rproc.struct_pf_interp,
            dmeas_pf=self.dmeas.struct_pf if self.dmeas is not None else None,
            rinit_per=self.rinit.struct_per,
            rproc_per=self.rproc.struct_per_interp,
            dmeas_per=self.dmeas.struct_per if self.dmeas is not None else None,
            rmeas_pf=self.rmeas.struct_pf if self.rmeas is not None else None,
            par_trans=self.par_trans,
            param_names=self.canonical_param_names,
        )

    def print_metadata(self) -> None:
        """Display environment and version metadata for this model instance.

        Prints the creation timestamp and the versions of key dependencies
        (pypomp, JAX, etc.) captured when this :class:`Pomp` object
        was constructed.  Useful for reproducibility and debugging.
        """
        self.metadata.print_metadata()

    def __eq__(self, other):
        """Check structural equality with another :class:`Pomp` object.

        Two instances are considered equal if they have identical parameter
        names, parameter values, data (``ys``, ``covars``), state names,
        initial time, model components (``rinit``, ``rproc``, ``dmeas``,
        ``rmeas``), ``par_trans``, ``results_history``, and ``fresh_key``.
        """
        if not isinstance(other, type(self)):
            return False

        # Canonical parameter names
        if self.canonical_param_names != other.canonical_param_names:
            return False

        # Parameter sets
        if (self._theta is None) != (other._theta is None):
            return False
        if self._theta is not None and other._theta is not None:
            if self._theta != other._theta:
                return False

        # Data and covariates
        if not self.ys.equals(other.ys):
            return False
        if (self.covars is None) != (other.covars is None):
            return False
        if self.covars is not None and other.covars is not None:
            if not self.covars.equals(other.covars):
                return False
        # Handle _covars_extended (can be None or JAX array)
        if (self._covars_extended is None) != (other._covars_extended is None):
            return False
        if self._covars_extended is not None and other._covars_extended is not None:
            if not jax.numpy.array_equal(self._covars_extended, other._covars_extended):
                return False
        # Compare JAX arrays using array_equal
        if not jax.numpy.array_equal(self._nstep_array, other._nstep_array):
            return False
        if not jax.numpy.array_equal(self._dt_array_extended, other._dt_array_extended):
            return False
        if self._max_steps_per_interval != other._max_steps_per_interval:
            return False

        # State names and initial time
        if self.statenames != other.statenames:
            return False
        if float(self.t0) != float(other.t0):
            return False

        # Model components: rely on their own __eq__ implementations
        if self.rinit != other.rinit:
            return False
        if self.rproc != other.rproc:
            return False
        if (self.dmeas is None) != (other.dmeas is None):
            return False
        if self.dmeas is not None and self.dmeas != other.dmeas:
            return False
        if (self.rmeas is None) != (other.rmeas is None):
            return False
        if self.rmeas is not None and self.rmeas != other.rmeas:
            return False

        if self.results_history != other.results_history:
            return False

        if self.par_trans != other.par_trans:
            return False

        # fresh_key: both None or numerically equal
        if (self.fresh_key is None) != (other.fresh_key is None):
            return False
        if self.fresh_key is not None and other.fresh_key is not None:
            if not jax.numpy.array_equal(
                jax.random.key_data(self.fresh_key),
                jax.random.key_data(other.fresh_key),
            ):
                return False

        return True

    @staticmethod
    def merge(*pomp_objs: "Pomp") -> "Pomp":
        """Merge multiple :class:`Pomp` objects into a single instance.

        Combines their parameter replicates and results histories.  All
        objects must share identical structural components (data, state
        names, model functions, and parameter names).  Useful for
        consolidating results from parallel estimation runs.

        Parameters
        ----------
        *pomp_objs : Pomp
            Two or more :class:`Pomp` objects to merge.  Must be
            structurally identical (same ``ys``, ``statenames``, ``rinit``,
            ``rproc``, ``dmeas``, ``rmeas``, and ``par_trans``).

        Returns
        -------
        Pomp
            A new :class:`Pomp` instance whose ``theta`` and
            ``results_history`` are the concatenation of all inputs.
        """
        if len(pomp_objs) == 0:
            raise ValueError("At least one Pomp object must be provided.")
        first = pomp_objs[0]

        for obj in pomp_objs:
            if not isinstance(obj, type(first)):
                raise TypeError("All merged objects must be of type Pomp.")
            if obj.canonical_param_names != first.canonical_param_names:
                raise ValueError(
                    "All Pomp objects must have the same canonical_param_names."
                )
            if obj.statenames != first.statenames:
                raise ValueError("All Pomp objects must have the same statenames.")
            if not obj.ys.equals(first.ys):
                raise ValueError("All Pomp objects must have the same ys data.")
            if obj.t0 != first.t0:
                raise ValueError("All Pomp objects must have the same t0.")
            if obj.rinit != first.rinit or obj.rproc != first.rproc:
                raise ValueError("All Pomp objects must have the same rinit and rproc.")
            if (obj.dmeas is None) != (first.dmeas is None):
                raise ValueError(
                    "All Pomp objects must have the same dmeas (both None or both not None)."
                )
            if obj.dmeas is not None and obj.dmeas != first.dmeas:
                raise ValueError("All Pomp objects must have the same dmeas.")
            if (obj.rmeas is None) != (first.rmeas is None):
                raise ValueError(
                    "All Pomp objects must have the same rmeas (both None or both not None)."
                )
            if obj.rmeas is not None and obj.rmeas != first.rmeas:
                raise ValueError("All Pomp objects must have the same rmeas.")
            if obj.par_trans != first.par_trans:
                raise ValueError("All Pomp objects must have the same par_trans.")

        thetas = []
        for obj in pomp_objs:
            if obj._theta is None:
                raise ValueError("Cannot merge Pomp objects with no parameters.")
            thetas.append(obj._theta)

        merged_theta = PompParameters.merge(*thetas)
        merged_history = ResultsHistory.merge(
            *[obj.results_history for obj in pomp_objs]
        )

        merged_pomp = deepcopy(first)
        merged_pomp._theta = merged_theta
        merged_pomp.results_history = merged_history
        merged_pomp.fresh_key = first.fresh_key

        return merged_pomp

    def __getstate__(self):
        """
        Custom pickling method to handle wrapped function objects.  This is
        necessary because the JAX-wrapped functions are not picklable.
        """
        state = self.__dict__.copy()

        # Use cloudpickle to store model functions by-value. This ensures that
        # the unpickling environment does not require the original source modules.
        if hasattr(self.rinit, "struct"):
            original_func = self.rinit.original_func
            state["_rinit_func_bytes"] = cloudpickle.dumps(original_func)

        if hasattr(self.rproc, "struct"):
            original_func = self.rproc.original_func
            state["_rproc_func_bytes"] = cloudpickle.dumps(original_func)
            state["_rproc_dt"] = getattr(self.rproc, "dt", None)
            state["_rproc_nstep"] = getattr(self.rproc, "nstep", None)
            state["_rproc_accumvars"] = getattr(self.rproc, "accumvars", None)

        if self.dmeas is not None and hasattr(self.dmeas, "struct"):
            original_func = self.dmeas.original_func
            state["_dmeas_func_bytes"] = cloudpickle.dumps(original_func)

        if self.rmeas is not None and hasattr(self.rmeas, "struct"):
            original_func = self.rmeas.original_func
            state["_rmeas_func_bytes"] = cloudpickle.dumps(original_func)

        # Store JAX key as raw bits (key is not picklable directly)
        if self.fresh_key is not None:
            state["_fresh_key_data"] = jax.random.key_data(self.fresh_key)

        # Remove the wrapped objects and key from state
        state.pop("rinit", None)
        state.pop("rproc", None)
        state.pop("dmeas", None)
        state.pop("rmeas", None)
        state.pop("fresh_key", None)

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct wrapped function objects. This is
        necessary because the JAX-wrapped functions are not picklable.
        """
        # Restore basic attributes
        self.__dict__.update(state)

        # Reconstruct JAX key from raw bits
        if "_fresh_key_data" in state:
            try:
                self.fresh_key = cast(
                    jax.Array, jax.random.wrap_key_data(state["_fresh_key_data"])
                )
            except Exception as e:
                warnings.warn(f"Failed to reconstruct JAX fresh_key: {e}", UserWarning)
                self.fresh_key = None
        elif "fresh_key" not in self.__dict__:
            self.fresh_key = None

        def _load_func(prefix: str) -> Any:
            func_bytes_key = f"_{prefix}_func_bytes"
            func_name_key = f"_{prefix}_func_name"
            module_key = f"_{prefix}_module"

            try:
                # Modern approach (by-value): Uses cloudpickle bytes to remove
                # environment dependencies.
                if func_bytes_key in state:
                    return cloudpickle.loads(state[func_bytes_key])

                # Legacy approach (by-reference): Provided for backward compatibility
                # with objects pickled in older versions of pypomp.
                elif func_name_key in state:
                    module = importlib.import_module(state[module_key])
                    return getattr(module, state[func_name_key])
            except Exception as e:
                warnings.warn(
                    f"Failed to reconstruct {prefix} function: {e}. "
                    f"The model may be unusable for simulations or estimation.",
                    UserWarning,
                )
            return None

        # Reconstruct rinit
        obj_rinit = _load_func("rinit")
        if obj_rinit is not None:
            if isinstance(obj_rinit, _RInit):
                self.rinit = obj_rinit
            else:
                self.rinit = _RInit(
                    struct=obj_rinit,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                )

        # Reconstruct rproc
        obj_rproc = _load_func("rproc")
        if obj_rproc is not None:
            if isinstance(obj_rproc, _RProc):
                self.rproc = obj_rproc
            else:
                kwargs = {}
                if state.get("_rproc_dt") is not None:
                    kwargs["dt"] = state["_rproc_dt"]
                if (
                    state.get("_rproc_nstep") is not None
                    and state.get("_rproc_dt") is None
                ):
                    kwargs["nstep"] = state["_rproc_nstep"]
                if state.get("_rproc_accumvars") is not None:
                    kwargs["accumvars"] = state["_rproc_accumvars"]
                self.rproc = _RProc(
                    struct=obj_rproc,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    **kwargs,
                )
                if state.get("_rproc_nstep") is not None:
                    if state.get("_rproc_dt") is not None:
                        self.rproc.nstep = state["_rproc_nstep"]

        # Reconstruct dmeas
        obj_dmeas = _load_func("dmeas")
        if obj_dmeas is not None:
            if isinstance(obj_dmeas, _DMeas):
                self.dmeas = obj_dmeas
            else:
                self.dmeas = _DMeas(
                    struct=obj_dmeas,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    y_names=list(self.ys.columns) if hasattr(self, "ys") else None,
                )

        # Reconstruct rmeas
        obj_rmeas = _load_func("rmeas")
        if obj_rmeas is not None:
            if isinstance(obj_rmeas, _RMeas):
                self.rmeas = obj_rmeas
            else:
                self.rmeas = _RMeas(
                    struct=obj_rmeas,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    y_names=list(self.ys.columns) if hasattr(self, "ys") else None,
                )

        # Set defaults if reconstruction failed or was missing
        if not hasattr(self, "rinit"):
            self.rinit = None  # type: ignore
        if not hasattr(self, "rproc"):
            self.rproc = None  # type: ignore
        if not hasattr(self, "rmeas"):
            self.rmeas = None
        if not hasattr(self, "dmeas"):
            self.dmeas = None

        # Clean up temporary state variables
        for key in [
            "_rinit_func_bytes",
            "_rinit_func_name",
            "_rinit_module",
            "_rproc_func_bytes",
            "_rproc_func_name",
            "_rproc_dt",
            "_rproc_nstep",
            "_rproc_accumvars",
            "_rproc_module",
            "_dmeas_func_bytes",
            "_dmeas_func_name",
            "_dmeas_module",
            "_rmeas_func_bytes",
            "_rmeas_func_name",
            "_rmeas_module",
            "_fresh_key_data",
        ]:
            if key in self.__dict__:
                del self.__dict__[key]
