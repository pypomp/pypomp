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
import time
from typing import Callable, Any, cast
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import warnings
from typing import Union, overload, Literal
from .viz import plot_traces_internal, plot_simulations_internal

from pypomp.types import ParamDict
from .metadata import ModelMetadata
from pypomp import functional as F
from .model_struct import _RInit, _RProc, _DMeas, _RMeas
import xarray as xr
from .algorithms.helpers import _calc_ys_covars, run_jax_batch_sharded
from .rw_sigma import RWSigma
from .learning_rate import LearningRate
from .par_trans import ParTrans
from .optimizer import Optimizer, Adam
from .results import (
    ResultsHistory,
    PompPFilterResult,
    PompMIFResult,
    PompTrainResult,
)
from .parameters import PompParameters
from pypomp.maths import logmeanexp
from pypomp import benchmarks
from pypomp.functional.structs import PompStruct


class Pomp:
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

    @staticmethod
    def sample_params(
        param_bounds: dict[str, tuple[float, float]], n: int, key: jax.Array
    ) -> PompParameters:
        """Sample ``n`` parameter sets uniformly within specified bounds.

        Generates random parameter vectors from independent uniform
        distributions.  Commonly used to create diverse starting points for
        global optimization (e.g. before running :meth:`mif` in parallel).

        Parameters
        ----------
        param_bounds : dict
            Dictionary mapping parameter names to ``(lower, upper)`` bound
            tuples.
        n : int
            Number of parameter sets to sample.
        key : jax.Array
            JAX random key for reproducibility.

        Returns
        -------
        PompParameters
            A :class:`~pypomp.PompParameters` object with ``n`` parameter
            rows drawn uniformly from ``param_bounds``.

        Examples
        --------
        >>> import jax
        >>> import pypomp as pp
        >>> bounds = {"beta": (0.1, 1.0), "gamma": (0.05, 0.5)}
        >>> theta = pp.Pomp.sample_params(bounds, n=20, key=jax.random.key(0))
        >>> theta.num_replicates()
        20
        """
        param_names = list(param_bounds.keys())
        low = jnp.array([param_bounds[p][0] for p in param_names])
        high = jnp.array([param_bounds[p][1] for p in param_names])

        sampled = jax.random.uniform(
            key, shape=(n, len(param_names)), minval=low, maxval=high
        )

        da = xr.DataArray(
            np.expand_dims(np.array(sampled), axis=1),
            dims=["theta_idx", "unit", "parameter"],
            coords={
                "theta_idx": np.arange(n),
                "unit": ["shared"],
                "parameter": param_names,
            },
        )
        return PompParameters(da)

    def print_metadata(self) -> None:
        """Display environment and version metadata for this model instance.

        Prints the creation timestamp and the versions of key dependencies
        (pypomp, JAX, etc.) captured when this :class:`Pomp` object
        was constructed.  Useful for reproducibility and debugging.
        """
        self.metadata.print_metadata()

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
        reps: int = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        track_time: bool = True,
    ) -> None:
        """Evaluate the log-likelihood via the bootstrap particle filter.

        Propagates a swarm of ``J`` particles through the latent state space
        using Sequential Monte Carlo (bootstrap filter) to estimate the
        marginal log-likelihood of the observed data.  Optionally computes
        conditional log-likelihoods, effective sample size, filtered means,
        and prediction means.

        JAX vectorises the computation across all parameter sets in ``theta``
        simultaneously.

        Parameters
        ----------
        J : int
            Number of particles.
        key : jax.Array or None, optional
            JAX random key.  Defaults to :attr:`fresh_key`.
        theta : PompParameters or None, optional
            Parameter set(s) to evaluate.  Defaults to :attr:`theta`.
        thresh : float, optional
            ESS-based resampling threshold in the interval :math:`[0, 1]`.
            Defaults to ``0.0`` (resample at every step).
        reps : int, optional
            Number of independent filter replicates per parameter set.
            Defaults to ``1``.
        CLL : bool, optional
            Whether to compute and store conditional log-likelihoods at
            each observation time.  Defaults to ``False``.
        ESS : bool, optional
            Whether to compute and store the effective sample size at each
            observation time.  Defaults to ``False``.
        filter_mean : bool, optional
            Whether to compute and store the filtered state mean at each
            observation time.  Defaults to ``False``.
        prediction_mean : bool, optional
            Whether to compute and store the predicted state mean at each
            observation time.  Defaults to ``False``.
        track_time : bool, optional
            Whether to record wall-clock execution time.  Defaults to
            ``True``.

        Returns
        -------
        None
            A :class:`~pypomp.core.results.PompPFilterResult` is appended
            to :attr:`results_history`.  Retrieve a dataframe summary with
            :meth:`results` or the log-likelihoods directly via ``model.theta.logLik``.

        See Also
        --------
        pypomp.functional.pfilter : Pure-functional JAX particle filter.

        Examples
        --------
        >>> model.fresh_key = jax.random.key(0)
        >>> model.pfilter(J=1000)
        >>> model.results()  # DataFrame with logLik and parameter columns
        """
        start_time = time.time()
        thresh = float(max(0.0, thresh))

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_for_results = deepcopy(theta_obj_in)
        new_key, old_key = self._update_fresh_key(key)
        n_theta_reps = theta_obj_in.num_replicates()

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        thetas_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        rep_keys = jax.random.split(new_key, n_theta_reps * reps).reshape(
            n_theta_reps, reps, *new_key.shape
        )

        results_jax = run_jax_batch_sharded(
            F.pfilter,
            {1: 0, 3: 0},
            {"logLik": 0, "CLL": 0, "ESS": 0, "filter_mean": 0, "prediction_mean": 0},
            self.to_struct(),
            thetas_array,
            J,
            rep_keys,
            thresh,
            CLL,
            ESS,
            filter_mean,
            prediction_mean,
        )

        results = jax.device_get(results_jax)

        del results_jax

        logLiks = results["logLik"]
        logLik_da = xr.DataArray(logLiks, dims=["theta_idx", "rep"])

        if track_time is True:
            execution_time = time.time() - start_time
        else:
            execution_time = None

        CLL_da = None
        ESS_da = None
        filter_mean_da = None
        prediction_mean_da = None

        if CLL and "CLL" in results:
            CLL_da = xr.DataArray(
                results["CLL"],
                dims=["theta_idx", "rep", "time"],
                coords={"time": self.ys.index},
            )

        if ESS and "ESS" in results:
            ESS_da = xr.DataArray(
                results["ESS"],
                dims=["theta_idx", "rep", "time"],
                coords={"time": self.ys.index},
            )

        if filter_mean and "filter_mean" in results:
            filter_mean_da = xr.DataArray(
                results["filter_mean"],
                dims=["theta_idx", "rep", "time", "state"],
                coords={"time": self.ys.index},
            )

        if prediction_mean and "prediction_mean" in results:
            prediction_mean_da = xr.DataArray(
                results["prediction_mean"],
                dims=["theta_idx", "rep", "time", "state"],
                coords={"time": self.ys.index},
            )

        del results

        logLik_estimates = logmeanexp(logLiks, axis=-1, ignore_nan=False)
        theta_obj_in.logLik = logLik_estimates
        self.theta = theta_obj_in

        result = PompPFilterResult(
            method="pfilter",
            execution_time=execution_time,
            key=old_key,
            theta=theta_for_results,
            logLiks=logLik_da,
            J=J,
            reps=reps,
            thresh=thresh,
            CLL_da=CLL_da,
            ESS_da=ESS_da,
            filter_mean=filter_mean_da,
            prediction_mean=prediction_mean_da,
        )

        self.results_history.add(result)

    def mif(
        self,
        J: int,
        M: int,
        rw_sd: RWSigma,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
        n_monitors: int = 0,
        track_time: bool = True,
    ) -> None:
        """Estimate parameters via the Iterated Filtering 2 (IF2) algorithm.

        Maximizes the marginal log-likelihood via the Iterated Filtering 2 (IF2)
        algorithm (Ionides et al. 2015 [1]_) by perturbing parameters with
        random walks that shrink (cool) over ``M`` iterations.  Each
        iteration runs a bootstrap particle filter with the perturbed
        parameter swarm, then records the mean parameter values as the
        estimate for that iteration.

        JAX vectorises the computation across all starting parameter sets
        in ``theta`` simultaneously.

        Parameters
        ----------
        J : int
            Number of particles.
        M : int
            Number of IF2 iterations.
        rw_sd : RWSigma
            Random walk standard deviation configuration, including per-
            parameter sigmas and a cooling schedule.  See
            :class:`~pypomp.RWSigma`.
        key : jax.Array or None, optional
            JAX random key.  Defaults to :attr:`fresh_key`.
        theta : PompParameters or None, optional
            Starting parameter set(s).  Defaults to :attr:`theta`.
        thresh : float, optional
            ESS-based resampling threshold in the interval :math:`[0, 1]`.
            Defaults to ``0.0``.
        n_monitors : int, optional
            Number of unperturbed particle filter runs to average for the
            log-likelihood monitor at each iteration.  Defaults to ``0``
            (uses the log-likelihood from the perturbed filter directly).
        track_time : bool, optional
            Whether to record wall-clock execution time.  Defaults to
            ``True``.

        Returns
        -------
        None
            A :class:`~pypomp.core.results.PompMIFResult` is appended to
            :attr:`results_history`, containing the log-likelihood monitor,
            parameter traces over iterations, and algorithm settings.

        See Also
        --------
        pypomp.functional.mif : Pure-functional JAX IF2.

        References
        ----------
        .. [1] Ionides, Edward L., Dao Nguyen, Yves Atchadé, Stilian Stoev, and Aaron A. King.
           "Inference for dynamic and latent variable models via iterated, perturbed Bayes maps."
           *Proceedings of the National Academy of Sciences* 112, no. 3 (2015): 719–724.
           https://doi.org/10.1073/pnas.1410597112.

        Examples
        --------
        >>> rw = pp.RWSigma({"beta": 0.02, "gamma": 0.01}).geometric_cooling(0.5)
        >>> model.fresh_key = jax.random.key(0)
        >>> model.mif(J=1000, M=50, rw_sd=rw)
        >>> model.traces()  # DataFrame with logLik and parameter traces
        """
        start_time = time.time()
        thresh = float(max(0.0, thresh))

        rw_param_names = list(rw_sd.all_names)
        if set(rw_param_names) != set(self.canonical_param_names):
            raise ValueError(
                "rw_sd.sigmas keys must match canonical_param_names up to reordering. "
                f"Got {sorted(rw_param_names)}, expected {sorted(self.canonical_param_names)}."
            )

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)

        new_key, old_key = self._update_fresh_key(key)
        n_reps = theta_obj_in.num_replicates()
        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.canonical_param_names
        )
        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0.")

        keys = jax.random.split(new_key, n_reps)

        theta_array_3d = jnp.repeat(theta_array[:, jnp.newaxis, :], J, axis=1)

        nLLs_jax, theta_traces_jax, final_swarm_jax = run_jax_batch_sharded(
            F.mif,
            {1: 0, 7: 0},
            [0, 0, 0],
            self.to_struct(),
            theta_array_3d,
            sigmas_array,
            sigmas_init_array,
            M,
            rw_sd.cooling_fn,
            J,
            keys,
            thresh,
            n_monitors,
        )

        nLLs = jax.device_get(nLLs_jax)
        theta_traces = jax.device_get(theta_traces_jax)

        del nLLs_jax, theta_traces_jax, final_swarm_jax

        param_names = self.canonical_param_names
        trace_vars = ["logLik"] + param_names

        # Prepend nan for the log-likelihood of the initial parameters (at iteration 0)
        nans = np.full((n_reps, 1), np.nan)
        logliks_with_nan = np.concatenate([nans, -nLLs], axis=1)  # shape: (n_reps, M+1)

        trace_data = np.concatenate(
            [logliks_with_nan[:, :, np.newaxis], theta_traces], axis=-1
        )

        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        final_thetas_mean = theta_traces[:, M, :]  # shape: (n_reps, n_params)

        final_theta_da = xr.DataArray(
            final_thetas_mean,
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_reps),
                "parameter": param_names,
            },
        )
        self.theta = PompParameters(final_theta_da, logLik=-nLLs)

        if track_time is True:
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompMIFResult(
            method="mif",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            J=J,
            M=M,
            rw_sd=rw_sd,
            thresh=thresh,
            n_monitors=n_monitors,
        )

        self.results_history.add(result)

    def train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.97,
        thresh: float = 0.0,
        alpha_cooling: float = 1.0,
        n_monitors: int = 1,
        track_time: bool = True,
    ) -> None:
        """Optimize parameters via a differentiable particle filter (MOP).

        Performs Maximum Likelihood Estimation using the Measurement Off-Parameter (MOP) particle filter (Tan et al. 2024 [1]_), treating the particle filter
        as a differentiable computation graph and applies gradient-based
        optimizers (e.g. Adam, SGD, Newton) via JAX reverse-mode
        automatic differentiation.

        .. warning::

            MOP gradients are only well-defined for **continuous-state**
            models.  For discrete-state models, use :meth:`mif` or
            :meth:`dpop_train` instead.

        JAX vectorises the computation across all starting parameter sets
        in ``theta`` simultaneously.  Results are appended to
        :attr:`results_history`.

        Parameters
        ----------
        J : int
            Number of particles used to estimate the MOP objective and
            its gradient.
        M : int
            Number of gradient steps to perform.
        eta : LearningRate
            Per-parameter learning rate schedules.  See
            :class:`~pypomp.LearningRate`.
        key : jax.Array or None, optional
            JAX random key.  Defaults to :attr:`fresh_key`.
        theta : PompParameters or None, optional
            Starting parameter set(s).  Defaults to :attr:`theta`.
        optimizer : Optimizer, optional
            Optimizer configuration object (e.g. :class:`~pypomp.Adam`,
            :class:`~pypomp.SGD`, :class:`~pypomp.Newton`).  Defaults to
            :class:`~pypomp.Adam`.
        alpha : float, optional
            MOP discount factor controlling the bias-variance trade-off.
            Defaults to ``0.97``.
        thresh : float, optional
            ESS-based resampling threshold.  Defaults to ``0.0``.
        alpha_cooling : float, optional
            Cosine cooling multiplier for ``alpha``.  At the end of
            training, ``alpha`` is moved ``alpha_cooling`` of the way from
            its initial value toward ``1.0``.  Defaults to ``1.0`` (no
            cooling).
        n_monitors : int, optional
            Number of unperturbed particle filter runs to average for the
            log-likelihood monitor.  Defaults to ``1``. Using more than 1 monitor
            increases computation time but can lead to more stable estimates.
        track_time : bool, optional
            Whether to record wall-clock execution time.  Defaults to
            ``True``.

        Returns
        -------
        None
            A :class:`~pypomp.core.results.PompTrainResult` is appended
            to :attr:`results_history`, containing log-likelihood and
            parameter traces over iterations.

        See Also
        --------
        pypomp.functional.train : Pure-functional JAX gradient training.

        References
        ----------
        .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
           for Partially Observed Markov Processes using Automatic Differentiation."
           *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.

        Examples
        --------
        >>> eta = pp.LearningRate({"beta": 0.01, "gamma": 0.005})
        >>> model.fresh_key = jax.random.key(0)
        >>> model.train(J=100, M=200, eta=eta)
        >>> model.results()
        """
        start_time = time.time()
        thresh = float(max(0.0, thresh))

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)

        n_reps = theta_obj_in.num_replicates()

        theta_obj_in.transform(self.par_trans, direction="to_est")
        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0")

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        # Convert eta to JAX array in canonical order
        eta_array = eta.to_array(self.canonical_param_names, M)

        new_key, old_key = self._update_fresh_key(key)
        keys = jnp.array(jax.random.split(new_key, n_reps))

        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        nLLs, theta_ests = run_jax_batch_sharded(
            F.train,
            {1: 0, 7: 0},
            [0, 0],
            self.to_struct(),
            theta_array,
            J,
            optimizer,
            M,
            eta_array,
            alpha,
            keys,
            alpha_cooling,
            thresh,
            n_monitors,
        )

        theta_ests_natural = self.par_trans._transform_array(
            np.asarray(theta_ests),
            self.canonical_param_names,
            direction="from_est",
        )

        joined_array = xr.DataArray(
            np.concatenate(
                [
                    -nLLs[..., np.newaxis],  # shape: (theta_idx, iteration, 1)
                    theta_ests_natural,  # shape: (theta_idx, iteration, n_theta)
                ],
                axis=-1,
            ),
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": range(0, n_reps),
                "iteration": range(0, M + 1),
                "variable": ["logLik"] + self.canonical_param_names,
            },
        )

        final_theta_da = xr.DataArray(
            theta_ests_natural[:, -1, :],
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_reps),
                "parameter": self.canonical_param_names,
            },
        )
        self.theta = PompParameters(final_theta_da, logLik=np.asarray(-nLLs))

        if track_time is True:
            nLLs.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompTrainResult(
            method="train",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=joined_array,
            optimizer=optimizer,
            J=J,
            M=M,
            eta=eta,
            alpha=alpha,
            thresh=thresh,
            alpha_cooling=alpha_cooling,
        )

        self.results_history.add(result)

    def dpop_train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.8,
        alpha_cooling: float = 1.0,
        decay: float = 0.0,
        process_weight_state: str | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Optimizes model parameters using the DPOP differentiable particle filter and gradient-based methods.

        .. warning::
            This method is experimental. Its API and behavior are subject to change in future releases.

        This method trains the model parameters to maximize the DPOP objective function using
        first-order optimizers like Adam or SGD, with optional learning rate decay. Gradients
        are computed efficiently via JAX reverse-mode automatic differentiation.

        Parameters
        ----------
        J : int
            Number of particles.
        M : int
            Number of gradient steps.
        eta : LearningRate
            Per-parameter learning rates as a LearningRate object.
        optimizer : Optimizer, default Adam()
            Optimizer configuration object, e.g. ``Adam()`` or ``SGD()``.
        alpha : float, default 0.8
            DPOP discount / cooling factor.
        alpha_cooling : float, default 1.0
            Cosine cooling factor for alpha. This factor represents the
            multiplier for the distance of alpha from 1.0 by the end of
            training. The default keeps alpha fixed.
        decay : float, default 0.0
            Learning-rate decay coefficient. At iteration m, the effective
            learning rate is ``eta / (1 + decay * m)``.
        process_weight_state : str or None, default None
            Name of the state component that stores the accumulated
            process log-weight (e.g. ``"logw"``).
        key : jax.Array or None, default None
            Random key. If None, uses ``self.fresh_key``.
        theta : PompParameters, default None
            Optional initial parameter(s). Defaults to self.theta.

        Returns
        -------
        nll_history : jax.Array, shape (M+1,)
            Mean DPOP negative log-likelihood per observation at each step.
        theta_history : jax.Array, shape (M+1, p)
            Parameter vector (estimation space) at each step.
        """
        warnings.warn(
            "dpop_train is experimental and its API and behavior are subject to change.",
            category=FutureWarning,
            stacklevel=2,
        )

        from .algorithms.train_dpop import dpop_train as _dpop_train

        new_key, _ = self._update_fresh_key(key)
        theta_obj = self._prepare_theta_input(theta)
        theta_nat = theta_obj.params(as_list=True)[0]
        param_names = self.canonical_param_names
        theta_est_dict = self.par_trans.to_est(cast(ParamDict, theta_nat))
        theta_init = jnp.array([theta_est_dict[name] for name in param_names])

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        # Full (M, p) per-iteration LR schedule (e.g. from
        # LearningRate(...).cosine_decay(...)); the kernel indexes row m.
        eta_array = eta.to_array(param_names, M)

        ys_array = jnp.array(self.ys.values)
        dt_array_extended = self._dt_array_extended
        nstep_array = self._nstep_array
        t0 = self.t0
        times_array = jnp.array(self.ys.index.values)

        rinitializer = self.rinit.struct_pf
        rprocess_interp = self.rproc.struct_pf_interp

        if self.dmeas is None:
            raise ValueError("dpop_train requires self.dmeas to be not None.")
        dmeasure = self.dmeas.struct_pf

        accumvars = self.rproc.accumvars
        covars_extended = self._covars_extended

        if process_weight_state is None:
            raise ValueError(
                "dpop_train requires a process-weight state. "
                "Please provide `process_weight_state` as the name of the "
                "state variable that accumulates the transition log-weight "
                "(e.g. 'logw')."
            )

        try:
            process_weight_index = int(self.statenames.index(process_weight_state))
        except ValueError as e:
            raise ValueError(
                f"State '{process_weight_state}' not found in statenames "
                f"{self.statenames}"
            ) from e

        ntimes = len(self.ys)
        opt_name = optimizer.__class__.__name__
        beta1 = getattr(optimizer, "beta1", 0.9)
        beta2 = getattr(optimizer, "beta2", 0.999)
        epsilon = getattr(optimizer, "epsilon", 1e-8)
        theta_hist, nll_hist = _dpop_train(
            theta_init=theta_init,
            ys=ys_array,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            t0=t0,
            times=times_array,
            J=J,
            rinitializer=rinitializer,
            rprocess_interp=rprocess_interp,
            dmeasure=dmeasure,
            accumvars=accumvars,
            covars_extended=covars_extended,
            alpha=alpha,
            process_weight_index=process_weight_index,
            ntimes=ntimes,
            key=new_key,
            M=M,
            eta=eta_array,
            optimizer=opt_name,
            decay=decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            alpha_cooling=alpha_cooling,
        )

        return nll_hist, theta_hist

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        *,
        as_pomp: Literal[True],
    ) -> "Pomp": ...

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: bool = False,
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], "Pomp"]:
        """Simulate latent states and observations from the POMP model.

        Propagates the latent state through time via ``rproc`` and draws
        synthetic observations from ``rmeas``.  JAX vectorises the
        computation across parameter sets and simulation replicates
        simultaneously.

        Parameters
        ----------
        key : jax.Array or None, optional
            JAX random key.  Defaults to :attr:`fresh_key`.
        theta : PompParameters or None, optional
            Parameter set(s) to simulate from.  Defaults to :attr:`theta`.
        times : jax.Array or None, optional
            Observation times at which to simulate.  Defaults to the
            original ``ys`` index.
        nsim : int, optional
            Number of independent simulation replicates per parameter set.
            Defaults to ``1``.
        as_pomp : bool, optional
            If ``True``, return a deep copy of this model with its ``ys``
            replaced by one simulation from the first parameter set.
            Overrides ``nsim`` to ``1``.  Defaults to ``False``.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame) or Pomp
            If ``as_pomp=False`` (default): a ``(states_df, obs_df)`` tuple
            of long-format DataFrames.  Each has columns ``theta_idx``,
            ``sim``, ``time``, plus one column per state/observation
            variable.

            If ``as_pomp=True``: a new :class:`Pomp` instance whose ``ys``
            contains the simulated observations for the first parameter
            replicate.

        See Also
        --------
        pypomp.functional.simulate : Pure-functional JAX simulation.

        Examples
        --------
        >>> model.fresh_key = jax.random.key(1)
        >>> states, obs = model.simulate(nsim=50)
        >>> obs.head()
        """
        if as_pomp:
            if nsim > 1:
                warnings.warn(
                    "as_pomp is True, but nsim > 1. Only 1 simulation will be performed as_pomp overrides nsim.",
                    UserWarning,
                )
            nsim = 1

        theta_obj_in = self._prepare_theta_input(theta)

        if self.rmeas is None:
            raise ValueError(
                "self.rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        thetas_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, thetas_array.shape[0])
        times_array = jnp.array(self.ys.index) if times is None else times
        X_sims, Y_sims = F.simulate(
            self.to_struct(),
            thetas_array,
            nsim,
            keys,
            times=times_array,
        )

        def _to_long(
            arr: Union[jax.Array, np.ndarray],
            times_vec: Union[jax.Array, np.ndarray, pd.Index],
            prefix: str,
        ) -> pd.DataFrame:
            vals = np.asarray(arr)  # (n_theta, n_sim, n_time, n_feat)
            n_theta_l, n_sim_l, n_time_l, n_feat_l = vals.shape
            flat = vals.reshape(n_theta_l * n_sim_l * n_time_l, n_feat_l)
            theta_idx_l = np.repeat(np.arange(n_theta_l), n_sim_l * n_time_l)
            sim_idx_l = np.tile(np.repeat(np.arange(n_sim_l), n_time_l), n_theta_l)
            time_vals_l = np.tile(
                np.asarray(times_vec).reshape(1, -1), (n_theta_l * n_sim_l, 1)
            ).reshape(-1)
            cols = pd.Index([f"{prefix}_{i}" for i in range(n_feat_l)])
            df = pd.DataFrame(flat, columns=cols)
            df.insert(0, "time", time_vals_l)
            df.insert(0, "sim", sim_idx_l)
            df.insert(0, "theta_idx", theta_idx_l)
            return df

        times0 = np.concatenate([np.array([self.t0]), np.array(times_array)])
        X_sims_long = _to_long(X_sims, times0, "state")
        Y_sims_long = _to_long(Y_sims, np.array(times_array), "obs")

        if as_pomp:
            simulated_ys_long = Y_sims_long[
                (Y_sims_long["theta_idx"] == 0) & (Y_sims_long["sim"] == 0)
            ].copy()
            simulated_ys = pd.DataFrame(
                simulated_ys_long.drop(columns=["theta_idx", "sim", "time"])
            )
            simulated_ys.index = pd.Index(simulated_ys_long["time"])
            simulated_ys.columns = self.ys.columns

            pomp_copy = deepcopy(self)
            pomp_copy.ys = simulated_ys
            pomp_copy.theta = theta_obj_in.subset([0])
            return pomp_copy

        return X_sims_long, Y_sims_long

    def probe(
        self,
        probes: dict[str, Callable[[pd.DataFrame], float]],
        nsim: int = 100,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
    ) -> pd.DataFrame:
        """Assess goodness-of-fit by comparing data probes to simulated probes.

        Computes user-supplied summary statistics ("probes") on both the
        original observed data and ``nsim`` simulated data sets.  The
        resulting DataFrame can be used to visually or formally test
        whether the model reproduces salient features of the data.

        Parameters
        ----------
        probes : dict of str to callable
            Dictionary mapping probe names to functions.  Each function
            receives a :class:`~pandas.DataFrame` of observations (with
            time as the index) and returns a scalar, e.g.
            ``{"mean": lambda df: df["cases"].mean()}``.
        nsim : int, optional
            Number of simulation replicates.  Defaults to ``100``.
        key : jax.Array or None, optional
            JAX random key.  Defaults to :attr:`fresh_key`.
        theta : PompParameters or None, optional
            Parameter set to simulate from.  Defaults to :attr:`theta`.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns ``probe``, ``value``,
            ``is_real_data``, ``theta_idx``, and ``sim``.
        """
        sim_result = self.simulate(nsim=nsim, key=key, theta=theta, as_pomp=False)
        assert isinstance(sim_result, tuple)
        _, y_sims = sim_result

        results = []

        for name, func in probes.items():
            results.append(
                {
                    "probe": name,
                    "value": float(func(self.ys)),
                    "is_real_data": True,
                    "theta_idx": pd.NA,
                    "sim": pd.NA,
                }
            )

        def apply_probes(group):
            theta_idx, sim_id = group.name
            df = pd.DataFrame(group.drop(columns=["time"]))
            df.index = pd.Index(group["time"])
            df.columns = self.ys.columns
            for name, func in probes.items():
                results.append(
                    {
                        "probe": name,
                        "value": float(func(df)),
                        "is_real_data": False,
                        "theta_idx": theta_idx,
                        "sim": sim_id,
                    }
                )

        y_sims.groupby(["theta_idx", "sim"]).apply(apply_probes, include_groups=False)  # type: ignore[call-overload]

        return pd.DataFrame(results)

    def traces(self) -> pd.DataFrame:
        """Return the full trace of log-likelihoods and parameters over all runs.

        Concatenates the parameter and log-likelihood histories from every
        :meth:`pfilter`, :meth:`mif`, and :meth:`train` call stored in
        :attr:`results_history`.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame containing concatenated trace data. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``iteration``: Counter indicating the global iteration.
            3. ``method``: The name of the method (e.g. ``'pfilter'``, ``'mif'``, ``'train'``).
            4. ``logLik``: The estimated log-likelihood.
            5. ``se``: The standard error of the log-likelihood estimate.
            6. Parameter columns: One column per model parameter in their defined order.
        """
        return self.results_history.traces()

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """Return a summary DataFrame for one run from the results history.

        Retrieves the final log-likelihoods and parameter values for all
        replicates associated with the run at position ``index`` in
        :attr:`results_history`.

        Parameters
        ----------
        index : int, optional
            Position in :attr:`results_history` to retrieve.  Defaults to
            ``-1`` (the most recent run).
        ignore_nan : bool, optional
            If ``True``, NaN log-likelihoods are excluded when computing
            the summary.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame with columns ``logLik`` and one column per
            parameter, indexed by ``theta_idx``.

        See Also
        --------
        pypomp.core.results.PompPFilterResult.to_dataframe : Dataframe returned by :class:`~pypomp.core.results.PompPFilterResult` class.
        pypomp.core.results.PompMIFResult.to_dataframe : Dataframe returned by :class:`~pypomp.core.results.PompMIFResult` class.
        pypomp.core.results.PompTrainResult.to_dataframe : Dataframe returned by :class:`~pypomp.core.results.PompTrainResult` class.
        """
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods from a particle filter run.

        Parameters
        ----------
        index : int, optional
            Position in :attr:`results_history` to retrieve.  Defaults to
            ``-1`` (the most recent run).  The indexed result must be a
            :meth:`pfilter` result with ``CLL=True``.
        average : bool, optional
            If ``True``, average the CLLs over replicates using logmeanexp.
            Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of conditional log-likelihoods. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``rep``: The replicate index (only if ``average=False``).
            3. ``time``: The observation time point.
            4. ``CLL``: The conditional log-likelihood value.
        """
        return self.results_history.CLL(index=index, average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Return effective sample sizes from a particle filter run.

        Parameters
        ----------
        index : int, optional
            Position in :attr:`results_history` to retrieve.  Defaults to
            ``-1`` (the most recent run).  The indexed result must be a
            :meth:`pfilter` result with ``ESS=True``.
        average : bool, optional
            If ``True``, average the ESS over replicates using arithmetic
            mean.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of ESS values. The columns appear in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``rep``: The replicate index (only if ``average=False``).
            3. ``time``: The observation time point.
            4. ``ESS``: The Effective Sample Size value.
        """
        return self.results_history.ESS(index=index, average=average)

    def time(self):
        """Return a summary of wall-clock execution times for all runs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``method`` (e.g. ``'pfilter'``,
            ``'mif'``) and ``time`` (execution time in seconds).
        """
        return self.results_history.time()

    def prune(self, n: int = 1, refill: bool = True):
        """Keep the top ``n`` parameter sets by log-likelihood.

        Discards poorly performing starting points after an estimation run,
        focusing subsequent work on the most promising candidates.  If
        ``refill`` is ``True``, the surviving sets are duplicated to restore
        the original number of replicates.

        Parameters
        ----------
        n : int, optional
            Number of top parameter sets to retain.  Defaults to ``1``.
        refill : bool, optional
            If ``True``, repeat the top ``n`` sets to match the previous
            number of replicates.  Defaults to ``True``.
        """
        self.theta.prune(n=n, refill=refill)

    def plot_traces(self, show: bool = True) -> Any:
        """Plot parameter and log-likelihood traces from the results history.

        Produces an interactive Plotly figure with one facet per parameter
        and one for ``logLik``.  Lines connect :meth:`mif` / :meth:`train`
        points for each replicate; :meth:`pfilter` runs appear as dots.
        Replicates are distinguished by colour.

        Parameters
        ----------
        show : bool, optional
            Whether to call ``fig.show()`` before returning.  Defaults to
            ``True``.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The Plotly figure object, or ``None`` if no results are stored.
        """
        traces = self.traces()
        fig = plot_traces_internal(traces, title="Pomp Traces")

        if fig is not None and show:
            fig.show()
        return fig

    def plot_simulations(
        self,
        key: jax.Array,
        nsim: int = 20,
        mode: str = "lines",
        theta: PompParameters | None = None,
        show: bool = True,
    ) -> Any:
        """Plot simulated trajectories alongside the observed data.

        Generates an interactive Plotly figure overlaying ``nsim`` simulated
        observation trajectories on the actual ``ys`` data, helping to
        assess qualitative goodness-of-fit.

        Parameters
        ----------
        key : jax.Array
            JAX random key for simulation.
        nsim : int, optional
            Number of simulation replicates.  Defaults to ``20``.
        mode : str, optional
            Plot mode: ``"lines"`` shows individual trajectories;
            ``"quantiles"`` shows a shaded quantile band.  Defaults to
            ``"lines"``.
        theta : PompParameters or None, optional
            Parameter set to simulate from.  Defaults to the first
            replicate of :attr:`theta`.
        show : bool, optional
            Whether to call ``fig.show()`` before returning.  Defaults to
            ``True``.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The Plotly figure object.
        """
        if theta is None:
            theta = (
                self.theta.subset([0])
                if self.theta and self.theta.num_replicates() > 1
                else self.theta
            )
        elif not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters instance")

        _, sims = self.simulate(nsim=nsim, theta=theta, key=key)
        fig = plot_simulations_internal(sims, self.ys, mode=mode)

        if fig is not None and show:
            fig.show()
        return fig

    def print_summary(self, n: int = 5):
        """Print a high-level summary of the model and its estimation history.

        Displays basic model statistics (number of observations, time steps,
        parameters, and parameter replicates) followed by a tabular summary
        of :attr:`results_history` listing each run's method and
        performance metrics.

        Parameters
        ----------
        n : int, optional
            Maximum number of history entries to display.  Defaults to
            ``5``.
        """
        print("Basics:")
        print("-------")
        print(f"Number of observations: {len(self.ys)}")
        print(f"Number of time steps: {len(self._dt_array_extended)}")
        print(f"Number of parameters: {self.theta.num_params()}")
        print(f"Number of parameter sets: {self.theta.num_replicates()}")
        print()
        self.results_history.print_summary(n=n)

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

    def arma(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        log_ys: bool = False,
        suppress_warnings: bool = True,
    ) -> float:
        """Fit an ARIMA benchmark model and return its log-likelihood.

        Fits an independent ARIMA(p, d, q) model to the observation data
        as a statistical baseline.  Wraps :func:`pypomp.benchmarks.arma`.

        Parameters
        ----------
        order : tuple of int, optional
            ``(p, d, q)`` order for the ARIMA model.  Defaults to
            ``(1, 0, 1)``.
        log_ys : bool, optional
            If ``True``, fit the model to ``log(y + 1)`` rather than the
            raw observations.  Defaults to ``False``.
        suppress_warnings : bool, optional
            If ``True``, suppress per-unit warnings from statsmodels and
            issue a single summary warning instead.  Defaults to ``True``.

        Returns
        -------
        float
            Sum of the per-unit ARIMA log-likelihoods.
        """
        return benchmarks.arma(
            self.ys, order=order, log_ys=log_ys, suppress_warnings=suppress_warnings
        )

    def negbin(
        self, autoregressive: bool = False, suppress_warnings: bool = True
    ) -> float:
        """Fit a Negative Binomial benchmark model and return its log-likelihood.

        Fits an independent (or AR(1)) Negative Binomial model to the
        observation data as a statistical baseline.  Wraps
        :func:`pypomp.benchmarks.negbin`.

        Parameters
        ----------
        autoregressive : bool, optional
            If ``True``, fit an AR(1) Negative Binomial model instead of
            the i.i.d. version.  Defaults to ``False``.
        suppress_warnings : bool, optional
            If ``True``, suppress per-unit warnings and issue a single
            summary warning instead.  Defaults to ``True``.

        Returns
        -------
        float
            Sum of the per-unit Negative Binomial log-likelihoods.
        """
        return benchmarks.negbin(
            self.ys,
            autoregressive=autoregressive,
            suppress_warnings=suppress_warnings,
        )
