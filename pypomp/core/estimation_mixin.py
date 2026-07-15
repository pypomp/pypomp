from __future__ import annotations
import time
import warnings
from typing import Callable, Any, cast, overload, Union, Literal
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from pypomp.types import ParamDict
from pypomp import functional as F
from pypomp.maths import logmeanexp
from pypomp import benchmarks
from .algorithms.helpers import run_jax_batch_sharded
from .rw_sigma import RWSigma
from .learning_rate import LearningRate
from .optimizer import Optimizer, Adam
from .results import (
    PompPFilterResult,
    PompMIFResult,
    PompTrainResult,
    PompPMCMCResult,
    PompABCResult,
)
from pypomp.proposals import _expand_proposal
from .parameters import PompParameters

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import PompInterface as Base
    from .pomp import Pomp
else:
    Base = object


def _flat_dprior(theta_arr: jax.Array) -> jax.Array:
    """Default flat improper log-prior."""
    return jnp.zeros((), dtype=theta_arr.dtype)


class PompEstimationMixin(Base):
    """
    Mixin class that implements estimation, simulation, and benchmark methods for Pomp.
    """

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

        theta_obj_in = theta_obj_in.transformed(self.par_trans, direction="to_est")
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

    def pmcmc(
        self,
        J: int,
        Nmcmc: int,
        proposal,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
        track_time: bool = True,
    ) -> None:
        """
        Particle Markov chain Monte Carlo (PMMH) for Bayesian parameter inference.

        Runs one independent PMCMC chain for each parameter replicate in ``theta``.
        Each chain uses a bootstrap particle filter likelihood estimate inside a
        Metropolis-Hastings update. Results are stored in
        :attr:`Pomp.results_history`.

        Args:
            J: Number of particles per particle-filter likelihood evaluation.
            Nmcmc: Number of MCMC iterations per chain.
            proposal: Proposal object from :mod:`pypomp.proposals`.
            dprior: Pure-JAX log-prior function with signature
                ``dprior(theta_arr) -> scalar``. If ``None``, a flat improper
                prior is used.
            key: JAX PRNG key. Defaults to :attr:`fresh_key`.
            theta: Starting parameter values. Defaults to :attr:`theta`.
            thresh: Adaptive resampling threshold passed to the particle filter.
            track_time: Whether to record execution time.

        Returns:
            None. Updates :attr:`Pomp.results_history` with a
            :class:`~pypomp.core.results.PompPMCMCResult`.
        """
        start_time = time.time()

        if self.dmeas is None:
            raise ValueError("pmcmc requires self.dmeas to be not None.")
        if J < 1:
            raise ValueError("J must be >= 1.")
        if Nmcmc < 1:
            raise ValueError("Nmcmc must be >= 1.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_chains = theta_obj_in.num_replicates()
        if n_chains < 1:
            raise ValueError("pmcmc requires at least one starting parameter set.")

        new_key, old_key = self._update_fresh_key(key)
        canonical_names = self.canonical_param_names
        theta_array = theta_obj_in.to_jax_array(canonical_names)
        proposal = _expand_proposal(proposal, canonical_names)
        log_prior = dprior if dprior is not None else _flat_dprior
        keys = jax.random.split(new_key, n_chains)

        ll_jax, lp_jax, theta_jax, accepts_jax = F.pmcmc(
            self.to_struct(),
            theta_array,
            proposal,
            log_prior,
            Nmcmc,
            J,
            thresh,
            keys,
        )

        ll_traces, lp_traces, theta_traces, accepts = jax.device_get(
            (ll_jax, lp_jax, theta_jax, accepts_jax)
        )

        trace_vars = ["logLik", "log_prior"] + list(canonical_names)
        trace_data = np.concatenate(
            [ll_traces[..., np.newaxis], lp_traces[..., np.newaxis], theta_traces],
            axis=-1,
        )
        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_chains),
                "iteration": np.arange(Nmcmc + 1),
                "variable": trace_vars,
            },
        )

        final_theta_da = xr.DataArray(
            theta_traces[:, -1, :],
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_chains),
                "parameter": canonical_names,
            },
        )
        self.theta = PompParameters(final_theta_da, logLik=ll_traces[:, -1])

        execution_time = time.time() - start_time if track_time else None
        result = PompPMCMCResult(
            method="pmcmc",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            Nmcmc=Nmcmc,
            J=J,
            accepts=np.asarray(accepts, dtype=np.int32),
        )
        self.results_history.add(result)

    def abc(
        self,
        Nabc: int,
        probes: dict[str, Callable],
        scale: dict[str, float],
        epsilon: float,
        proposal,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        track_time: bool = True,
    ) -> None:
        """
        Approximate Bayesian Computation with a Metropolis-Hastings outer loop.

        The probe functions must be pure JAX callables accepting a simulated
        observation array with shape ``(n_obs, ydim)`` and returning a scalar.
        One independent ABC-MCMC chain is run for each parameter replicate in
        ``theta``. Results are stored in :attr:`Pomp.results_history`.

        Args:
            Nabc: Number of ABC-MCMC iterations per chain.
            probes: Mapping from probe name to pure-JAX summary statistic.
            scale: Positive scaling factor for each probe.
            epsilon: ABC distance threshold.
            proposal: Proposal object from :mod:`pypomp.proposals`.
            dprior: Pure-JAX log-prior function. If ``None``, a flat improper
                prior is used.
            key: JAX PRNG key. Defaults to :attr:`fresh_key`.
            theta: Starting parameter values. Defaults to :attr:`theta`.
            track_time: Whether to record execution time.

        Returns:
            None. Updates :attr:`Pomp.results_history` with a
            :class:`~pypomp.core.results.PompABCResult`.
        """
        start_time = time.time()

        if self.rmeas is None:
            raise ValueError("abc requires self.rmeas to be not None.")
        if Nabc < 1:
            raise ValueError("Nabc must be >= 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not probes:
            raise ValueError("probes must be a non-empty dict.")
        if set(scale.keys()) != set(probes.keys()):
            raise ValueError("scale keys must match probes keys.")
        for name, value in scale.items():
            if value <= 0:
                raise ValueError(f"scale['{name}'] must be positive.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_chains = theta_obj_in.num_replicates()
        if n_chains < 1:
            raise ValueError("abc requires at least one starting parameter set.")

        new_key, old_key = self._update_fresh_key(key)
        canonical_names = self.canonical_param_names
        theta_array = theta_obj_in.to_jax_array(canonical_names)
        proposal = _expand_proposal(proposal, canonical_names)
        log_prior = dprior if dprior is not None else _flat_dprior

        probe_names = sorted(probes.keys())
        scale_arr = jnp.asarray([float(scale[name]) for name in probe_names])

        def probe_fn(y_arr: jax.Array) -> jax.Array:
            return jnp.stack(
                [jnp.asarray(probes[name](y_arr)).reshape(()) for name in probe_names]
            )

        obs_probes = probe_fn(jnp.asarray(self.ys.values))
        keys = jax.random.split(new_key, n_chains)
        ydim = int(self.ys.shape[1])

        dist_jax, lp_jax, theta_jax, accepts_jax = F.abc(
            self.to_struct(),
            theta_array,
            proposal,
            log_prior,
            probe_fn,
            obs_probes,
            scale_arr,
            float(epsilon),
            ydim,
            Nabc,
            keys,
        )

        dist_traces, lp_traces, theta_traces, accepts = jax.device_get(
            (dist_jax, lp_jax, theta_jax, accepts_jax)
        )

        trace_vars = ["distance", "log_prior"] + list(canonical_names)
        trace_data = np.concatenate(
            [
                dist_traces[..., np.newaxis],
                lp_traces[..., np.newaxis],
                theta_traces,
            ],
            axis=-1,
        )
        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_chains),
                "iteration": np.arange(Nabc + 1),
                "variable": trace_vars,
            },
        )

        final_theta_da = xr.DataArray(
            theta_traces[:, -1, :],
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_chains),
                "parameter": canonical_names,
            },
        )
        self.theta = PompParameters(final_theta_da)

        execution_time = time.time() - start_time if track_time else None
        result = PompABCResult(
            method="abc",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            Nabc=Nabc,
            epsilon=float(epsilon),
            accepts=np.asarray(accepts, dtype=np.int32),
        )
        self.results_history.add(result)

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
    ) -> Pomp: ...

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: bool = False,
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], Pomp]:
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
            return cast(Any, pomp_copy)

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
