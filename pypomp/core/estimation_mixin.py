from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from pypomp import benchmarks
from pypomp import functional as F
from pypomp.functional.abc import abc
from pypomp.functional.dpop import dpop_train
from pypomp.functional.pmcmc import pmcmc
from pypomp.maths import logmeanexp
from pypomp.proposals import Proposal

from .algorithms.helpers import run_jax_batch_sharded
from .learning_rate import LearningRate
from .optimizer import Adam, Optimizer
from .parameters import PompParameters
from .results import (
    build_abc_result,
    build_dpop_train_result,
    build_mif_result,
    build_pfilter_result,
    build_pmcmc_result,
    build_train_result,
)
from .rw_sigma import RWSigma

if TYPE_CHECKING:
    from .interfaces import PompInterface as Base
    from .pomp import Pomp
else:
    Base = object


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
        *,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
        reps: int = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
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

        Returns
        -------
        None
            A :class:`~pypomp.core.results.Result` is appended
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

        execution_time = time.time() - start_time

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

        result = build_pfilter_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_for_results,
            logLiks=logLik_da,
            J=J,
            reps=reps,
            thresh=thresh,
            CLL=CLL_da,
            ESS=ESS_da,
            filter_mean=filter_mean_da,
            prediction_mean=prediction_mean_da,
        )

        self.results_history.add(result)

    def mif(
        self,
        J: int,
        M: int,
        rw_sd: RWSigma,
        *,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
        n_monitors: int = 0,
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

        Returns
        -------
        None
            A :class:`~pypomp.core.results.Result` is appended to
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

        rw_param_names = list(rw_sd.param_names)
        if set(rw_param_names) != set(self.canonical_param_names):
            raise ValueError(
                "rw_sd.sigmas keys must match canonical_param_names up to reordering. "
                f"Got {sorted(rw_param_names)}, expected {sorted(self.canonical_param_names)}."
            )

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)

        new_key, old_key = self._update_fresh_key(key)
        n_reps = theta_obj_in.num_replicates()
        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0.")

        keys = jax.random.split(new_key, n_reps)

        theta_array_3d = jnp.repeat(theta_array[:, jnp.newaxis, :], J, axis=1)

        logliks_jax, theta_traces_jax, final_swarm_jax = run_jax_batch_sharded(
            F.mif,
            {1: 0, 5: 0},
            [0, 0, 0],
            self.to_struct(),
            theta_array_3d,
            J,
            M,
            rw_sd,
            keys,
            thresh,
            n_monitors,
        )

        logliks = jax.device_get(logliks_jax)
        theta_traces = jax.device_get(theta_traces_jax)

        del logliks_jax, theta_traces_jax, final_swarm_jax

        param_names = self.canonical_param_names
        trace_vars = ["logLik"] + param_names

        # Prepend nan for the log-likelihood of the initial parameters (at iteration 0)
        nans = np.full((n_reps, 1), np.nan)
        logliks_with_nan = np.concatenate(
            [nans, logliks], axis=1
        )  # shape: (n_reps, M+1)

        traces_da = xr.DataArray(
            np.concatenate(
                [
                    logliks_with_nan[:, :, np.newaxis],
                    theta_traces,
                ],
                axis=-1,
            ),
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        self.theta = PompParameters(
            xr.DataArray(
                theta_traces[:, -1, :],
                dims=["theta_idx", "parameter"],
                coords={
                    "theta_idx": np.arange(n_reps),
                    "parameter": param_names,
                },
            ),
            logLik=logliks[:, -1],
        )

        execution_time = time.time() - start_time

        result = build_mif_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces=traces_da,
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
        *,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        optimizer: Optimizer | None = None,
        alpha: float = 0.97,
        thresh: float = 0.0,
        alpha_cooling: float = 1.0,
        n_monitors: int = 1,
    ) -> None:
        """Optimize parameters via a differentiable particle filter (MOP).

        Performs Maximum Likelihood Estimation using the Measurement Off-Parameter (MOP) particle filter (Tan et al. 2024 [1]_), treating the particle filter
        as a differentiable computation graph and applies gradient-based
        optimizers (e.g. Adam, SGD, Newton) via JAX reverse-mode
        automatic differentiation.

        .. warning::

            MOP gradients are only well-defined for **continuous-state**
            models.  For discrete-state models, use :meth:`mif` or
            :meth:`_dpop_train` (experimental) instead.

        .. note::

            Training requires the number of integration steps between
            consecutive observations to be constant across all intervals.
            Setting `nstep` ensures this, but `dt` can also yield constant steps.

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

        Returns
        -------
        None
            A :class:`~pypomp.core.results.Result` is appended
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
        optimizer = optimizer or Adam()
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

        new_key, old_key = self._update_fresh_key(key)
        keys = jnp.array(jax.random.split(new_key, n_reps))

        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        nLLs_jax, theta_ests_jax = run_jax_batch_sharded(
            F.train,
            {1: 0, 5: 0},
            [0, 0],
            self.to_struct(),
            theta_array,
            J,
            M,
            eta,
            keys,
            optimizer,
            alpha,
            alpha_cooling,
            thresh,
            n_monitors,
        )

        nLLs, theta_ests = jax.device_get((nLLs_jax, theta_ests_jax))
        del nLLs_jax, theta_ests_jax

        theta_ests_natural = self.par_trans._transform_array(
            theta_ests,
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
                "theta_idx": range(n_reps),
                "iteration": range(M + 1),
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

        execution_time = time.time() - start_time

        result = build_train_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces=joined_array,
            optimizer=optimizer,
            J=J,
            M=M,
            eta=eta,
            alpha=alpha,
            thresh=thresh,
            alpha_cooling=alpha_cooling,
        )

        self.results_history.add(result)

    def _dpop_train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        *,
        optimizer: Optimizer | None = None,
        alpha: float = 0.8,
        alpha_cooling: float = 1.0,
        process_weight_state: str | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
    ) -> None:
        """
        Optimizes model parameters using the DPOP differentiable particle filter and gradient-based methods.

        .. warning::
            This method is experimental. Its API and behavior are subject to change in future releases.

        This method is analogous to :meth:`train` as an optimization algorithm
        for parameter estimation, but it can handle continuous states.
        It additionally incorporates a per-interval transition log-weight that
        is assumed to be stored in one of the state components.

        The process log-weight is expected to be accumulated over a single
        observation interval by the user-specified process model.  At the
        beginning of each interval, the corresponding state component should be
        reset to zero (this is naturally handled by ``accumvars``).

        .. note::

            Training requires the number of integration steps between
            consecutive observations to be constant across all intervals.
            Setting `nstep` ensures this, but `dt` can also yield constant steps.

        This method trains the model parameters to maximize the DPOP objective function using
        first-order optimizers like Adam or SGD. Gradients
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
        process_weight_state : str or None, default None
            Name of the state component that stores the accumulated
            process log-weight (e.g. ``"logw"``).
        key : jax.Array or None, default None
            Random key. If None, uses ``self.fresh_key``.
        theta : PompParameters, default None
            Optional initial parameter(s). Defaults to self.theta.

        Returns
        -------
        None
            A :class:`~pypomp.core.results.Result` is appended
            to :attr:`results_history`, containing log-likelihood and
            parameter traces over iterations.
        """
        warnings.warn(
            "dpop_train is experimental and its API and behavior are subject to change.",
            category=FutureWarning,
            stacklevel=2,
        )

        start_time = time.time()
        optimizer = optimizer or Adam()

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)

        n_reps = theta_obj_in.num_replicates()

        theta_obj_in = theta_obj_in.transformed(self.par_trans, direction="to_est")
        if self.dmeas is None:
            raise ValueError("dpop_train requires self.dmeas to be not None.")
        if J < 1:
            raise ValueError("J should be greater than 0")

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

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

        new_key, old_key = self._update_fresh_key(key)
        keys = jnp.array(jax.random.split(new_key, n_reps))

        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        nLLs_jax, theta_ests_jax = run_jax_batch_sharded(
            dpop_train,
            {1: 0, 8: 0},
            [0, 0],
            self.to_struct(),
            theta_array,
            J,
            optimizer,
            M,
            eta,
            alpha,
            process_weight_index,
            keys,
            alpha_cooling,
        )

        nLLs, theta_ests = jax.device_get((nLLs_jax, theta_ests_jax))
        del nLLs_jax, theta_ests_jax

        theta_ests_natural = self.par_trans._transform_array(
            theta_ests,
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
                "theta_idx": range(n_reps),
                "iteration": range(M + 1),
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

        execution_time = time.time() - start_time

        result = build_dpop_train_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces=joined_array,
            optimizer=optimizer,
            J=J,
            M=M,
            eta=eta,
            alpha=alpha,
            alpha_cooling=alpha_cooling,
            process_weight_state=process_weight_state,
        )

        self.results_history.add(result)

    def _pmcmc(
        self,
        J: int,
        M: int,
        proposal: Proposal,
        *,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
    ) -> None:
        """
        Particle Markov chain Monte Carlo (PMMH) for Bayesian parameter inference.

        Runs one independent PMCMC chain for each parameter replicate in ``theta``.
        Each chain uses a bootstrap particle filter likelihood estimate inside a
        Metropolis-Hastings update. Results are stored in
        :attr:`Pomp.results_history`.

        Parameters
        ----------
        J : int
            Number of particles per particle-filter likelihood evaluation.
        M : int
            Number of MCMC iterations per chain.
        proposal : Proposal
            Proposal object from :mod:`pypomp.proposals`.
        dprior : Callable, optional
            Pure-JAX log-prior function. If ``None``, uses the model's prior or a
            flat improper prior. See :ref:`dprior-tutorial`.
        key : jax.Array, optional
            JAX PRNG key. Defaults to :attr:`fresh_key`.
        theta : PompParameters, optional
            Starting parameter values. Defaults to :attr:`theta`.
        thresh : float, default 0.0
            Adaptive resampling threshold passed to the particle filter.

        Returns
        -------
        None
            Updates :attr:`Pomp.results_history` with a
            :class:`~pypomp.core.results.Result`.
        """
        start_time = time.time()

        if self.dmeas is None:
            raise ValueError("pmcmc requires self.dmeas to be not None.")
        if J < 1:
            raise ValueError("J must be >= 1.")
        if M < 1:
            raise ValueError("M must be >= 1.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_chains = theta_obj_in.num_replicates()
        if n_chains < 1:
            raise ValueError("pmcmc requires at least one starting parameter set.")

        new_key, old_key = self._update_fresh_key(key)
        canonical_names = self.canonical_param_names
        theta_array = theta_obj_in.to_jax_array(canonical_names)

        keys = jax.random.split(new_key, n_chains)

        ll_jax, lp_jax, theta_jax, accepts_jax = pmcmc(
            struct=self.to_struct(),
            thetas_array=theta_array,
            proposal=proposal,
            J=J,
            M=M,
            thresh=thresh,
            keys=keys,
            dprior=dprior,
        )

        ll_traces, lp_traces, theta_traces, accepts = jax.device_get(
            (ll_jax, lp_jax, theta_jax, accepts_jax)
        )
        del ll_jax, lp_jax, theta_jax, accepts_jax

        trace_vars = ["logLik", "log_prior"] + list(canonical_names)
        trace_data = np.concatenate(
            [
                ll_traces[..., np.newaxis],
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
                "iteration": np.arange(M + 1),
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

        execution_time = time.time() - start_time
        result = build_pmcmc_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces=traces_da,
            M=M,
            J=J,
            accepts=np.asarray(accepts, dtype=np.int32),
        )
        self.results_history.add(result)

    def _abc(
        self,
        M: int,
        probes: dict[str, Callable],
        epsilon: float,
        proposal: Proposal,
        *,
        scale: dict[str, float] | None = None,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
    ) -> None:
        r"""
        Approximate Bayesian Computation with a Metropolis-Hastings outer loop.

        The probe functions must be pure JAX callables accepting a dict that
        maps each observation name to a simulated ``(n_obs,)`` JAX array and
        returning a scalar. One independent ABC-MCMC chain is run for each
        parameter replicate in ``theta``. Results are stored in
        :attr:`Pomp.results_history`.

        Parameters
        ----------
        M : int
            Number of ABC-MCMC iterations per chain.
        probes : dict
            Mapping from probe name (``str``) to a pure-JAX summary-statistic
            callable ``probe_fn(y) -> scalar``, where ``y`` is a dict mapping
            each observation name to a ``(n_obs,)`` JAX array of simulated
            values for that variable, e.g. ``lambda y: jnp.mean(y["cases"])``.
        epsilon : float
            ABC distance rejection threshold.
        proposal : Proposal
            Proposal object from :mod:`pypomp.proposals`.
        scale : dict, optional
            Mapping from probe name (``str``, matching the keys of ``probes``)
            to a positive scaling factor (``float``) used to normalize probe
            differences in the squared scaled Euclidean distance, e.g.,
            :math:`d = \sum_i \left( \frac{s_i(y^*) - s_i(y)}{w_i} \right)^2`
            where :math:`w_i` is ``scale[i]``. If ``None``, a scale of ``1.0``
            is used for every probe.
        dprior : Callable, optional
            Pure-JAX log-prior function. If ``None``, uses the model's prior if given,
            otherwise a flat improper prior. See :ref:`dprior-tutorial`.
        key : jax.Array, optional
            JAX PRNG key. Defaults to :attr:`fresh_key`.
        theta : PompParameters, optional
            Starting parameter values. Defaults to :attr:`theta`.

        Returns
        -------
        None
            Updates :attr:`Pomp.results_history` with a
            :class:`~pypomp.core.results.Result`.
        """
        start_time = time.time()

        if self.rmeas is None:
            raise ValueError("abc requires self.rmeas to be not None.")
        if M < 1:
            raise ValueError("M must be >= 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_chains = theta_obj_in.num_replicates()
        if n_chains < 1:
            raise ValueError("abc requires at least one starting parameter set.")

        new_key, old_key = self._update_fresh_key(key)
        canonical_names = self.canonical_param_names
        theta_array = theta_obj_in.to_jax_array(canonical_names)

        keys = jax.random.split(new_key, n_chains)

        dist_jax, lp_jax, theta_jax, accepts_jax = abc(
            struct=self.to_struct(),
            thetas_array=theta_array,
            proposal=proposal,
            probes=probes,
            scale=scale,
            epsilon=float(epsilon),
            M=M,
            keys=keys,
            dprior=dprior,
        )

        dist_traces, lp_traces, theta_traces, accepts = jax.device_get(
            (dist_jax, lp_jax, theta_jax, accepts_jax)
        )
        del dist_jax, lp_jax, theta_jax, accepts_jax

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
                "iteration": np.arange(M + 1),
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

        execution_time = time.time() - start_time
        result = build_abc_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces=traces_da,
            M=M,
            epsilon=float(epsilon),
            accepts=np.asarray(accepts, dtype=np.int32),
        )
        self.results_history.add(result)

    @overload
    def simulate(
        self,
        nsim: int = 1,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        key: jax.Array | None = None,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        nsim: int = 1,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        key: jax.Array | None = None,
        *,
        as_pomp: Literal[True],
    ) -> Pomp: ...

    def simulate(
        self,
        nsim: int = 1,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        key: jax.Array | None = None,
        as_pomp: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | Pomp:
        """Simulate latent states and observations from the POMP model.

        Propagates the latent state through time via ``rproc`` and draws
        synthetic observations from ``rmeas``.  JAX vectorises the
        computation across parameter sets and simulation replicates
        simultaneously.

        Parameters
        ----------
        nsim : int, optional
            Number of independent simulation replicates per parameter set.
            Defaults to ``1``.
        theta : PompParameters or None, optional
            Parameter set(s) to simulate from.  Defaults to :attr:`theta`.
        times : jax.Array or None, optional
            Observation times at which to simulate.  Defaults to the
            original ``ys`` index.
        key : jax.Array or None, optional
            JAX random key.  Defaults to :attr:`fresh_key`.
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
                    stacklevel=2,
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
        X_sims_jax, Y_sims_jax = F.simulate(
            self.to_struct(),
            nsim,
            thetas_array,
            times=times_array,
            keys=keys,
        )
        X_sims, Y_sims = jax.device_get((X_sims_jax, Y_sims_jax))
        del X_sims_jax, Y_sims_jax

        def _to_long(
            arr: jax.Array | np.ndarray,
            times_vec: jax.Array | np.ndarray | pd.Index,
            column_names: list[str],
        ) -> pd.DataFrame:
            vals = np.asarray(arr)  # (n_theta, n_sim, n_time, n_feat)
            n_theta_l, n_sim_l, n_time_l, n_feat_l = vals.shape
            flat = vals.reshape(n_theta_l * n_sim_l * n_time_l, n_feat_l)
            theta_idx_l = np.repeat(np.arange(n_theta_l), n_sim_l * n_time_l)
            sim_idx_l = np.tile(np.repeat(np.arange(n_sim_l), n_time_l), n_theta_l)
            time_vals_l = np.tile(
                np.asarray(times_vec).reshape(1, -1), (n_theta_l * n_sim_l, 1)
            ).reshape(-1)
            cols = pd.Index(column_names)
            df = pd.DataFrame(flat, columns=cols)
            df.insert(0, "time", time_vals_l)
            df.insert(0, "sim", sim_idx_l)
            df.insert(0, "theta_idx", theta_idx_l)
            return df

        times0 = np.concatenate([np.array([self.t0]), np.array(times_array)])
        X_sims_long = _to_long(X_sims, times0, self.statenames)
        Y_sims_long = _to_long(Y_sims, np.array(times_array), list(self.ys.columns))

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
            return cast("Pomp", pomp_copy)

        return X_sims_long, Y_sims_long

    def probe(
        self,
        probes: dict[str, Callable[[dict[str, jax.Array]], float]],
        *,
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
            receives a dict mapping each observation name to a ``(n_obs,)``
            JAX array (with time ordered as in :attr:`Pomp.ys`) and returns
            a scalar, e.g. ``{"mean": lambda y: jnp.mean(y["cases"])}``.
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

        real_dict = {
            col: jnp.asarray(self.ys[col].to_numpy()) for col in self.ys.columns
        }
        for name, func in probes.items():
            results.append(
                {
                    "probe": name,
                    "value": float(func(real_dict)),
                    "is_real_data": True,
                    "theta_idx": pd.NA,
                    "sim": pd.NA,
                }
            )

        def apply_probes(group):
            theta_idx, sim_id = group.name
            df = pd.DataFrame(group.drop(columns=["time"]))
            df.columns = self.ys.columns
            y_dict = {col: jnp.asarray(df[col].to_numpy()) for col in df.columns}
            for name, func in probes.items():
                results.append(
                    {
                        "probe": name,
                        "value": float(func(y_dict)),
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
