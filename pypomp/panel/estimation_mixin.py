from __future__ import annotations
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
import numpy as np
import time
import pypomp.functional as F
from copy import deepcopy
from typing import TYPE_CHECKING, cast, Callable, overload, Literal
import warnings


from ..core.algorithms.train_panel_dpop import _vmapped_panel_dpop_train_internal
from ..core.algorithms.helpers import run_jax_batch_sharded
from ..core.rw_sigma import RWSigma
from ..core.learning_rate import LearningRate
from ..core.optimizer import Optimizer, Adam
from ..core.results import (
    build_panel_pfilter_result,
    build_panel_mif_result,
    build_panel_train_result,
    build_panel_dpop_train_result,
)

from ..core.parameters import PanelParameters
from ..maths import logmeanexp
from .. import benchmarks


if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
    from ..core.pomp import Pomp
else:
    Base = object  # At runtime, this is just a normal class


class PanelEstimationMixin(Base):
    """
    Handles Simulation, Particle Filtering, and MIF algorithms.
    """

    def _update_fresh_key(
        self, key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        old_key = self.fresh_key if key is None else key
        if old_key is None:
            raise ValueError(
                "Both the key argument and the fresh_key attribute are None. At least one key must be given."
            )
        new_f_key, new_key = jax.random.split(old_key)
        self.fresh_key = new_f_key
        return new_key, old_key

    def _get_covars_per_unit(self, unit_names: list[str]) -> jax.Array | None:
        has_covars = [
            self.unit_objects[u]._covars_extended is not None for u in unit_names
        ]
        if all(has_covars):
            return jnp.stack(
                [jnp.array(self.unit_objects[u]._covars_extended) for u in unit_names],
                axis=0,
            )
        if any(has_covars):
            raise NotImplementedError(
                "Some units have covariates, but not all units have covariates. This is not supported yet."
            )
        return None

    def _prepare_theta_input(
        self,
        theta: PanelParameters | None,
    ) -> PanelParameters:
        """Convert theta input to PanelParameters."""
        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be provided or self.theta must exist")
            return self.theta
        if not isinstance(theta, PanelParameters):
            raise TypeError("theta must be a PanelParameters instance or None")
        return theta

    def _get_unit_param_permutation(self, unit_name: str) -> jax.Array:
        unit_canonical = self.unit_objects[unit_name].canonical_param_names
        panel_canonical = self.canonical_param_names

        try:
            permutation = [panel_canonical.index(name) for name in unit_canonical]
        except ValueError as e:
            missing = set(unit_canonical) - set(panel_canonical)
            raise ValueError(
                f"Unit '{unit_name}' has parameters {missing} that are not in the panel's parameter list. "
                f"Panel parameters: {panel_canonical}, Unit parameters: {unit_canonical}"
            ) from e
        return jnp.array(permutation, dtype=jnp.int32)

    def _dataframe_to_array_canonical(
        self, df: pd.DataFrame, param_names: list[str], column_name: str
    ) -> jnp.ndarray:
        ordered_values = [df.loc[name, column_name] for name in param_names]
        return jnp.array(ordered_values, dtype=float)

    def get_unit_parameters(
        self,
        unit: str,
        theta: PanelParameters | None = None,
    ) -> list[dict[str, float]]:
        """Get parameter values for a specific unit across all replicates.

        Parameters
        ----------
        unit : str
            Name of the unit.
        theta : PanelParameters or None, optional
            Parameter values.  If ``None`` (default), the parameter values of the
            panel model are used.

        Returns
        -------
        list of dict
            List of dictionaries containing the parameter values for the
            specified unit across replicates.
        """
        theta = self._prepare_theta_input(theta)

        tll = theta.num_replicates()
        params: list[dict[str, float]] = [{} for _ in range(tll)]

        theta_list = theta.params(as_list=True)
        for i in range(tll):
            theta_dict = theta_list[i]
            if theta_dict["shared"] is not None:
                params[i].update(
                    cast(dict[str, float], theta_dict["shared"].iloc[:, 0].to_dict())
                )
            if theta_dict["unit_specific"] is not None:
                params[i].update(
                    cast(dict[str, float], theta_dict["unit_specific"][unit].to_dict())
                )

        return params

    @staticmethod
    def sample_params(
        param_bounds: dict,
        units: list[str],
        n: int,
        key: jax.Array,
        shared_names: list[str] | None = None,
    ) -> PanelParameters:
        """Sample parameters uniformly within bounds for a panel model.

        Parameters
        ----------
        param_bounds : dict of str to tuple of float
            Mapping from parameter names to ``(lower, upper)`` bounds.
        units : list of str
            Unit names.
        n : int
            Number of replicates to sample.
        key : jax.Array
            JAX random key.
        shared_names : list of str, optional
            List of shared parameter names.  If ``None`` (default), all
            parameters are considered unit-specific.

        Returns
        -------
        PanelParameters
            A new parameter set containing the sampled values.
        """
        shared = list(shared_names or [])
        specific = [k for k in param_bounds if k not in shared]

        key_s, key_u = jax.random.split(key)

        if shared:
            low_s = jnp.array([param_bounds[p][0] for p in shared])
            high_s = jnp.array([param_bounds[p][1] for p in shared])
            s_samples = jax.random.uniform(
                key_s, (n, len(shared)), minval=low_s, maxval=high_s
            )
            shared_values = np.array(s_samples)
        else:
            shared_values = np.empty((n, 0))

        if specific:
            low_u = jnp.array([param_bounds[p][0] for p in specific])
            high_u = jnp.array([param_bounds[p][1] for p in specific])
            low_u_3d = low_u[jnp.newaxis, jnp.newaxis, :]
            high_u_3d = high_u[jnp.newaxis, jnp.newaxis, :]
            u_samples = jax.random.uniform(
                key_u, (n, len(units), len(specific)), minval=low_u_3d, maxval=high_u_3d
            )
            unit_specific_values = np.array(u_samples)
        else:
            unit_specific_values = np.empty((n, len(units), 0))

        return PanelParameters.from_arrays(
            shared_values=shared_values,
            unit_specific_values=unit_specific_values,
            shared_names=shared,
            unit_specific_names=specific,
            unit_names=units,
        )

    @overload
    def simulate(
        self,
        key: jax.Array,
        theta: PanelParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        key: jax.Array,
        theta: PanelParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        *,
        as_pomp: Literal[True],
    ) -> "Base": ...

    def simulate(
        self,
        key: jax.Array,
        theta: PanelParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | Base:
        """Simulate latent states and observations from the panel model.

        Parameters
        ----------
        key : jax.Array
            JAX random key.
        theta : PanelParameters or None, optional
            Parameters to simulate from.  If ``None``, defaults to ``self.theta``.
        times : jax.Array or None, optional
            Times at which to simulate the model.  If ``None``, defaults to
            the times coordinate of the data.
        nsim : int, optional
            Number of simulations to run per replicate.  Defaults to ``1``.
        as_pomp : bool, optional
            If ``True``, return a new ``PanelPomp`` object containing the
            simulated observations for the first parameter replicate and
            simulation index.  Defaults to ``False``.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame) or PanelPomp
            If ``as_pomp=False``, returns a tuple ``(X_sims, Y_sims)`` of dataframes
            in long format.  If ``as_pomp=True``, returns a deep copy of the
            original panel model with simulated observations.
        """
        if as_pomp:
            if nsim > 1:
                warnings.warn(
                    "as_pomp is True, but nsim > 1. Only 1 simulation will be performed as_pomp overrides nsim.",
                    UserWarning,
                )
            nsim = 1

        theta = self._prepare_theta_input(theta)

        X_sims_list = []
        Y_sims_list = []
        new_unit_objects: dict[str, "Pomp"] = {}
        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, theta=theta)
            from ..core.parameters import PompParameters

            theta_obj = PompParameters(theta_list)
            key, subkey = jax.random.split(key)
            result = obj.simulate(
                key=subkey,
                theta=theta_obj,
                times=times,
                nsim=nsim,
                as_pomp=as_pomp,
            )
            if as_pomp:
                from ..core.pomp import Pomp

                assert isinstance(result, Pomp)
                new_unit_objects[unit] = result
            else:
                assert isinstance(result, tuple)
                X_sims, Y_sims = result
                X_sims.insert(0, "unit", unit)
                Y_sims.insert(0, "unit", unit)
                X_sims_list.append(X_sims)
                Y_sims_list.append(Y_sims)

        if as_pomp:
            panel_copy = deepcopy(self)
            panel_copy.unit_objects = new_unit_objects
            panel_copy.theta = theta.subset([0])
            return panel_copy

        X_sims_long = pd.concat(X_sims_list)
        Y_sims_long = pd.concat(Y_sims_list)

        return X_sims_long, Y_sims_long

    def probe(
        self,
        probes: dict[str, Callable[[pd.DataFrame], float]],
        key: jax.Array,
        nsim: int = 100,
        theta: PanelParameters | None = None,
    ) -> pd.DataFrame:
        """Evaluate probe statistics on real and simulated data for each unit.

        Parameters
        ----------
        probes : dict of str to callable
            Dictionary mapping probe names to functions.  Each function must
            take a dataframe of observations for a single unit and return a
            scalar float.
        key : jax.Array
            JAX random key.
        nsim : int, optional
            Number of simulations to run per replicate.  Defaults to ``100``.
        theta : PanelParameters or None, optional
            Parameters to simulate from.  If ``None``, defaults to ``self.theta``.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
                `probe`, `value`, `is_real_data`, `theta_idx`, `sim`, `unit`
        """
        sim_result = self.simulate(nsim=nsim, key=key, theta=theta)
        assert isinstance(sim_result, tuple)
        _, y_sims = sim_result

        results = []

        for unit, obj in self.unit_objects.items():
            for name, func in probes.items():
                results.append(
                    {
                        "probe": name,
                        "value": float(func(obj.ys)),
                        "is_real_data": True,
                        "theta_idx": pd.NA,
                        "sim": pd.NA,
                        "unit": unit,
                    }
                )

        for grp_key, group in y_sims.groupby(["unit", "theta_idx", "sim"]):
            unit_name, replicate_id, sim_id = grp_key
            obj = self.unit_objects[str(unit_name)]
            df = pd.DataFrame(group.drop(columns=["unit", "theta_idx", "sim", "time"]))
            df.index = pd.Index(group["time"])
            df.columns = obj.ys.columns
            for name, func in probes.items():
                results.append(
                    {
                        "probe": name,
                        "value": float(func(df)),
                        "is_real_data": False,
                        "theta_idx": replicate_id,
                        "sim": sim_id,
                        "unit": unit_name,
                    }
                )

        return pd.DataFrame(results)

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: PanelParameters | None = None,
        thresh: float = 0.0,
        reps: int = 1,
        chunk_size: int = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
    ) -> None:
        """Run the bootstrap particle filter (SMC) algorithm.

        Evaluates the likelihood of the panel data at the specified parameter values.

        Parameters
        ----------
        J : int
            Number of particles per unit.
        key : jax.Array or None, optional
            JAX random key.  If ``None``, uses the model's ``fresh_key``.
        theta : PanelParameters or None, optional
            Parameter sets to use.  If ``None``, defaults to ``self.theta``.
        thresh : float, optional
            Resampling threshold.  If ``0.0`` (default), always resample at
            each observation time.
        reps : int, optional
            Number of replicates per parameter set.  Defaults to ``1``.
        chunk_size : int, optional
            Number of units to process per batch.  Defaults to ``1``.
        CLL : bool, optional
            Whether to compute conditional log-likelihoods.  Defaults to
            ``False``.
        ESS : bool, optional
            Whether to compute effective sample sizes.  Defaults to ``False``.
        filter_mean : bool, optional
            Whether to compute filtering state means.  Defaults to ``False``.
        prediction_mean : bool, optional
            Whether to compute predicted state means.  Defaults to ``False``.

        Returns
        -------
        None
            Updates the unit-specific log-likelihoods ``self.theta.logLik_unit``
            and appends a :class:`~pypomp.core.results.Result` to the history.
        """
        # 1. Setup keys and prepare parameters.
        start_time = time.time()
        thresh = float(max(0.0, thresh))
        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_for_result = deepcopy(theta_obj_in)
        new_key, old_key = self._update_fresh_key(key)

        n_theta_reps = theta_obj_in.num_replicates()
        unit_names = list(self.unit_objects.keys())
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        chunk_size = max(1, int(chunk_size))

        # 2. Extract unit-level parameter values from panel parameters.
        thetas_panel = jnp.stack(
            [
                jnp.array(
                    [
                        [t[name] for name in self.unit_objects[u].canonical_param_names]
                        for t in self.get_unit_parameters(u, theta=theta_obj_in)
                    ]
                )
                for u in unit_names
            ],
            axis=1,
        )
        thetas_panel_repl = jnp.repeat(thetas_panel, reps, axis=0)

        # 3. Construct PanelPompStruct and apply padding if necessary.
        padding = (chunk_size - (U % chunk_size)) % chunk_size
        U_padded = U + padding

        rep_unit_keys = jax.random.split(new_key, n_theta_reps * reps * U_padded)
        rep_unit_keys = rep_unit_keys.reshape(
            (n_theta_reps * reps, U_padded) + rep_unit_keys.shape[1:]
        )

        struct = self.to_struct()
        if padding > 0:
            thetas_panel_repl = jnp.pad(
                thetas_panel_repl, ((0, 0), (0, padding), (0, 0))
            )
            ys_padded = jnp.pad(struct.ys_per_unit, ((0, padding), (0, 0), (0, 0)))
            covars_padded = (
                jnp.pad(struct.covars_per_unit, ((0, padding), (0, 0), (0, 0)))
                if struct.covars_per_unit is not None
                else None
            )
            struct = struct._replace(
                ys_per_unit=ys_padded, covars_per_unit=covars_padded
            )

        # 4. Execute the sharded panel particle filter in parallel batches.
        results_jax = run_jax_batch_sharded(
            F.panel_pfilter,
            {1: 0, 3: 0},
            {
                "logLik": 0,
                "CLL": 0,
                "ESS": 0,
                "filter_mean": 0,
                "prediction_mean": 0,
            },
            struct,
            thetas_panel_repl,
            J,
            rep_unit_keys,
            thresh=thresh,
            chunk_size=chunk_size,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
        )

        results = jax.device_get(results_jax)
        del results_jax

        # 5. Extract and reshape log-likelihood values.
        logLik_per_unit = results["logLik"][:, :U]
        logLik_per_unit = logLik_per_unit.reshape(n_theta_reps, reps, U)

        results_da = xr.DataArray(
            logLik_per_unit,
            dims=["theta_idx", "rep", "unit"],
            coords={"unit": unit_names, "rep": range(reps)},
        ).transpose("theta_idx", "unit", "rep")

        results_np = np.array(results_da.values)
        logLik_unit = logmeanexp(results_np, axis=-1, ignore_nan=False)

        theta_obj_in.logLik_unit = logLik_unit
        self.theta = theta_obj_in

        # 6. Extract and format requested diagnostics.
        def _reshape_and_stack_diagnostics(arr, dims, coord_names):
            if arr is None or arr.size == 0:
                return None
            arr = arr[:, :U]
            arr = arr.reshape((n_theta_reps, reps, U) + arr.shape[2:])
            arr = np.moveaxis(arr, 1, 2)
            coords = {"unit": unit_names, "rep": range(reps)}
            for i, coord_name in enumerate(coord_names):
                coord_idx = -(len(coord_names) - i)
                if coord_name == "time":
                    coords[coord_name] = rep_unit.ys.index
                else:
                    coords[coord_name] = range(arr.shape[coord_idx])
            return xr.DataArray(arr, dims=dims, coords=coords)

        CLL_da = (
            _reshape_and_stack_diagnostics(
                results.get("CLL"), ["theta_idx", "unit", "rep", "time"], ["time"]
            )
            if CLL
            else None
        )
        ESS_da = (
            _reshape_and_stack_diagnostics(
                results.get("ESS"), ["theta_idx", "unit", "rep", "time"], ["time"]
            )
            if ESS
            else None
        )
        filter_mean_da = (
            _reshape_and_stack_diagnostics(
                results.get("filter_mean"),
                ["theta_idx", "unit", "rep", "time", "state"],
                ["time", "state"],
            )
            if filter_mean
            else None
        )
        prediction_mean_da = (
            _reshape_and_stack_diagnostics(
                results.get("prediction_mean"),
                ["theta_idx", "unit", "rep", "time", "state"],
                ["time", "state"],
            )
            if prediction_mean
            else None
        )

        # 7. Record execution results in history.
        execution_time = time.time() - start_time
        result = build_panel_pfilter_result(
            execution_time=execution_time,
            key=old_key,
            theta=theta_for_result,
            logLiks=results_da,
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
        key: jax.Array | None = None,
        theta: PanelParameters | None = None,
        thresh: float = 0.0,
        n_monitors: int = 0,
        block: bool = True,
    ) -> None:
        """Estimate parameters using the (Marginalized) Panel Iterated Filtering (MPIF/PIF) algorithm.

        Performs parameter estimation using the (Marginalized) Panel Iterated
        Filtering (MPIF/PIF) algorithm (Bretó et al. 2020 [1]_; Wheeler et al. 2025 [2]_).

        Parameters
        ----------
        J : int
            Number of particles per unit.
        M : int
            Number of iterations (cooling cycles).
        rw_sd : RWSigma
            Random walk standard deviations and cooling schedule.
        key : jax.Array or None, optional
            JAX random key.  If ``None``, uses the model's ``fresh_key``.
        theta : PanelParameters or None, optional
            Initial parameter estimates.  If ``None``, defaults to ``self.theta``.
        thresh : float, optional
            Resampling threshold for the particle filter.  Defaults to ``0.0``.
        n_monitors : int, optional
            Number of unperturbed particle filter runs to estimate log-likelihood
            at each iteration.  Defaults to ``0`` (use perturbed filter).
        block : bool, optional
            Whether to use block updates (MPIF).  If ``False``, uses standard PIF.
            Defaults to ``True``.

        Returns
        -------
        None
            Updates ``self.theta`` with final estimates and appends a
            :class:`~pypomp.core.results.Result` to the history.

        References
        ----------
        .. [1] Bretó, Carles, Edward L. Ionides, and Aaron A. King. "Panel Data Analysis
           via Mechanistic Models." *Journal of the American Statistical Association*
           115, no. 531 (2020): 1178–1188. https://doi.org/10.1080/01621459.2019.1604367.
        .. [2] Wheeler, Jesse, Aaron J. Abkemeier, and Edward L. Ionides. "Iterating
           marginalized Bayes maps for likelihood maximization with application to nonlinear
           panel models." *arXiv preprint arXiv:2511.17438* (2025). https://arxiv.org/abs/2511.17438.
        """
        start_time = time.time()
        thresh = float(max(0.0, thresh))
        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_for_result = deepcopy(theta_obj_in)

        n_reps = theta_obj_in.num_replicates()
        unit_names = self.get_unit_names()
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        if J < 1 or M < 1:
            raise ValueError("J and M must be greater than 0.")
        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        shared_index = self.canonical_shared_param_names
        n_shared = len(shared_index)
        if n_shared == 0:
            shared_array = jnp.zeros((n_reps, J, 0))
        else:
            shared_vals = theta_obj_in.to_jax_array(shared_index, unit_names=unit_names)
            shared_array = jnp.repeat(shared_vals[:, jnp.newaxis, 0, :], J, axis=1)

        spec_index = self.canonical_unit_param_names
        n_spec = len(spec_index)
        if n_spec == 0:
            unit_array = jnp.zeros((n_reps, J, U, 0))
        else:
            spec_vals = theta_obj_in.to_jax_array(spec_index, unit_names=unit_names)
            unit_array = jnp.repeat(spec_vals[:, jnp.newaxis, :, :], J, axis=1)

        key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(key, n_reps)

        struct = self.to_struct()
        (
            shared_traces_jax,
            unit_traces_jax,
            final_shared_swarm_jax,
            final_unit_swarm_jax,
        ) = run_jax_batch_sharded(
            F.panel_mif,
            {1: 0, 2: 0, 6: 0},
            [0, 0, 0, 0],
            struct,
            shared_array,
            unit_array,
            rw_sd,
            M,
            J,
            keys,
            thresh,
            n_monitors,
            block,
        )

        shared_traces = (
            np.array(shared_traces_jax) if shared_traces_jax is not None else None
        )
        unit_traces = np.array(unit_traces_jax) if unit_traces_jax is not None else None

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            shared_traces = np.sum(unit_traces[:, :, :, 0], axis=-1, keepdims=True)
            shared_index = []
        if unit_traces is None:
            unit_traces = np.zeros((shared_traces.shape[0], M + 1, U, 1))

        shared_da = xr.DataArray(
            shared_traces,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(shared_traces.shape[0]),
                "iteration": np.arange(M + 1),
                "variable": ["logLik"] + shared_index,
            },
        )
        unit_da = xr.DataArray(
            unit_traces,
            dims=["theta_idx", "iteration", "unit", "variable"],
            coords={
                "theta_idx": np.arange(unit_traces.shape[0]),
                "iteration": np.arange(M + 1),
                "unit": unit_names,
                "variable": ["unitLogLik"] + spec_index,
            },
        )

        self.theta = PanelParameters.from_arrays(
            shared_values=shared_traces[:, -1, 1:],
            unit_specific_values=unit_traces[:, -1, :, 1:],
            shared_names=shared_index,
            unit_specific_names=spec_index,
            unit_names=unit_names,
            logLik_unit=unit_traces[:, -1, :, 0].astype(float),
            estimation_scale=False,
        )

        result = build_panel_mif_result(
            execution_time=time.time() - start_time,
            key=old_key,
            theta=theta_for_result,
            shared_traces=shared_da,
            unit_traces=unit_da,
            J=J,
            M=M,
            rw_sd=rw_sd,
            thresh=thresh,
            n_monitors=n_monitors,
            block=block,
            logLiks=xr.DataArray(
                np.concatenate(
                    [shared_traces[:, -1, 0:1], unit_traces[:, -1, :, 0]], axis=1
                ),
                dims=["theta_idx", "unit"],
                coords={
                    "theta_idx": np.arange(n_reps),
                    "unit": ["shared"] + unit_names,
                },
            ),
        )
        self.results_history.add(result)

    def train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        chunk_size: int = 1,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.97,
        key: jax.Array | None = None,
        theta: PanelParameters | None = None,
        alpha_cooling: float = 1.0,
    ):
        """Estimate parameters using MOP-based gradient-descent optimization.

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
            Number of particles per unit.
        M : int
            Number of training iterations (gradient steps).
        eta : LearningRate
            Learning rates per parameter.
        chunk_size : int, optional
            Number of units to process in parallel per gradient step.  Defaults
            to ``1``.
        optimizer : Optimizer, optional
            Optimizer configuration object.  Defaults to ``Adam()``.
        alpha : float, optional
            MOP discount factor.  Defaults to ``0.97``.
        key : jax.Array or None, optional
            JAX random key.  If ``None``, uses the model's ``fresh_key``.
        theta : PanelParameters or None, optional
            Initial parameter estimates.  If ``None``, defaults to ``self.theta``.
        alpha_cooling : float, optional
            Cooling factor for the MOP discount factor ``alpha`` using cosine decay.
            Defaults to ``1.0``.

        Returns
        -------
        None
            Updates ``self.theta`` with final estimates and appends a
            :class:`~pypomp.core.results.Result` to the history.

        References
        ----------
        .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
           for Partially Observed Markov Processes using Automatic Differentiation."
           *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
        """
        start_time = time.time()
        theta_obj_in: PanelParameters = deepcopy(self._prepare_theta_input(theta))
        theta_for_result = deepcopy(theta_obj_in)

        n_reps = theta_obj_in.num_replicates()

        key, old_key = self._update_fresh_key(key)
        if J < 1 or M < 1:
            raise ValueError("J and M must be greater than 0.")

        unit_names = self.get_unit_names()
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        chunk_size = max(1, int(chunk_size))

        shared_index = self.canonical_shared_param_names
        n_shared = len(shared_index)
        if n_shared == 0:
            shared_array = jnp.zeros((n_reps, 0))
            shared_index = []
        else:
            shared_array = theta_obj_in.to_jax_array(
                shared_index, unit_names=unit_names
            )[:, 0, :]

        spec_index = self.canonical_unit_param_names
        n_spec = len(spec_index)
        if n_spec == 0:
            unit_array = jnp.zeros((n_reps, U, 0))
            spec_index = []
        else:
            unit_array = theta_obj_in.to_jax_array(spec_index, unit_names=unit_names)

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        eta_shared = eta.to_array(shared_index, M)
        eta_spec = eta.to_array(spec_index, M)

        keys = jax.random.split(key, n_reps * M * U).reshape(
            (n_reps, M, U) + key.shape[1:]
        )

        struct = self.to_struct()
        (
            logliks_history,
            shared_history_natural,
            unit_history_natural,
        ) = run_jax_batch_sharded(
            F.panel_train,
            {1: 0, 2: 0, 9: 0},
            [0, 0, 0],
            struct,
            shared_array,
            unit_array,
            J,
            optimizer,
            M,
            eta_shared,
            eta_spec,
            alpha,
            keys,
            alpha_cooling,
            chunk_size,
        )

        shared_traces = None
        if (
            shared_history_natural is not None
            and logliks_history is not None
            and n_shared > 0
        ):
            shared_ll_expanded = np.expand_dims(-np.array(logliks_history), axis=-1)
            shared_traces = np.concatenate(
                [shared_ll_expanded, np.array(shared_history_natural)], axis=-1
            )

        unit_traces = None
        if unit_history_natural is not None and n_spec > 0:
            nan_ll = np.full((n_reps, M + 1, U, 1), np.nan, dtype=float)
            unit_traces = np.concatenate([nan_ll, unit_history_natural], axis=-1)

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_traces = np.expand_dims(-np.array(logliks_history), axis=-1)
            shared_index = []

        if unit_traces is None:
            n_reps = shared_traces.shape[0]
            unit_traces = np.zeros((n_reps, M + 1, U, 1), dtype=float)

        shared_da = xr.DataArray(
            shared_traces,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(shared_traces.shape[0]),
                "iteration": np.arange(M + 1),
                "variable": ["logLik"] + shared_index,
            },
        )
        unit_da = xr.DataArray(
            unit_traces,
            dims=["theta_idx", "iteration", "unit", "variable"],
            coords={
                "theta_idx": np.arange(unit_traces.shape[0]),
                "iteration": np.arange(M + 1),
                "unit": unit_names,
                "variable": ["unitLogLik"] + spec_index,
            },
        )

        self.theta = PanelParameters.from_arrays(
            shared_values=shared_traces[:, -1, 1:],
            unit_specific_values=unit_traces[:, -1, :, 1:],
            shared_names=shared_index,
            unit_specific_names=spec_index,
            unit_names=unit_names,
            logLik_unit=np.full((n_reps, U), np.nan),
            estimation_scale=False,
        )

        result = build_panel_train_result(
            execution_time=time.time() - start_time,
            key=old_key,
            theta=theta_for_result,
            shared_traces=shared_da,
            unit_traces=unit_da,
            logLiks=xr.DataArray(  # Placeholder as we don't have unit logliks separated
                np.full((n_reps, U + 1), np.nan),
                dims=["theta_idx", "unit"],
                coords={
                    "theta_idx": np.arange(n_reps),
                    "unit": ["shared"] + unit_names,
                },
            ),
            J=J,
            M=M,
            eta=eta,
            optimizer=optimizer,
            alpha=alpha,
            alpha_cooling=alpha_cooling,
        )

        self.results_history.add(result)

    def dpop_train(
        self,
        J: int,
        M: int,
        eta: "LearningRate | dict[str, float] | float",
        chunk_size: int = 1,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.97,
        alpha_cooling: float = 1.0,
        decay: float = 0.0,
        process_weight_state: str | None = None,
        key: jax.Array | None = None,
        theta: PanelParameters | None = None,
    ):
        """Estimate parameters using DPOP-based gradient-descent optimization.

        .. warning::
           This method is experimental.  Its API and behavior are subject to change
           in future releases.

        Parameters
        ----------
        J : int
            Number of particles per unit.
        M : int
            Number of training iterations.
        eta : LearningRate or dict or float
            Learning rate(s).
        chunk_size : int, optional
            Number of units to process per gradient step.  Defaults to ``1``.
        optimizer : Optimizer, optional
            Optimizer configuration object.  Defaults to ``Adam()``.
        alpha : float, optional
            DPOP discount / cooling factor.  Defaults to ``0.97``.
        alpha_cooling : float, optional
            Cosine cooling factor for alpha.  Defaults to ``1.0``.
        decay : float, optional
            Learning-rate decay coefficient.  Defaults to ``0.0``.
        process_weight_state : str or None, optional
            Name of the state component that stores the accumulated process
            log-weight (e.g. ``"logw"``).
        key : jax.Array or None, optional
            JAX random key.  If ``None``, uses the model's ``fresh_key``.
        theta : PanelParameters or None, optional
            Initial parameter estimates.  If ``None``, defaults to ``self.theta``.
        """
        warnings.warn(
            "dpop_train is experimental and its API and behavior are subject to change.",
            category=FutureWarning,
            stacklevel=2,
        )

        start_time = time.time()
        theta_obj_in: PanelParameters = deepcopy(self._prepare_theta_input(theta))
        if theta_obj_in is None:
            raise ValueError("theta must be provided or self.theta must exist")

        key, old_key = self._update_fresh_key(key)
        if J < 1:
            raise ValueError("J should be greater than 0.")
        if M < 1:
            raise ValueError("M should be greater than 0.")

        unit_names = self.get_unit_names()
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        # Determine chunk size
        chunk_size_value = int(chunk_size)

        if chunk_size_value < 1:
            chunk_size_value = 1
        chunk_size_value = min(chunk_size_value, U)
        if U % chunk_size_value != 0:
            original_chunk_size = chunk_size_value
            chunk_size_value = max(
                d for d in range(chunk_size_value, 0, -1) if U % d == 0
            )
            warnings.warn(
                "chunk_size does not divide the number of units; "
                f"using chunk_size={chunk_size_value} instead of {original_chunk_size}.",
                UserWarning,
            )
        chunk_size = chunk_size_value

        # Determine process_weight_index
        if process_weight_state is None:
            raise ValueError(
                "dpop_train requires a process-weight state. "
                "Please provide `process_weight_state` as the name of the "
                "state variable that accumulates the transition log-weight "
                "(e.g. 'logw')."
            )
        try:
            process_weight_index = int(rep_unit.statenames.index(process_weight_state))
        except ValueError:
            raise ValueError(
                f"process_weight_state '{process_weight_state}' not found in "
                f"statenames: {rep_unit.statenames}"
            )

        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )

        dt_array_extended = rep_unit._dt_array_extended
        nstep_array = rep_unit._nstep_array
        t0 = rep_unit.t0
        times = jnp.array(rep_unit.ys.index)

        rinitializers = rep_unit.rinit.struct_pf
        rprocesses_interp = rep_unit.rproc.struct_pf_interp
        dmeasures = rep_unit.dmeas.struct_pf
        accumvars = rep_unit.rproc.accumvars

        has_covars = [
            self.unit_objects[u]._covars_extended is not None for u in unit_names
        ]
        if all(has_covars):
            covars_per_unit = jnp.stack(
                [jnp.array(self.unit_objects[u]._covars_extended) for u in unit_names],
                axis=0,
            )
        elif any(has_covars):
            raise NotImplementedError(
                "Some units have covariates, but not all units have covariates. This is not supported yet."
            )
        else:
            covars_per_unit = None

        n_reps = theta_obj_in.num_replicates()

        theta_obj_in = theta_obj_in.transformed(rep_unit.par_trans, direction="to_est")

        shared_index = self.canonical_shared_param_names
        n_shared = len(shared_index)
        if n_shared == 0:
            shared_array = jnp.zeros((n_reps, 0))
            shared_index = []
        else:
            shared_array = theta_obj_in.to_jax_array(
                shared_index, unit_names=unit_names
            )[:, 0, :]

        spec_index = self.canonical_unit_param_names
        n_spec = len(spec_index)
        if n_spec == 0:
            unit_array = jnp.zeros((n_reps, 0, U))
            spec_index = []
        else:
            unit_array = theta_obj_in.to_jax_array(
                spec_index, unit_names=unit_names
            ).transpose(0, 2, 1)

        if isinstance(eta, LearningRate):
            # Full (M, p) per-iteration schedule (e.g. cosine_decay)
            eta_shared = eta.to_array(shared_index, M)
            eta_spec = eta.to_array(spec_index, M)
        else:
            # Constant dict/float -> broadcast to an (M, p) schedule; the scalar
            # `decay` (reciprocal) is still applied per-iteration in the kernel.
            eta_dict = (
                eta
                if isinstance(eta, dict)
                else {p: eta for p in self.canonical_param_names}
            )
            eta_shared_vec = jnp.array(
                [eta_dict.get(p, 0.0) for p in shared_index], dtype=float
            )
            eta_spec_vec = jnp.array(
                [eta_dict.get(p, 0.0) for p in spec_index], dtype=float
            )
            eta_shared = jnp.broadcast_to(eta_shared_vec, (M, eta_shared_vec.shape[0]))
            eta_spec = jnp.broadcast_to(eta_spec_vec, (M, eta_spec_vec.shape[0]))

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )
        n_obs = ys_per_unit.shape[1]
        ntimes = n_obs

        keys = jax.random.split(key, n_reps * M * U)
        keys = keys.reshape((n_reps, M, U) + keys.shape[1:])

        (
            logliks_history,
            shared_history,
            unit_history,
        ) = _vmapped_panel_dpop_train_internal(
            shared_array,
            unit_array,
            unit_param_permutations,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            ys_per_unit,
            covars_per_unit,
            keys,
            J,
            rinitializers,
            rprocesses_interp,
            dmeasures,
            accumvars,
            chunk_size,
            optimizer,
            M,
            eta_shared,
            eta_spec,
            alpha,
            alpha_cooling,
            n_obs,
            U,
            process_weight_index,
            ntimes,
            decay,
        )
        logliks_trace = -np.array(logliks_history)

        shared_history_arr = np.array(shared_history) if len(shared_index) > 0 else None
        unit_history_arr = (
            np.transpose(np.array(unit_history), (0, 1, 3, 2))
            if len(spec_index) > 0
            else None
        )

        if shared_history_arr is None and unit_history_arr is None:
            raise ValueError(
                "Both shared_traces and unit_traces are None; cannot build traces."
            )

        shared_trans, unit_trans = rep_unit.par_trans._transform_panel_array(
            shared_array=shared_history_arr,
            unit_array=unit_history_arr,
            shared_names=shared_index,
            unit_specific_names=spec_index,
            direction="from_est",
        )

        shared_ll_expanded = np.expand_dims(logliks_trace, axis=-1)
        if shared_trans is not None:
            shared_traces = np.concatenate([shared_ll_expanded, shared_trans], axis=-1)
        else:
            shared_traces = shared_ll_expanded

        if len(spec_index) > 0 and unit_trans is not None:
            nan_ll = np.full((n_reps, M + 1, U, 1), np.nan, dtype=float)
            unit_traces = np.concatenate([nan_ll, unit_trans], axis=-1)
        else:
            unit_traces = np.zeros((n_reps, M + 1, U, 1), dtype=float)

        shared_vars = ["logLik"] + shared_index
        unit_vars = ["unitLogLik"] + spec_index

        shared_da = xr.DataArray(
            shared_traces,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": jnp.arange(shared_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": shared_vars,
            },
        )
        unit_da = xr.DataArray(
            unit_traces,
            dims=["theta_idx", "iteration", "unit", "variable"],
            coords={
                "theta_idx": jnp.arange(unit_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "unit": unit_names,
                "variable": unit_vars,
            },
        )

        logLik_unit_out = unit_traces[:, -1, :, 0].astype(float)

        self.theta = PanelParameters.from_arrays(
            shared_values=shared_traces[:, -1, 1:],
            unit_specific_values=unit_traces[:, -1, :, 1:],
            shared_names=shared_index,
            unit_specific_names=spec_index,
            unit_names=unit_names,
            logLik_unit=logLik_unit_out,
            estimation_scale=False,
        )

        execution_time = time.time() - start_time

        result = build_panel_dpop_train_result(
            execution_time=execution_time,
            key=old_key,
            theta=self.theta,
            shared_traces=shared_da,
            unit_traces=unit_da,
            logLiks=xr.DataArray(
                np.full((n_reps, U + 1), np.nan),
                dims=["theta_idx", "unit"],
                coords={
                    "theta_idx": jnp.arange(n_reps),
                    "unit": ["shared"] + unit_names,
                },
            ),
            J=J,
            M=M,
            eta=eta,
            optimizer=optimizer,
            alpha=alpha,
            alpha_cooling=alpha_cooling,
            process_weight_state=process_weight_state,
            decay=decay,
        )

        self.results_history.add(result)

    def arma(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        log_ys: bool = False,
        suppress_warnings: bool = True,
    ) -> pd.DataFrame:
        """Fit an independent ARIMA model to the observation data of each unit.

        This is a wrapper around :func:`pypomp.benchmarks.arma`.

        Parameters
        ----------
        order : tuple of (int, int, int), optional
            The ``(p, d, q)`` order for the ARIMA model.  Defaults to
            ``(1, 0, 1)``.
        log_ys : bool, optional
            If ``True``, fit the model to ``log(y + 1)``.  Defaults to ``False``.
        suppress_warnings : bool, optional
            If ``True``, suppress statsmodels warning messages during fitting.
            Defaults to ``True``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``'unit'`` and ``'logLik'`` containing
            the individual unit log-likelihoods and their sum (labeled as
            ``'[[TOTAL]]'``).
        """
        import warnings

        results = []
        total_llf = 0.0

        if suppress_warnings:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for name, unit in self.unit_objects.items():
                    llf = benchmarks.arma(
                        unit.ys, order=order, log_ys=log_ys, suppress_warnings=False
                    )
                    results.append({"unit": name, "logLik": llf})
                    total_llf += llf

            if len(w) > 0:
                warnings.warn(
                    f"arma: {len(w)} warnings were produced by statsmodels across units. "
                    "Set suppress_warnings=False to see the raw output.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            for name, unit in self.unit_objects.items():
                llf = benchmarks.arma(
                    unit.ys, order=order, log_ys=log_ys, suppress_warnings=False
                )
                results.append({"unit": name, "logLik": llf})
                total_llf += llf

        # Insert total at the beginning
        results.insert(0, {"unit": "[[TOTAL]]", "logLik": total_llf})
        return pd.DataFrame(results)

    def negbin(
        self, autoregressive: bool = False, suppress_warnings: bool = True
    ) -> pd.DataFrame:
        """Fit a Negative Binomial model to the observation data of each unit.

        Parameters
        ----------
        autoregressive : bool, optional
            If ``True``, fit an autoregressive AR(1) model.  Defaults to
            ``False`` (iid).
        suppress_warnings : bool, optional
            If ``True``, suppress statsmodels warning messages during fitting.
            Defaults to ``True``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``'unit'`` and ``'logLik'`` containing
            the individual unit log-likelihoods and their sum (labeled as
            ``'[[TOTAL]]'``).
        """
        import warnings

        results = []
        total_llf = 0.0

        if suppress_warnings:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for name, unit in self.unit_objects.items():
                    llf = benchmarks.negbin(
                        unit.ys, autoregressive=autoregressive, suppress_warnings=False
                    )
                    results.append({"unit": name, "logLik": llf})
                    total_llf += llf

            if len(w) > 0:
                warnings.warn(
                    f"negbin: {len(w)} warnings were produced by statsmodels across units. "
                    "Set suppress_warnings=False to see the raw output.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            for name, unit in self.unit_objects.items():
                llf = benchmarks.negbin(
                    unit.ys, autoregressive=autoregressive, suppress_warnings=False
                )
                results.append({"unit": name, "logLik": llf})
                total_llf += llf

        # Insert total at the beginning
        results.insert(0, {"unit": "[[TOTAL]]", "logLik": total_llf})
        return pd.DataFrame(results)
