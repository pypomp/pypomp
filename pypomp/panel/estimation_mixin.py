from __future__ import annotations
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
import numpy as np
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Union, cast, Callable, overload, Literal, Any
import warnings

from ..core.algorithms.pfilter import _chunked_panel_pfilter_internal
from ..core.algorithms.mif import _jv_panel_mif_internal, _jv_panel_mif_internal_vmap
from ..core.algorithms.train import _vmapped_panel_train_internal
from ..core.algorithms.train_panel_dpop import _vmapped_panel_dpop_train_internal
from ..core.algorithms.helpers import run_jax_batch_sharded
from ..core.rw_sigma import RWSigma
from ..core.learning_rate import LearningRate
from ..core.optimizer import Optimizer, Adam
from ..core.results import (
    PanelPompPFilterResult,
    PanelPompMIFResult,
    PanelPompTrainResult,
    PanelPompDpopTrainResult,
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
        """
        Get the parameter values for a specific unit across all replicates.

        Args
        ----
        unit (str): The name of the unit to get the parameter values for.
        theta (PanelParameters | None): The parameter values to get the parameter values for. If None, the parameter values of the panel will be used.

        Returns
        -------
        list[dict[str, float]]: A list of dictionaries containing the parameter values for the specified unit across all replicates.
        """
        theta = self._prepare_theta_input(theta)

        tll = theta.num_replicates()
        params: list[dict[str, float]] = [{} for _ in range(tll)]

        theta_list = theta.params()
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
        """
        Sample parameters for PanelPomp models.

        Args:
            param_bounds (dict): Dictionary mapping parameter names to (lower, upper) bounds.
            units (list[str]): List of unit names.
            n (int): Number of parameter sets to sample.
            key (jax.Array): JAX random key for reproducibility.
            shared_names (list[str], optional): List of shared parameter names. If None, all parameters are considered unit-specific.

        Returns:
            PanelParameters: A PanelParameters object containing the sampled parameters.
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

        shared_da = xr.DataArray(
            shared_values,
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n),
                "parameter": shared,
            },
        )
        unit_specific_da = xr.DataArray(
            unit_specific_values,
            dims=["theta_idx", "unit", "parameter"],
            coords={
                "theta_idx": np.arange(n),
                "unit": units,
                "parameter": specific,
            },
        )
        ds = xr.Dataset(
            data_vars={
                "shared": shared_da,
                "unit_specific": unit_specific_da,
            }
        )
        ds.attrs["shared_names"] = shared
        ds.attrs["unit_specific_names"] = specific
        return PanelParameters(ds)

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
        """
        Simulate the :class:`~pypomp.panel.panel.PanelPomp` model.

        Args:
            key (jax.Array): JAX random key.
            theta (:class:`~pypomp.core.parameters.PanelParameters`, optional): Parameter sets to use.
                If None, uses `self.theta`.
            times (jax.Array, optional): Times at which to simulate the model.
                If None, uses `self.times`.
            nsim (int, optional): Number of simulations to run.
            as_pomp (bool, optional): If True, returns a new :class:`~pypomp.panel.panel.PanelPomp` object containing the simulated
                observations for the first parameter replicate and simulation, instead of DataFrames.

        Returns:
            Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame], :class:`~pypomp.panel.panel.PanelPomp`]:
                If as_pomp is False, returns a tuple of (X_sims, Y_sims) DataFrames.
                If as_pomp is True, returns a deep copy of the original model with simulated observations.
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
        """
        Evaluate probe statistics on the model's true data and simulated data for each unit.

        Args:
            probes (dict[str, Callable[[pd.DataFrame], float]]): A dictionary of probe functions.
                Each function should receive a DataFrame of observations for a single unit and return a numeric scalar.
                Example: `{"mean": lambda df: df["obs"].mean()}`
            key (jax.Array): JAX random key for the simulations.
            nsim (int, optional): Number of simulations to run per parameter set. Defaults to 100.
            theta: Parameters to simulate from.

        Returns:
            pd.DataFrame: A long-format DataFrame with columns:
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
            unit_name, replicate_id, sim_id = cast(tuple[Any, Any, Any], grp_key)
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
        """
        Run the particle filter (SMC) algorithm on the :class:`~pypomp.panel.panel.PanelPomp` model.

        Args:
            J (int): Number of particles per unit.
            key (jax.Array, optional): JAX random key. If None, uses `self.fresh_key`.
            theta (:class:`~pypomp.core.parameters.PanelParameters`, optional): Parameter sets to use.
                If None, uses `self.theta`.
            thresh (float, optional): Resampling threshold. If 0.0, always resample.
            reps (int, optional): Number of replicates per parameter set.
            chunk_size (int, optional): Number of units to process per batch.
            CLL (bool, optional): Whether to compute conditional log-likelihoods.
            ESS (bool, optional): Whether to compute effective sample sizes.
            filter_mean (bool, optional): Whether to compute filtering means.
            prediction_mean (bool, optional): Whether to compute prediction means.

        Returns:
            None. Updates :attr:`self.theta.logLik_unit` and adds a :class:`~pypomp.core.results.PanelPompPFilterResult` to :attr:`self.results_history`.
        """
        start_time = time.time()
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

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )
        covars_per_unit = self._get_covars_per_unit(unit_names)

        thetas_per_unit = []
        for unit in unit_names:
            theta_list = self.get_unit_parameters(unit, theta=theta_obj_in)
            obj = self.unit_objects[unit]
            unit_arr = jnp.array(
                [[t[name] for name in obj.canonical_param_names] for t in theta_list]
            )
            thetas_per_unit.append(unit_arr)

        thetas_panel = jnp.stack(thetas_per_unit, axis=1)  # (n_theta_reps, U, n_params)
        thetas_panel_repl = jnp.repeat(
            thetas_panel, reps, axis=0
        )  # (n_theta_reps * reps, U, n_params)

        padding = (chunk_size - (U % chunk_size)) % chunk_size
        U_padded = U + padding

        # Pre-allocate keys at padded size: jnp.pad on PRNG keys would fail
        # because their dtype (key<fry>) cannot be filled with a scalar. When
        # padding == 0 (the common case), this matches the unpadded behavior.
        rep_unit_keys = jax.random.split(new_key, n_theta_reps * reps * U_padded)
        rep_unit_keys = rep_unit_keys.reshape(
            (n_theta_reps * reps, U_padded) + rep_unit_keys.shape[1:]
        )

        if padding > 0:
            thetas_panel_repl = jnp.pad(
                thetas_panel_repl, ((0, 0), (0, padding), (0, 0))
            )
            ys_per_unit = jnp.pad(ys_per_unit, ((0, padding), (0, 0), (0, 0)))
            if covars_per_unit is not None:
                covars_per_unit = jnp.pad(
                    covars_per_unit, ((0, padding), (0, 0), (0, 0))
                )

        results_jax = run_jax_batch_sharded(
            _chunked_panel_pfilter_internal,
            {0: 0, 7: 0},
            {
                "neg_loglik": 0,
                "CLL": 0,
                "ESS": 0,
                "filter_mean": 0,
                "prediction_mean": 0,
            },
            thetas_panel_repl,
            rep_unit._dt_array_extended,
            rep_unit._nstep_array,
            rep_unit.t0,
            jnp.array(rep_unit.ys.index),
            ys_per_unit,
            covars_per_unit,
            rep_unit_keys,
            J,
            rep_unit.rinit.struct_pf,
            rep_unit.rproc.struct_pf_interp,
            rep_unit.dmeas.struct_pf,
            rep_unit.rproc.accumvars,
            thresh,
            chunk_size,
            CLL,
            ESS,
            filter_mean,
            prediction_mean,
        )

        results = jax.device_get(results_jax)
        del results_jax

        neg_logliks = results["neg_loglik"][:, :U]  # shape: (n_theta_reps * reps, U)
        neg_logliks = neg_logliks.reshape(n_theta_reps, reps, U)

        results_da = xr.DataArray(
            (-neg_logliks),
            dims=["theta_idx", "rep", "unit"],
            coords={"unit": unit_names, "rep": range(reps)},
        ).transpose("theta_idx", "unit", "rep")

        results_np = np.array(results_da.values)
        logLik_unit = logmeanexp(
            results_np, axis=-1, ignore_nan=False
        )  # shape: (n_theta_reps, len(self.unit_objects))

        theta_obj_in.logLik_unit = logLik_unit
        self.theta = theta_obj_in

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

        execution_time = time.time() - start_time

        result = PanelPompPFilterResult(
            method="pfilter",
            execution_time=execution_time,
            key=old_key,
            theta=theta_for_result,
            logLiks=results_da,
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
        theta: PanelParameters | None = None,
        thresh: float = 0,
        n_monitors: int = 0,
        block: bool = True,
        vmap_chunk_size: int | None = None,
    ) -> None:
        """
        Estimate parameters using the Panel Iterated Filtering (PIF) algorithm for :class:`~pypomp.panel.panel.PanelPomp`.

        Args:
            J (int): Number of particles per unit.
            M (int): Number of iterations (cooling cycles).
            rw_sd (:class:`~pypomp.core.rw_sigma.RWSigma`): Random walk standard deviations for parameter perturbations.
            key (jax.Array, optional): JAX random key. If None, uses `self.fresh_key`.
            theta (:class:`~pypomp.core.parameters.PanelParameters`, optional): Initial parameter estimates.
                If None, uses `self.theta`.
            thresh (float): Resampling threshold for the particle filter.
            n_monitors (int): Number of particle filter runs to average for
                log-likelihood estimation. Defaults to 0 (uses estimate from perturbed
                filter).
            block (bool): Whether to use block updates, i.e., Marginalized Panel Iterated Filtering (MPIF). Uses Panel Iterated Filtering (PIF) if False.
            vmap_chunk_size (int, optional): (Experimental) If set, process units in parallel via
                jax.vmap in chunks of this size instead of sequentially. Shared
                parameters are independently perturbed per unit and averaged across
                units at the end of each chunk. Padding is applied if the chunk
                size does not evenly divide the number of units.

        Returns:
            None. Updates :attr:`self.theta` with final estimates and adds a :class:`~pypomp.core.results.PanelPompMIFResult` to :attr:`self.results_history`.
        """
        start_time = time.time()
        theta_obj_in: PanelParameters = deepcopy(self._prepare_theta_input(theta))
        theta_for_result = deepcopy(theta_obj_in)

        n_reps = theta_obj_in.num_replicates()
        unit_names = self.get_unit_names()
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.canonical_param_names
        )

        if J < 1 or M < 1:
            raise ValueError("J and M must be greater than 0.")
        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )
        covars_per_unit = self._get_covars_per_unit(unit_names)

        theta_obj_in.transform(rep_unit.par_trans, direction="to_est")

        shared_index = self.canonical_shared_param_names
        n_shared = len(shared_index)
        if n_shared == 0:
            shared_array = jnp.zeros((n_reps, 0, J))
        else:
            shared_vals = theta_obj_in.to_jax_array(shared_index, unit_names=unit_names)
            shared_array = jnp.tile(shared_vals[:, 0, :, None], (1, 1, J))

        spec_index = self.canonical_unit_param_names
        n_spec = len(spec_index)
        if n_spec == 0:
            unit_array = jnp.zeros((n_reps, 0, J, U))
        else:
            spec_vals = theta_obj_in.to_jax_array(spec_index, unit_names=unit_names)
            unit_array = jnp.tile(
                spec_vals.transpose(0, 2, 1)[:, :, None, :], (1, 1, J, 1)
            )

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )

        key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(key, n_reps)

        # TODO: if the vmap mode works well, remove the sequential mode
        if vmap_chunk_size is not None:
            U_padded = U + (vmap_chunk_size - (U % vmap_chunk_size)) % vmap_chunk_size
            padding = U_padded - U
            unit_mask = jnp.concatenate([jnp.ones(U), jnp.zeros(padding)])

            if padding > 0:
                unit_param_permutations = jnp.pad(
                    unit_param_permutations, ((0, padding), (0, 0))
                )
                ys_per_unit = jnp.pad(
                    ys_per_unit, ((0, padding),) + ((0, 0),) * (ys_per_unit.ndim - 1)
                )
                if covars_per_unit is not None:
                    covars_per_unit = jnp.pad(
                        covars_per_unit,
                        ((0, padding),) + ((0, 0),) * (covars_per_unit.ndim - 1),
                    )
                if unit_array.shape[1] > 0:
                    unit_array = jnp.pad(
                        unit_array, ((0, 0), (0, 0), (0, 0), (0, padding))
                    )

            res = run_jax_batch_sharded(
                _jv_panel_mif_internal_vmap,
                {0: 0, 1: 0, 21: 0},
                [0, 0, 0, 0],
                shared_array,
                unit_array,
                rep_unit._dt_array_extended,
                rep_unit._nstep_array,
                rep_unit.t0,
                jnp.array(rep_unit.ys.index),
                ys_per_unit,
                rep_unit.rinit.struct_per,
                rep_unit.rproc.struct_per_interp,
                rep_unit.dmeas.struct_per,
                sigmas_array,
                sigmas_init_array,
                rep_unit.rproc.accumvars,
                covars_per_unit,
                unit_param_permutations,
                unit_mask,
                M,
                rw_sd.cooling_fn,
                J,
                U_padded,
                thresh,
                keys,
                vmap_chunk_size,
                rep_unit.rinit.struct_pf,
                rep_unit.rproc.struct_pf_interp,
                rep_unit.dmeas.struct_pf,
                n_monitors,
                block,
            )
            shared_array_f, unit_array_f, shared_traces, unit_traces = res
            if padding > 0:
                if unit_array_f.shape[1] > 0:
                    unit_array_f = unit_array_f[:, :, :, :U]
                unit_traces = unit_traces[:, :, :, :U]
        else:
            shared_array_f, unit_array_f, shared_traces, unit_traces = (
                run_jax_batch_sharded(
                    _jv_panel_mif_internal,
                    {0: 0, 1: 0, 20: 0},
                    [0, 0, 0, 0],
                    shared_array,
                    unit_array,
                    rep_unit._dt_array_extended,
                    rep_unit._nstep_array,
                    rep_unit.t0,
                    jnp.array(rep_unit.ys.index),
                    ys_per_unit,
                    rep_unit.rinit.struct_per,
                    rep_unit.rproc.struct_per_interp,
                    rep_unit.dmeas.struct_per,
                    sigmas_array,
                    sigmas_init_array,
                    rep_unit.rproc.accumvars,
                    covars_per_unit,
                    unit_param_permutations,
                    M,
                    rw_sd.cooling_fn,
                    J,
                    U,
                    thresh,
                    keys,
                    rep_unit.rinit.struct_pf,
                    rep_unit.rproc.struct_pf_interp,
                    rep_unit.dmeas.struct_pf,
                    n_monitors,
                    block,
                )
            )

        shared_traces, unit_traces = rep_unit.par_trans._transform_panel_traces(
            shared_traces=np.array(shared_traces),
            unit_traces=np.array(unit_traces),
            shared_param_names=shared_index,
            unit_param_names=spec_index,
            direction="from_est",
        )

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            shared_traces = np.sum(unit_traces[:, :, 0, :], axis=-1, keepdims=True)
            shared_index = []
        if unit_traces is None:
            unit_traces = np.zeros((shared_traces.shape[0], M + 1, 1, U))

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
            dims=["theta_idx", "iteration", "variable", "unit"],
            coords={
                "theta_idx": np.arange(unit_traces.shape[0]),
                "iteration": np.arange(M + 1),
                "variable": ["unitLogLik"] + spec_index,
                "unit": unit_names,
            },
        )

        ds = xr.Dataset(
            data_vars={
                "shared": xr.DataArray(
                    shared_traces[:, -1, 1:].astype(float),
                    dims=["theta_idx", "parameter"],
                    coords={
                        "theta_idx": np.arange(n_reps),
                        "parameter": shared_index,
                    },
                ),
                "unit_specific": xr.DataArray(
                    unit_traces[:, -1, 1:, :].transpose(0, 2, 1).astype(float),
                    dims=["theta_idx", "unit", "parameter"],
                    coords={
                        "theta_idx": np.arange(n_reps),
                        "unit": unit_names,
                        "parameter": spec_index,
                    },
                ),
            }
        )
        ds.attrs["shared_names"] = shared_index
        ds.attrs["unit_specific_names"] = spec_index

        self.theta = PanelParameters(
            theta=ds,
            logLik_unit=unit_traces[:, -1, 0, :].astype(float),
            estimation_scale=False,
        )

        result = PanelPompMIFResult(
            method="mif",
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
                    [shared_traces[:, -1, 0:1], unit_traces[:, -1, 0, :]], axis=1
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
        """
        Estimate parameters using chunked gradient-descent optimization (SGD/Adam).

        This method performs stochastic gradient descent (or Adam) iterations over
        the likelihood of the panel POMP. It operates by drawing particles for a
        subset of units (defined by `chunk_size`), calculating gradients for both
        shared and unit-specific parameters, and updating estimates.

        Args:
            J (int): Number of particles per unit.
            M (int): Number of training iterations.
            eta (:class:`~pypomp.core.learning_rate.LearningRate`): Learning rates per parameter as a :class:`~pypomp.core.learning_rate.LearningRate` object.
            chunk_size (int, optional): Number of units to process per
                gradient calculation step.
            optimizer (:class:`~pypomp.core.optimizer.Optimizer`, optional): The optimizer configuration object to use
                (e.g., `pp.Adam()`, `pp.SGD()`, `pp.FullMatrixAdam()`, etc.). Defaults to `pp.Adam()`.
                Hyperparameters like gradient clipping (`clip_norm`) or Adam beta values are
                configured directly inside the optimizer instance.
            alpha (float, optional): Learning rate decay factor per iteration.
            key (jax.Array, optional): JAX PRNG key. If None, uses the
                `fresh_key` attribute.
            theta (:class:`~pypomp.core.parameters.PanelParameters`, optional): Initial parameter estimates.
                If None, uses the current `theta` attribute.
            alpha_cooling (float, optional): Cooling factor for the MOP discount factor (alpha) using cosine decay. This factor represents the multiplier for the distance of alpha from 1.0 by the end of training (i.e., alpha approaches 1.0). Defaults to 1.0 (no cooling).

        Returns:
            None. Updates :attr:`self.theta` and adds a :class:`~pypomp.core.results.PanelPompTrainResult` to :attr:`self.results_history`.
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

        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )
        covars_per_unit = self._get_covars_per_unit(unit_names)

        theta_obj_in.transform(rep_unit.par_trans, direction="to_est")

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

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        eta_shared = eta.to_array(shared_index, M)
        eta_spec = eta.to_array(spec_index, M)

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )

        keys = jax.random.split(key, n_reps * M * U).reshape(
            (n_reps, M, U) + key.shape[1:]
        )

        opt_name = optimizer.__class__.__name__
        beta1 = getattr(optimizer, "beta1", 0.9)
        beta2 = getattr(optimizer, "beta2", 0.999)
        epsilon = getattr(optimizer, "epsilon", 1e-8 if opt_name == "Adam" else 1e-4)

        (
            logliks_history,
            shared_history,
            unit_history,
        ) = run_jax_batch_sharded(
            _vmapped_panel_train_internal,
            {0: 0, 1: 0, 9: 0},
            [0, 0, 0],
            shared_array,
            unit_array,
            unit_param_permutations,
            rep_unit._dt_array_extended,
            rep_unit._nstep_array,
            rep_unit.t0,
            jnp.array(rep_unit.ys.index),
            ys_per_unit,
            covars_per_unit,
            keys,
            J,
            rep_unit.rinit.struct_pf,
            rep_unit.rproc.struct_pf_interp,
            rep_unit.dmeas.struct_pf,
            rep_unit.rproc.accumvars,
            chunk_size,
            opt_name,
            M,
            eta_shared,
            eta_spec,
            alpha,
            alpha_cooling,
            ys_per_unit.shape[1],
            U,
            optimizer.clip_norm,
            beta1,
            beta2,
            epsilon,
        )

        shared_traces_in = None
        if n_shared > 0:
            shared_ll_expanded = np.expand_dims(np.array(logliks_history), axis=-1)
            shared_traces_in = np.concatenate(
                [shared_ll_expanded, np.array(shared_history)], axis=-1
            )

        unit_traces_in = None
        if n_spec > 0:
            nan_ll = np.full((n_reps, M + 1, 1, U), np.nan, dtype=float)
            unit_traces_in = np.concatenate([nan_ll, np.array(unit_history)], axis=-2)

        shared_traces, unit_traces = rep_unit.par_trans._transform_panel_traces(
            shared_traces=shared_traces_in,
            unit_traces=unit_traces_in,
            shared_param_names=shared_index,
            unit_param_names=spec_index,
            direction="from_est",
        )

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_traces = np.expand_dims(np.array(logliks_history), axis=-1)
            shared_index = []

        if unit_traces is None:
            n_reps = shared_traces.shape[0]
            unit_traces = np.zeros((n_reps, M + 1, 1, U), dtype=float)

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
            dims=["theta_idx", "iteration", "variable", "unit"],
            coords={
                "theta_idx": np.arange(unit_traces.shape[0]),
                "iteration": np.arange(M + 1),
                "variable": ["unitLogLik"] + spec_index,
                "unit": unit_names,
            },
        )

        ds = xr.Dataset(
            data_vars={
                "shared": xr.DataArray(
                    shared_traces[:, -1, 1:].astype(float),
                    dims=["theta_idx", "parameter"],
                    coords={
                        "theta_idx": np.arange(n_reps),
                        "parameter": shared_index,
                    },
                ),
                "unit_specific": xr.DataArray(
                    unit_traces[:, -1, 1:, :].transpose(0, 2, 1).astype(float),
                    dims=["theta_idx", "unit", "parameter"],
                    coords={
                        "theta_idx": np.arange(n_reps),
                        "unit": unit_names,
                        "parameter": spec_index,
                    },
                ),
            }
        )
        ds.attrs["shared_names"] = shared_index
        ds.attrs["unit_specific_names"] = spec_index

        self.theta = PanelParameters(
            theta=ds,
            logLik_unit=np.full((n_reps, U), np.nan),
            estimation_scale=False,
        )

        result = PanelPompTrainResult(
            method="train",
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
        chunk_size: Union[int, str] = 1,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.97,
        alpha_cooling: float = 1.0,
        decay: float = 0.0,
        process_weight_state: str | None = None,
        key: jax.Array | None = None,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
    ):
        """
        Estimate parameters using DPOP gradient-descent optimization (SGD/Adam).

        This method performs stochastic gradient descent (or Adam) iterations over
        the DPOP likelihood of the panel POMP. It operates by drawing particles for a
        subset of units (defined by `chunk_size`), calculating gradients for both
        shared and unit-specific parameters, and updating estimates.

        Args:
            J (int): Number of particles per unit.
            M (int): Number of training iterations.
            eta (LearningRate | dict[str, float] | float): Learning rate(s). A
                LearningRate gives a full per-iteration schedule, e.g.
                ``LearningRate(rates).cosine_decay(0.05, M)``; a dict maps param
                names to constant rates; a float is one global constant rate. For
                the dict/float forms the scalar ``decay`` still applies reciprocal
                LR decay; for a LearningRate the schedule is used as-is.
            chunk_size (Union[int, str], optional): Number of units to process
                per gradient calculation step.
            optimizer (Optimizer, optional): Optimizer configuration object,
                e.g. ``Adam()`` or ``SGD()``. Adam hyperparameters (beta1, beta2,
                epsilon) are read from the object; pass ``Adam(beta1=0.0)`` to
                disable momentum (e.g. for the high-variance alpha=0 arm,
                matching the dmop/IFAD convention).
            alpha (float, optional): DPOP discount / cooling factor.
            alpha_cooling (float, optional): Cosine cooling factor for alpha.
                This factor represents the multiplier for the distance of alpha
                from 1.0 by the end of training. The default keeps alpha fixed.
            decay (float, optional): Learning-rate decay coefficient. At iteration m,
                the effective learning rate is ``eta / (1 + decay * m)``.
            process_weight_state (str or None): Name of the state component that
                stores the accumulated process log-weight (e.g. ``"logw"``).
            key (jax.Array, optional): JAX PRNG key. If None, uses the
                `fresh_key` attribute.
            theta (PanelParameters, optional): Initial parameter estimates.
                If None, uses the current `theta` attribute.
        """
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
        if chunk_size == "auto":
            try:
                import psutil

                bytes_per_unit = (
                    J * len(rep_unit.statenames) * len(rep_unit.ys.index) * 200
                )
                mem = psutil.virtual_memory()
                avail = mem.available * 0.4
                max_units = max(1, int(avail / bytes_per_unit))
                chunk_size_value = min(U, max_units)
                try:
                    device = jax.devices()[0]
                    if device.platform == "gpu":
                        memory_stats = device.memory_stats()
                        if memory_stats is not None:
                            avail = (
                                memory_stats["bytes_limit"]
                                - memory_stats["bytes_in_use"]
                            )
                            max_units = max(1, int(avail * 0.4 / bytes_per_unit))
                            chunk_size_value = min(U, max_units)
                except Exception:
                    pass
            except Exception:
                chunk_size_value = max(1, U // 4)
        else:
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
            process_weight_index = int(
                rep_unit.statenames.index(process_weight_state)
            )
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

        theta_obj_in.transform(rep_unit.par_trans, direction="to_est")

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
            eta_shared = jnp.broadcast_to(
                eta_shared_vec, (M, eta_shared_vec.shape[0])
            )
            eta_spec = jnp.broadcast_to(eta_spec_vec, (M, eta_spec_vec.shape[0]))

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )
        n_obs = ys_per_unit.shape[1]
        ntimes = n_obs

        keys = jax.random.split(key, n_reps * M * U)
        keys = keys.reshape((n_reps, M, U) + keys.shape[1:])

        opt_name = optimizer.__class__.__name__
        beta1 = getattr(optimizer, "beta1", 0.9)
        beta2 = getattr(optimizer, "beta2", 0.999)
        epsilon = getattr(optimizer, "epsilon", 1e-8)

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
            opt_name,
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
            beta1,
            beta2,
            epsilon,
        )
        logliks_trace = -np.array(logliks_history)

        shared_traces_in = None
        unit_traces_in = None

        if len(shared_index) > 0:
            shared_ll_expanded = np.expand_dims(logliks_trace, axis=-1)
            shared_traces_in = np.concatenate(
                [shared_ll_expanded, np.array(shared_history)], axis=-1
            )

        if len(spec_index) > 0:
            nan_ll = np.full((n_reps, M + 1, 1, U), np.nan, dtype=float)
            unit_traces_in = np.concatenate([nan_ll, np.array(unit_history)], axis=-2)

        shared_traces, unit_traces = rep_unit.par_trans._transform_panel_traces(
            shared_traces=shared_traces_in,
            unit_traces=unit_traces_in,
            shared_param_names=shared_index,
            unit_param_names=spec_index,
            direction="from_est",
        )

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_ll = np.expand_dims(logliks_trace, axis=-1)
            shared_traces = shared_ll
            shared_index = []

        if unit_traces is None:
            n_reps = shared_traces.shape[0]
            unit_ll = np.zeros((n_reps, M + 1, 1, U), dtype=float)
            unit_traces = unit_ll

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
            dims=["theta_idx", "iteration", "variable", "unit"],
            coords={
                "theta_idx": jnp.arange(unit_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": unit_vars,
                "unit": unit_names,
            },
        )

        logLik_unit_out = np.full((n_reps, U), np.nan)

        ds = xr.Dataset(
            data_vars={
                "shared": xr.DataArray(
                    shared_traces[:, -1, 1:].astype(float),
                    dims=["theta_idx", "parameter"],
                    coords={
                        "theta_idx": np.arange(n_reps),
                        "parameter": shared_index,
                    },
                ),
                "unit_specific": xr.DataArray(
                    unit_traces[:, -1, 1:, :].transpose(0, 2, 1).astype(float),
                    dims=["theta_idx", "unit", "parameter"],
                    coords={
                        "theta_idx": np.arange(n_reps),
                        "unit": unit_names,
                        "parameter": spec_index,
                    },
                ),
            }
        )
        ds.attrs["shared_names"] = shared_index
        ds.attrs["unit_specific_names"] = spec_index

        self.theta = PanelParameters(
            theta=ds,
            logLik_unit=logLik_unit_out,
            estimation_scale=False,
        )

        execution_time = time.time() - start_time

        result = PanelPompDpopTrainResult(
            method="dpop_train",
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
        """
        Fits an independent ARIMA model to the observation data for each unit and returns
        a DataFrame with the estimated log-likelihoods for each unit and the total.

        This is a wrapper around `pypomp.benchmarks.arma`.

        Args:
            order (tuple, optional): The (p, d, q) order for the ARIMA model. Defaults to (1, 0, 1).
            log_ys (bool, optional): If True, fits the model to log(y+1). Defaults to False.
            suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels
                and issues a summary warning instead. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with columns 'unit' and 'logLik' containing results for each unit
                and their sum (labeled as '[[TOTAL]]' in the first row).
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
        """
        Fits a Negative Binomial model to the observation data for each unit and
        returns a DataFrame with the estimated log-likelihoods for each unit and the total.

        Args:
            autoregressive (bool, optional): If True, fits an AR(1) model.
                Defaults to False (iid).
            suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels/optimization
                and issues a summary warning instead. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with columns 'unit' and 'logLik' containing results for each unit
                and their sum (labeled as '[[TOTAL]]' in the first row).
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
