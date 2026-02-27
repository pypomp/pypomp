import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
import numpy as np
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Union, cast

from ..pfilter import _chunked_panel_pfilter_internal
from ..mif import _jv_panel_mif_internal
from ..train import _vmapped_panel_train_internal
from ..RWSigma_class import RWSigma
from ..results import (
    PanelPompPFilterResult,
    PanelPompMIFResult,
    PanelPompTrainResult,
)
from ..parameters import PanelParameters
from ..util import logmeanexp
from ..benchmarks import (
    arma_benchmark as _arma_benchmark,
    negbin_benchmark as _negbin_benchmark,
)

if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
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
        self.fresh_key, new_key = jax.random.split(old_key)
        return new_key, old_key

    def _prepare_theta_input(
        self,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ],
    ) -> PanelParameters:
        """Convert various theta inputs to PanelParameters."""
        if isinstance(theta, PanelParameters):
            return theta
        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be provided or self.theta must exist")
            return self.theta
        return PanelParameters(theta=theta)

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
        theta = self._prepare_theta_input(theta)

        tll = theta.num_replicates()
        params = [{} for _ in range(tll)]

        for i in range(tll):
            theta_dict = theta.theta[i]
            if theta_dict["shared"] is not None:
                params[i].update(theta_dict["shared"].iloc[:, 0].to_dict())
            if theta_dict["unit_specific"] is not None:
                params[i].update(theta_dict["unit_specific"][unit].to_dict())

        return params

    @staticmethod
    def sample_params(
        param_bounds: dict,
        units: list[str],
        n: int,
        key: jax.Array,
        shared_names: list[str] | None = None,
    ) -> list[dict[str, pd.DataFrame | None]]:
        """
        Sample parameters for PanelPomp models.

        Args:
            param_bounds (dict): Dictionary mapping parameter names to (lower, upper) bounds.
            units (list[str]): List of unit names.
            n (int): Number of parameter sets to sample.
            key (jax.Array): JAX random key for reproducibility.
            shared_names (list[str], optional): List of shared parameter names. If None, all parameters are considered unit-specific.

        Returns:
            list[dict[str, pd.DataFrame | None]]: List of n dictionaries containing sampled parameters. Each dictionary contains "shared" and "unit_specific" keys mapping to DataFrames or None.
        """
        shared = shared_names or []
        specific = [k for k in param_bounds if k not in shared]
        keys = jax.random.split(key, n)

        def _sample(k, names, n_cols):
            if not names:
                return None
            # Create arrays of shape (n_params, 1) for broadcasting
            low = jnp.array([param_bounds[p][0] for p in names])[:, None]
            high = jnp.array([param_bounds[p][1] for p in names])[:, None]
            # Sample (n_params, n_cols)
            return low + jax.random.uniform(k, (len(names), n_cols)) * (high - low)

        results = []
        for i in range(n):
            k_s, k_u = jax.random.split(keys[i])
            s_arr = _sample(k_s, shared, 1)
            u_arr = _sample(k_u, specific, len(units))

            results.append(
                {
                    "shared": pd.DataFrame(
                        s_arr, index=pd.Index(shared), columns=pd.Index(["shared"])
                    )
                    if shared
                    else None,
                    "unit_specific": pd.DataFrame(
                        u_arr, index=pd.Index(specific), columns=pd.Index(units)
                    )
                    if specific
                    else None,
                }
            )

        return results

    def simulate(
        self,
        key: jax.Array,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
        times: jax.Array | None = None,
        nsim: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        theta = self._prepare_theta_input(theta)

        X_sims_list = []
        Y_sims_list = []
        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, theta=theta)
            key, subkey = jax.random.split(key)
            X_sims, Y_sims = obj.simulate(
                key=subkey,
                theta=theta_list,
                times=times,
                nsim=nsim,
            )
            X_sims.insert(0, "unit", unit)
            Y_sims.insert(0, "unit", unit)
            X_sims_list.append(X_sims)
            Y_sims_list.append(Y_sims)
        X_sims_long = pd.concat(X_sims_list)
        Y_sims_long = pd.concat(Y_sims_list)
        return X_sims_long, Y_sims_long

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
        thresh: float = 0.0,
        reps: int = 1,
        chunk_size: Union[int, str] = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
    ) -> None:
        """
        Run the particle filter (SMC) algorithm on the PanelPomp model.

        Args:
            J (int): Number of particles per unit.
            key (jax.Array, optional): JAX random key. If None, uses `self.fresh_key`.
            theta (PanelParameters | dict | list, optional): Parameter sets to use.
                If None, uses `self.theta`.
            thresh (float, optional): Resampling threshold. If 0.0, always resample.
            reps (int, optional): Number of replicates per parameter set.
            chunk_size (Union[int, str], optional): Number of units to process
                per batch. 'auto' will attempt to estimate based on memory.
            CLL (bool, optional): Whether to compute conditional log-likelihoods.
            ESS (bool, optional): Whether to compute effective sample sizes.
            filter_mean (bool, optional): Whether to compute filtering means.
            prediction_mean (bool, optional): Whether to compute prediction means.

        Returns:
            None: Updates `self.theta.logLik_unit` and adds result to `self.results_history`.
        """
        start_time = time.time()
        theta_obj_in = deepcopy(self._prepare_theta_input(theta))

        new_key, old_key = self._update_fresh_key(key)

        n_theta_reps = theta_obj_in.num_replicates()
        unit_names = list(self.unit_objects.keys())
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        # "auto" is experimental and should maybe be deleted
        if chunk_size == "auto":
            try:
                import psutil

                bytes_per_unit = (
                    J * len(rep_unit.statenames) * len(rep_unit.ys.index) * 200
                )  # rough estimate
                mem = psutil.virtual_memory()
                avail = mem.available * 0.4
                max_units = max(1, int(avail / bytes_per_unit))
                chunk_size = min(U, max_units)
                try:
                    device = jax.devices()[0]
                    if device.platform == "gpu":
                        avail = (
                            device.memory_stats()["bytes_limit"]
                            - device.memory_stats()["bytes_in_use"]
                        )
                        max_units = max(1, int(avail * 0.4 / bytes_per_unit))
                        chunk_size = min(U, max_units)
                except Exception:
                    pass
            except Exception:
                chunk_size = max(1, U // 4)
        else:
            chunk_size = int(chunk_size)

        if chunk_size < 1:
            chunk_size = 1

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )
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

        rep_unit_keys = jax.random.split(new_key, n_theta_reps * reps * U)
        rep_unit_keys = rep_unit_keys.reshape(
            (n_theta_reps * reps, U) + rep_unit_keys.shape[1:]
        )

        padding = (chunk_size - (U % chunk_size)) % chunk_size

        if padding > 0:
            thetas_panel_repl = jnp.pad(
                thetas_panel_repl, ((0, 0), (0, padding), (0, 0))
            )
            ys_per_unit = jnp.pad(ys_per_unit, ((0, padding), (0, 0), (0, 0)))
            if covars_per_unit is not None:
                covars_per_unit = jnp.pad(
                    covars_per_unit, ((0, padding), (0, 0), (0, 0))
                )
            rep_unit_keys = jnp.pad(
                rep_unit_keys,
                ((0, 0), (0, padding)) + ((0, 0),) * (rep_unit_keys.ndim - 2),
            )

        results_jax = _chunked_panel_pfilter_internal(
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
            dims=["theta", "replicate", "unit"],
            coords={"unit": unit_names, "replicate": range(reps)},
        ).transpose("theta", "unit", "replicate")

        results_np = np.array(results_da.values)
        logLik_unit = np.apply_along_axis(logmeanexp, -1, results_np, ignore_nan=False)  # type: ignore # shape: (n_theta_reps, len(self.unit_objects))

        self.theta.logLik_unit = logLik_unit

        def _reshape_and_stack_diagnostics(arr, dims, coord_names):
            if arr is None or arr.size == 0:
                return None
            arr = arr[:, :U]
            arr = arr.reshape((n_theta_reps, reps, U) + arr.shape[2:])
            arr = np.moveaxis(arr, 1, 2)
            coords = {"unit": unit_names, "replicate": range(reps)}
            for i, coord_name in enumerate(coord_names):
                coord_idx = -(len(coord_names) - i)
                if coord_name == "time":
                    coords[coord_name] = rep_unit.ys.index
                else:
                    coords[coord_name] = range(arr.shape[coord_idx])
            return xr.DataArray(arr, dims=dims, coords=coords)

        CLL_da = (
            _reshape_and_stack_diagnostics(
                results.get("CLL"), ["theta", "unit", "replicate", "time"], ["time"]
            )
            if CLL
            else None
        )

        ESS_da = (
            _reshape_and_stack_diagnostics(
                results.get("ESS"), ["theta", "unit", "replicate", "time"], ["time"]
            )
            if ESS
            else None
        )

        filter_mean_da = (
            _reshape_and_stack_diagnostics(
                results.get("filter_mean"),
                ["theta", "unit", "replicate", "time", "state"],
                ["time", "state"],
            )
            if filter_mean
            else None
        )

        prediction_mean_da = (
            _reshape_and_stack_diagnostics(
                results.get("prediction_mean"),
                ["theta", "unit", "replicate", "time", "state"],
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
            theta=theta_obj_in,
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
        a: float,
        key: jax.Array | None = None,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
        thresh: float = 0,
        block: bool = True,
    ) -> None:
        """
        Estimate parameters using the Panel Iterated Filtering (PIF) algorithm for PanelPomp.

        Args:
            J (int): Number of particles per unit.
            M (int): Number of iterations (cooling cycles).
            rw_sd (RWSigma): Random walk standard deviations for parameter perturbations.
            a (float): Cooling factor (perturbation variance reduction per unit time).
            key (jax.Array, optional): JAX random key. If None, uses `self.fresh_key`.
            theta (PanelParameters | dict | list, optional): Initial parameter estimates.
                If None, uses `self.theta`.
            thresh (float, optional): Resampling threshold for the particle filter.
            block (bool, optional): Whether to use block updates, i.e., Marginalized Panel Iterated Filtering (MPIF) (currently only block=True is supported).

        Returns:
            None: Updates `self.theta` with final estimates and adds result to `self.results_history`.
        """
        start_time = time.time()
        theta_obj_in: PanelParameters = deepcopy(self._prepare_theta_input(theta))
        if theta_obj_in is None:
            raise ValueError("theta must be provided or self.theta must exist")

        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.canonical_param_names
        )
        key, old_key = self._update_fresh_key(key)
        if J < 1:
            raise ValueError("J should be greater than 0.")
        if M < 1:
            raise ValueError("M should be greater than 0.")
        if a < 0 or a > 1:
            raise ValueError("a should be between 0 and 1.")
        if block is False:
            raise NotImplementedError("block=False is not supported yet.")

        unit_names = self.get_unit_names()
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )

        dt_array_extended = rep_unit._dt_array_extended
        nstep_array = rep_unit._nstep_array
        t0 = rep_unit.t0
        times = jnp.array(rep_unit.ys.index)

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        rinitializers = rep_unit.rinit.struct_per
        rprocesses_interp = rep_unit.rproc.struct_per_interp
        dmeasures = rep_unit.dmeas.struct_per
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

        # Extract theta list from PanelParameters and transform to estimation scale if needed
        theta_list = theta_obj_in.theta
        if not theta_obj_in.estimation_scale:
            # Create a copy to avoid modifying the original
            theta_list = [
                {
                    "shared": t["shared"].copy() if t["shared"] is not None else None,
                    "unit_specific": (
                        t["unit_specific"].copy()
                        if t["unit_specific"] is not None
                        else None
                    ),
                }
                for t in theta_list
            ]
            theta_trans_list = rep_unit.par_trans.panel_transform_list(
                theta_list, direction="to_est"
            )
        else:
            theta_trans_list = theta_list

        # Extract shared and unit_specific DataFrames from transformed list
        shared_trans_list = [t.get("shared") for t in theta_trans_list]
        spec_trans_list = [t.get("unit_specific") for t in theta_trans_list]

        # Store original shared/unit_specific for later use
        shared = [t.get("shared") for t in theta_list]
        unit_specific = [t.get("unit_specific") for t in theta_list]

        if all(df is None for df in shared_trans_list):
            n_shared = 0
            shared_array = jnp.zeros((n_reps, 0, J))
            shared_index: list[str] = []
        else:
            shared_index = self.canonical_shared_param_names
            n_shared = len(shared_index)
            # PanelParameters ensures all shared are None or all are not None
            # Cast to list[pd.DataFrame] since we know all are not None here
            shared_trans_list_nonnull = cast(list[pd.DataFrame], shared_trans_list)
            shared_array = jnp.stack(
                [
                    jnp.tile(
                        self._dataframe_to_array_canonical(
                            df, self.canonical_shared_param_names, "shared"
                        ).reshape(n_shared, 1),
                        (1, J),
                    )
                    for df in shared_trans_list_nonnull
                ],
                axis=0,
            )

        if all(df is None for df in spec_trans_list):
            n_spec = 0
            unit_array = jnp.zeros((n_reps, 0, J, U))
            spec_index: list[str] = []
        else:
            spec_index = self.canonical_unit_param_names
            n_spec = len(spec_index)
            # PanelParameters ensures all unit_specific are None or all are not None
            # Cast to list[pd.DataFrame] since we know all are not None here
            spec_trans_list_nonnull = cast(list[pd.DataFrame], spec_trans_list)
            unit_array = jnp.stack(
                [
                    jnp.stack(
                        [
                            jnp.tile(
                                self._dataframe_to_array_canonical(
                                    df, self.canonical_unit_param_names, unit
                                ).reshape(n_spec, 1),
                                (1, J),
                            )
                            for unit in unit_names
                        ],
                        axis=2,
                    )
                    for df in spec_trans_list_nonnull
                ],
                axis=0,
            )

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )

        old_key = key
        keys = jax.random.split(key, n_reps)

        (
            shared_array_f,
            unit_array_f,
            shared_traces,
            unit_traces,
        ) = _jv_panel_mif_internal(
            shared_array,
            unit_array,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            ys_per_unit,
            rinitializers,
            rprocesses_interp,
            dmeasures,
            sigmas_array,
            sigmas_init_array,
            accumvars,
            covars_per_unit,
            unit_param_permutations,
            M,
            a,
            J,
            U,
            thresh,
            keys,
        )

        shared_traces, unit_traces = rep_unit.par_trans.transform_panel_traces(
            shared_traces=np.array(shared_traces),
            unit_traces=np.array(unit_traces),
            shared_param_names=shared_index,
            unit_param_names=spec_index,
            unit_names=unit_names,
            direction="from_est",
        )

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_ll = np.sum(unit_traces[:, :, 0, :], axis=-1, keepdims=True)
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
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": jnp.arange(shared_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": shared_vars,
            },
        )
        unit_da = xr.DataArray(
            unit_traces,
            dims=["replicate", "iteration", "variable", "unit"],
            coords={
                "replicate": jnp.arange(unit_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": unit_vars,
                "unit": unit_names,
            },
        )

        shared_final_logliks = shared_traces[:, -1, 0]
        unit_final_logliks = unit_traces[:, -1, 0, :]

        full_logliks = xr.DataArray(
            jnp.concatenate(
                [shared_final_logliks.reshape(-1, 1), unit_final_logliks], axis=1
            ),
            dims=["replicate", "unit"],
            coords={"replicate": jnp.arange(n_reps), "unit": ["shared"] + unit_names},
        )

        # PanelParameters ensures all shared are None or all are not None
        if shared and shared[0] is not None:
            shared_list_out = [
                pd.DataFrame(
                    shared_traces[rep, -1, 1:].reshape(-1, 1),
                    index=pd.Index(shared_index),
                    columns=pd.Index(["shared"]),
                )
                for rep in range(shared_traces.shape[0])
            ]
        else:
            shared_list_out = None

        # PanelParameters ensures all unit_specific are None or all are not None
        if unit_specific and unit_specific[0] is not None:
            specific_list_out = [
                pd.DataFrame(
                    unit_traces[rep, -1, 1:, :],
                    index=pd.Index(spec_index),
                    columns=pd.Index(unit_names),
                )
                for rep in range(unit_traces.shape[0])
            ]
        else:
            specific_list_out = None

        theta_list_out = [
            {
                "shared": shared_list_out[rep] if shared_list_out else None,
                "unit_specific": specific_list_out[rep] if specific_list_out else None,
            }
            for rep in range(n_reps)
        ]

        # unit_final_logliks has shape (n_reps, U)
        logLik_unit_out = np.array(unit_final_logliks)

        self.theta.theta = theta_list_out
        self.theta.logLik_unit = logLik_unit_out
        self.theta.estimation_scale = False

        execution_time = time.time() - start_time

        result = PanelPompMIFResult(
            method="mif",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_in,
            shared_traces=shared_da,
            unit_traces=unit_da,
            logLiks=full_logliks,
            J=J,
            M=M,
            rw_sd=rw_sd,
            a=a,
            thresh=thresh,
            block=block,
        )

        self.results_history.add(result)

    def train(
        self,
        J: int,
        M: int,
        eta: dict[str, float] | float,
        chunk_size: Union[int, str] = 1,
        optimizer: str = "Adam",
        alpha: float = 0.97,
        key: jax.Array | None = None,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
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
            eta (dict[str, float] | float): Learning rate(s). Can be a float for a
                global learning rate or a dictionary mapping parameter names to rates.
            chunk_size (Union[int, str], optional): Number of units to process
                per gradient calculation step. 'auto' will attempt to estimate
                concurrency based on hardware.
            optimizer (str, optional): Optimizer type. Supported: 'Adam', 'SGD'.
            alpha (float, optional): Learning rate decay factor per iteration.
            key (jax.Array, optional): JAX PRNG key. If None, uses the
                `fresh_key` attribute.
            theta (PanelParameters, optional): Initial parameter estimates.
                If None, uses the current `theta` attribute.

        Returns:
            None: Updates `self.theta` and appends result to `self.results_history`.
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

        # Determine chunk size similar to pfilter
        # This is experimental and should maybe be removed
        if chunk_size == "auto":
            try:
                import psutil

                bytes_per_unit = (
                    J * len(rep_unit.statenames) * len(rep_unit.ys.index) * 200
                )  # rough estimate
                mem = psutil.virtual_memory()
                avail = mem.available * 0.4
                max_units = max(1, int(avail / bytes_per_unit))
                chunk_size = min(U, max_units)
                try:
                    device = jax.devices()[0]
                    if device.platform == "gpu":
                        avail = (
                            device.memory_stats()["bytes_limit"]
                            - device.memory_stats()["bytes_in_use"]
                        )
                        max_units = max(1, int(avail * 0.4 / bytes_per_unit))
                        chunk_size = min(U, max_units)
                except Exception:
                    pass
            except Exception:
                chunk_size = max(1, U // 4)
        else:
            chunk_size = int(chunk_size)

        if chunk_size < 1:
            chunk_size = 1

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

        theta_list = theta_obj_in.theta
        if not theta_obj_in.estimation_scale:
            theta_list = [
                {
                    "shared": t["shared"].copy() if t["shared"] is not None else None,
                    "unit_specific": (
                        t["unit_specific"].copy()
                        if t["unit_specific"] is not None
                        else None
                    ),
                }
                for t in theta_list
            ]
            theta_trans_list = rep_unit.par_trans.panel_transform_list(
                theta_list, direction="to_est"
            )
        else:
            theta_trans_list = theta_list

        shared_trans_list = [t.get("shared") for t in theta_trans_list]
        spec_trans_list = [t.get("unit_specific") for t in theta_trans_list]

        shared = [t.get("shared") for t in theta_list]
        unit_specific = [t.get("unit_specific") for t in theta_list]

        if all(df is None for df in shared_trans_list):
            shared_array = jnp.zeros((n_reps, 0))
            shared_index: list[str] = []
        else:
            shared_index = self.canonical_shared_param_names
            shared_trans_list_nonnull = cast(list[pd.DataFrame], shared_trans_list)
            shared_array = jnp.stack(
                [
                    self._dataframe_to_array_canonical(
                        df, self.canonical_shared_param_names, "shared"
                    )
                    for df in shared_trans_list_nonnull
                ],
                axis=0,
            )

        if all(df is None for df in spec_trans_list):
            unit_array = jnp.zeros((n_reps, 0, U))
            spec_index: list[str] = []
        else:
            spec_index = self.canonical_unit_param_names
            spec_trans_list_nonnull = cast(list[pd.DataFrame], spec_trans_list)
            unit_array = jnp.stack(
                [
                    jnp.stack(
                        [
                            self._dataframe_to_array_canonical(
                                df, self.canonical_unit_param_names, unit
                            )
                            for unit in unit_names
                        ],
                        axis=1,
                    )
                    for df in spec_trans_list_nonnull
                ],
                axis=0,
            )

        eta_dict = (
            eta
            if isinstance(eta, dict)
            else {p: eta for p in self.canonical_param_names}
        )
        eta_shared = jnp.array(
            [eta_dict.get(p, 0.0) for p in shared_index], dtype=float
        )
        eta_spec = jnp.array([eta_dict.get(p, 0.0) for p in spec_index], dtype=float)

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )
        n_obs = ys_per_unit.shape[1]

        keys = jax.random.split(key, n_reps * M * U)
        keys = keys.reshape((n_reps, M, U) + keys.shape[1:])

        (
            logliks_history,
            shared_history,
            unit_history,
        ) = _vmapped_panel_train_internal(
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
            n_obs,
            U,
        )

        shared_traces_in = None
        unit_traces_in = None

        if len(shared_index) > 0:
            shared_ll_expanded = np.expand_dims(np.array(logliks_history), axis=-1)
            shared_traces_in = np.concatenate(
                [shared_ll_expanded, np.array(shared_history)], axis=-1
            )

        if len(spec_index) > 0:
            nan_ll = np.full((n_reps, M + 1, 1, U), np.nan, dtype=float)
            unit_traces_in = np.concatenate([nan_ll, np.array(unit_history)], axis=-2)

        shared_traces, unit_traces = rep_unit.par_trans.transform_panel_traces(
            shared_traces=shared_traces_in,
            unit_traces=unit_traces_in,
            shared_param_names=shared_index,
            unit_param_names=spec_index,
            unit_names=unit_names,
            direction="from_est",
        )

        if shared_traces is None:
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_ll = np.expand_dims(np.array(logliks_history), axis=-1)
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
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": jnp.arange(shared_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": shared_vars,
            },
        )
        unit_da = xr.DataArray(
            unit_traces,
            dims=["replicate", "iteration", "variable", "unit"],
            coords={
                "replicate": jnp.arange(unit_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": unit_vars,
                "unit": unit_names,
            },
        )

        if shared and shared[0] is not None:
            shared_list_out = [
                pd.DataFrame(
                    shared_traces[rep, -1, 1:].reshape(-1, 1),
                    index=pd.Index(shared_index),
                    columns=pd.Index(["shared"]),
                )
                for rep in range(shared_traces.shape[0])
            ]
        else:
            shared_list_out = None

        if unit_specific and unit_specific[0] is not None:
            specific_list_out = [
                pd.DataFrame(
                    unit_traces[rep, -1, 1:, :],
                    index=pd.Index(spec_index),
                    columns=pd.Index(unit_names),
                )
                for rep in range(unit_traces.shape[0])
            ]
        else:
            specific_list_out = None

        theta_list_out = [
            {
                "shared": shared_list_out[rep] if shared_list_out else None,
                "unit_specific": specific_list_out[rep] if specific_list_out else None,
            }
            for rep in range(n_reps)
        ]

        logLik_unit_out = np.full((n_reps, U), np.nan)

        self.theta.theta = theta_list_out
        self.theta.logLik_unit = logLik_unit_out
        self.theta.estimation_scale = False

        execution_time = time.time() - start_time

        result = PanelPompTrainResult(
            method="train",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_in,
            shared_traces=shared_da,
            unit_traces=unit_da,
            logLiks=xr.DataArray(  # Placeholder as we don't have unit logliks separated
                np.full((n_reps, U + 1), np.nan),
                dims=["replicate", "unit"],
                coords={
                    "replicate": jnp.arange(n_reps),
                    "unit": ["shared"] + unit_names,
                },
            ),
            J=J,
            M=M,
            eta=eta,
            optimizer=optimizer,
            alpha=alpha,
        )

        self.results_history.add(result)

    def arma_benchmark(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        log_ys: bool = False,
        suppress_warnings: bool = True,
    ) -> pd.DataFrame:
        """
        Fits an independent ARIMA model to the observation data for each unit and returns
        a DataFrame with the estimated log-likelihoods for each unit and the total.

        This is a wrapper around `pypomp.benchmarks.arma_benchmark`.

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
                    llf = _arma_benchmark(
                        unit.ys, order=order, log_ys=log_ys, suppress_warnings=False
                    )
                    results.append({"unit": name, "logLik": llf})
                    total_llf += llf

            if len(w) > 0:
                warnings.warn(
                    f"arma_benchmark: {len(w)} warnings were produced by statsmodels across units. "
                    "Set suppress_warnings=False to see the raw output.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            for name, unit in self.unit_objects.items():
                llf = _arma_benchmark(
                    unit.ys, order=order, log_ys=log_ys, suppress_warnings=False
                )
                results.append({"unit": name, "logLik": llf})
                total_llf += llf

        # Insert total at the beginning
        results.insert(0, {"unit": "[[TOTAL]]", "logLik": total_llf})
        return pd.DataFrame(results)

    def negbin_benchmark(
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
                    llf = _negbin_benchmark(
                        unit.ys, autoregressive=autoregressive, suppress_warnings=False
                    )
                    results.append({"unit": name, "logLik": llf})
                    total_llf += llf

            if len(w) > 0:
                warnings.warn(
                    f"negbin_benchmark: {len(w)} warnings were produced by statsmodels across units. "
                    "Set suppress_warnings=False to see the raw output.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            for name, unit in self.unit_objects.items():
                llf = _negbin_benchmark(
                    unit.ys, autoregressive=autoregressive, suppress_warnings=False
                )
                results.append({"unit": name, "logLik": llf})
                total_llf += llf

        # Insert total at the beginning
        results.insert(0, {"unit": "[[TOTAL]]", "logLik": total_llf})
        return pd.DataFrame(results)
