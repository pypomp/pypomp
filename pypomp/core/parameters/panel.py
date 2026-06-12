from __future__ import annotations
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
import xarray as xr
from typing import (
    Union,
    Literal,
    Any,
    overload,
)

from .base import ParameterSet
from ..par_trans import ParTrans


def _standardize_panel_theta(
    theta: Union[
        dict[str, pd.DataFrame | None],
        list[dict[str, pd.DataFrame | None]],
        None,
    ],
) -> tuple[xr.Dataset, list[str], list[str]]:
    if theta is None:
        shared_da = xr.DataArray(
            np.empty((0, 0)),
            dims=["theta_idx", "parameter"],
            coords={"theta_idx": [], "parameter": []},
        )
        unit_specific_da = xr.DataArray(
            np.empty((0, 0, 0)),
            dims=["theta_idx", "unit", "parameter"],
            coords={"theta_idx": [], "unit": [], "parameter": []},
        )
        ds = xr.Dataset(
            data_vars={
                "shared": shared_da,
                "unit_specific": unit_specific_da,
            }
        )
        return ds, [], []

    if isinstance(theta, dict):
        theta_list = [theta]
    else:
        theta_list = list(theta)

    if not isinstance(theta_list, list):
        raise TypeError("theta must be a dictionary or a list of dictionaries")

    # Copy the structures, convert to floats, and validate keys to avoid side-effects
    clean_theta = []
    for i, t in enumerate(theta_list):
        keys = set(t.keys())
        if keys != {"shared", "unit_specific"}:
            raise ValueError(
                f"Each parameter dictionary must have exactly the keys 'shared' and 'unit_specific'. "
                f"Found keys {keys} in item {i}."
            )
        if not all(isinstance(v, (pd.DataFrame, type(None))) for v in t.values()):
            raise TypeError(
                f"All values in each dictionary must be None or pd.DataFrames. "
                f"Found values {t.values()} of type {type(t.values())} in item {i}."
            )
        t_copy = {"shared": t["shared"], "unit_specific": t["unit_specific"]}
        if t_copy["shared"] is not None:
            t_copy["shared"] = t_copy["shared"].astype(float)
        if t_copy["unit_specific"] is not None:
            t_copy["unit_specific"] = t_copy["unit_specific"].astype(float)
        clean_theta.append(t_copy)

    # Consistency checks
    shared_none = [t["shared"] is None for t in clean_theta]
    unit_none = [t["unit_specific"] is None for t in clean_theta]

    some_shared_none = any(shared_none) and not all(shared_none)
    some_unit_specific_none = any(unit_none) and not all(unit_none)
    if some_shared_none:
        raise ValueError(
            "Some, but not all, shared parameters are None. This is not supported."
        )
    if some_unit_specific_none:
        raise ValueError(
            "Some, but not all, unit-specific parameters are None. This is not supported."
        )

    # Check dataframe consistency
    ref = clean_theta[0]
    if ref["shared"] is not None:
        ref_s_idx = ref["shared"].index
        ref_s_cols = ref["shared"].columns
        if len(ref_s_cols) != 1:
            raise ValueError("Shared parameters must have exactly one column.")
    else:
        ref_s_idx = []

    if ref["unit_specific"] is not None:
        ref_u_idx = ref["unit_specific"].index
        ref_u_cols = ref["unit_specific"].columns
    else:
        ref_u_idx = []
        ref_u_cols = []

    shared_param_names = set(ref_s_idx)
    unit_param_names = set(ref_u_idx)
    overlap = shared_param_names.intersection(unit_param_names)
    if overlap:
        raise ValueError(
            f"Parameter name(s) found in both shared and unit-specific parameters: {sorted(overlap)}"
        )

    for i, t in enumerate(clean_theta[1:], 1):
        if t["shared"] is not None:
            if not t["shared"].index.equals(ref_s_idx):
                raise ValueError(f"Shared parameter index mismatch at replicate {i}.")
        if t["unit_specific"] is not None:
            if not t["unit_specific"].index.equals(ref_u_idx):
                raise ValueError(f"Unit parameter index mismatch at replicate {i}.")
            if not t["unit_specific"].columns.equals(ref_u_cols):
                raise ValueError(f"Unit columns mismatch at replicate {i}.")

    # Gather names
    shared_names_list = list(ref_s_idx)
    unit_specific_names_list = list(ref_u_idx)

    if ref["unit_specific"] is not None:
        unit_names = list(ref_u_cols)
    else:
        unit_names = []

    reps = len(clean_theta)

    if ref["shared"] is not None:
        shared_values = np.stack(
            [t["shared"].loc[shared_names_list].iloc[:, 0].values for t in clean_theta]
        )
    else:
        shared_values = np.zeros((reps, 0))

    if ref["unit_specific"] is not None:
        unit_values = np.stack(
            [
                t["unit_specific"].loc[unit_specific_names_list, unit_names].T.values
                for t in clean_theta
            ]
        )
    else:
        unit_values = np.zeros((reps, len(unit_names), 0))

    shared_da = xr.DataArray(
        shared_values,
        dims=["theta_idx", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "parameter": shared_names_list,
        },
    )
    unit_specific_da = xr.DataArray(
        unit_values,
        dims=["theta_idx", "unit", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "unit": unit_names,
            "parameter": unit_specific_names_list,
        },
    )

    ds = xr.Dataset(
        data_vars={
            "shared": shared_da,
            "unit_specific": unit_specific_da,
        }
    )
    ds.attrs["shared_names"] = shared_names_list
    ds.attrs["unit_specific_names"] = unit_specific_names_list
    return ds, shared_names_list, unit_specific_names_list


class PanelParameters(ParameterSet[xr.Dataset]):
    """
    Manages parameters for PanelPomp models.

    Internal storage is a 3D ``xarray.DataArray`` with dimensions
    ``("theta_idx", "unit", "parameter")``.

    Parameters
    ----------
    theta : PanelParameters | dict | list | xr.DataArray, optional
        Parameters for the panel model. Accepts:

        - A single dictionary with ``"shared"`` and ``"unit_specific"`` keys (each containing a DataFrame).
        - A list of such dictionaries.
        - An existing :class:`~pypomp.core.parameters.PanelParameters` object.
        - An existing ``xarray.DataArray`` with dimensions ``("theta_idx", "unit", "parameter")``.
    logLik_unit : np.ndarray, optional
        A numpy array of unit-specific log-likelihoods of shape ``(num_theta_idx, n_units)``.
    estimation_scale : bool, optional
        Whether the parameters are on the estimation scale. Defaults to False.
    """

    _data: xr.Dataset
    estimation_scale: bool
    _logLik_unit: np.ndarray
    _logLik: np.ndarray
    _canonical_shared_param_names: list[str]
    _canonical_unit_param_names: list[str]
    _canonical_param_names: list[str]

    def __init__(
        self,
        theta: Union[
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            "PanelParameters",
            xr.Dataset,
            None,
        ],
        logLik_unit: np.ndarray | None = None,
        estimation_scale: bool = False,
    ):
        if isinstance(theta, PanelParameters):
            self._data = theta._data.copy(deep=True)
            self.estimation_scale = theta.estimation_scale
            self._canonical_shared_param_names = list(
                theta._canonical_shared_param_names
            )
            self._canonical_unit_param_names = list(theta._canonical_unit_param_names)
            self._canonical_param_names = list(theta._canonical_param_names)
            self._logLik_unit = (
                theta.logLik_unit.copy()
                if logLik_unit is None
                else self._format_logLik_unit(
                    logLik_unit, self._data.sizes["theta_idx"]
                )
            )
            self._logLik = self._logLik_unit.sum(axis=1)
            return

        if isinstance(theta, xr.Dataset):
            self._data = theta.copy(deep=True)
            raw_s = self._data.attrs.get("shared_names")
            if raw_s is None:
                raw_s = (
                    list(self._data["shared"].coords["parameter"].values)
                    if "shared" in self._data
                    else []
                )
            self._canonical_shared_param_names = [str(x) for x in raw_s]

            raw_u = self._data.attrs.get("unit_specific_names")
            if raw_u is None:
                raw_u = (
                    list(self._data["unit_specific"].coords["parameter"].values)
                    if "unit_specific" in self._data
                    else []
                )
            self._canonical_unit_param_names = [str(x) for x in raw_u]
        else:
            ds, s_names, u_names = _standardize_panel_theta(theta)
            self._data = ds
            self._canonical_shared_param_names = [str(x) for x in s_names]
            self._canonical_unit_param_names = [str(x) for x in u_names]

        self.estimation_scale = estimation_scale
        self._logLik_unit = self._format_logLik_unit(
            logLik_unit, self._data.sizes["theta_idx"]
        )
        self._logLik = self._logLik_unit.sum(axis=1)
        self._canonical_param_names = list(
            set(self._canonical_shared_param_names + self._canonical_unit_param_names)
        )

    def _format_logLik_unit(
        self, ll_unit: np.ndarray | None, n_reps: int
    ) -> np.ndarray:
        """Standardize logLik dimensions."""
        n_units = 0
        if n_reps > 0:
            n_units = len(self.get_unit_names())

        if ll_unit is None:
            return np.full((n_reps, n_units), np.nan)

        ll_unit = np.array(ll_unit, dtype=float)
        if ll_unit.ndim == 1 and n_reps == 1:
            return ll_unit.reshape(1, -1)
        if ll_unit.shape != (n_reps, n_units):
            if n_units == 0 and ll_unit.size == 0:
                return np.empty((n_reps, 0))
            raise ValueError(
                f"logLik_unit shape mismatch: {ll_unit.shape} vs ({n_reps}, {n_units})"
            )
        return ll_unit

    @property
    def logLik(self) -> np.ndarray:
        """
        Get the overall log-likelihood for each parameter set (theta_idx).
        """
        return self._logLik

    @logLik.setter
    def logLik(self, value):
        raise AttributeError(
            "Cannot set logLik directly on PanelParameters. "
            "Please assign unit-specific log-likelihoods to logLik_unit instead."
        )

    @property
    def logLik_unit(self) -> np.ndarray:
        """
        Get or set the unit-specific log-likelihoods for each parameter set (theta_idx).
        """
        return self._logLik_unit

    @logLik_unit.setter
    def logLik_unit(self, value):
        self._logLik_unit = self._format_logLik_unit(value, self.num_replicates())
        self._logLik = self._logLik_unit.sum(axis=1)

    def get_shared_param_names(self) -> list[str]:
        """Return the list of shared parameter names."""
        return self._canonical_shared_param_names

    def get_unit_param_names(self) -> list[str]:
        """Return the list of unit-specific parameter names."""
        return self._canonical_unit_param_names

    def get_unit_names(self) -> list[str]:
        """Return the list of unit names."""
        if (
            "unit_specific" in self._data
            and "unit" in self._data["unit_specific"].coords
        ):
            return list(self._data["unit_specific"].coords["unit"].values)
        return []

    def subset(self, indices: Union[int, list[int], slice]) -> "PanelParameters":
        """
        Return a new PanelParameters object with the specified parameter set (theta_idx) indices.
        """
        if isinstance(indices, int):
            indices = [indices]

        sub_data = self._data.isel(theta_idx=indices)
        sub_data.coords["theta_idx"] = np.arange(sub_data.sizes["theta_idx"])

        sub_ll = self._logLik_unit[indices]
        return PanelParameters(
            sub_data, logLik_unit=sub_ll, estimation_scale=self.estimation_scale
        )

    def to_jax_array(
        self,
        param_names: list[str] | None = None,
        unit_names: list[str] | None = None,
        **kwargs,
    ) -> jax.Array:
        """
        Convert to a JAX array matching the order of param_names and unit_names.

        Parameters
        ----------
        param_names : list[str], optional
            A list of parameter names in the desired order. If None (default),
            returns the array matching the canonical order of parameters.
        unit_names : list[str], optional
            A list of unit names in the desired order. If None (default),
            returns array for all units.

        Returns
        -------
        jax.Array
            A JAX array of shape (num_theta_idx, n_units, n_params).
        """
        if param_names is None:
            param_names = self.get_param_names()
        reps = self.num_replicates()
        if reps == 0:
            return jnp.empty((0, 0, 0))

        if unit_names is None:
            existing_units = self.get_unit_names()
            if not existing_units:
                raise ValueError(
                    "unit_names required when no unit_specific parameters exist"
                )
            unit_names = existing_units

        n_units = len(unit_names)
        n_params = len(param_names)

        shared_keys = set(self._canonical_shared_param_names)
        specific_keys = set(self._canonical_unit_param_names)
        existing_units = self.get_unit_names()

        # Pre-validate keys and units
        for p_name in param_names:
            if p_name not in shared_keys and p_name not in specific_keys:
                raise KeyError(f"Parameter '{p_name}' not found.")
            if p_name in specific_keys:
                for u in unit_names:
                    if u not in existing_units:
                        raise KeyError(f"Unit mismatch for parameter {p_name}")

        out_array = np.zeros((reps, n_units, n_params))

        for p_idx, p_name in enumerate(param_names):
            if p_name in specific_keys:
                out_array[:, :, p_idx] = (
                    self._data["unit_specific"]
                    .sel(parameter=p_name, unit=unit_names)
                    .values
                )
            else:  # p_name in shared_keys
                shared_vals = self._data["shared"].sel(parameter=p_name).values
                out_array[:, :, p_idx] = np.broadcast_to(
                    shared_vals[:, None], (reps, n_units)
                )

        return jnp.array(out_array)

    def mix_and_match(self) -> None:
        """
        Mixes unit-specific and shared parameters independently by sorting each
        unit's unit-specific parameters and the shared parameters in descending
        order of their respective log-likelihood contribution.
        """
        unit_names = self.get_unit_names()
        if self.num_replicates() == 0:
            return

        shared_ranks = self._logLik.argsort()[::-1]

        unit_ranks = {}
        for u_idx, unit in enumerate(unit_names):
            unit_ranks[unit] = self._logLik_unit[:, u_idx].argsort()[::-1]

        reps = self.num_replicates()

        shared_keys = self._canonical_shared_param_names
        specific_keys = self._canonical_unit_param_names

        # Reorder shared parameters
        new_shared_da = (
            self._data["shared"]
            .sel(parameter=shared_keys)
            .isel(theta_idx=shared_ranks)
            .assign_coords(theta_idx=np.arange(reps))
        )

        # Reorder unit-specific parameters and unit log-likelihoods
        new_unit_values = np.zeros((reps, len(unit_names), len(specific_keys)))
        new_ll_unit = np.zeros_like(self._logLik_unit)
        for u_idx, unit in enumerate(unit_names):
            best_idx = unit_ranks[unit]
            new_unit_values[:, u_idx, :] = (
                self._data["unit_specific"]
                .sel(unit=unit, parameter=specific_keys)
                .isel(theta_idx=best_idx)
                .values
            )
            new_ll_unit[:, u_idx] = self._logLik_unit[best_idx, u_idx]

        unit_specific_da = xr.DataArray(
            new_unit_values,
            dims=["theta_idx", "unit", "parameter"],
            coords={
                "theta_idx": np.arange(reps),
                "unit": unit_names,
                "parameter": specific_keys,
            },
        )
        self._data = xr.Dataset(
            data_vars={
                "shared": new_shared_da,
                "unit_specific": unit_specific_da,
            }
        )

        self._logLik_unit = new_ll_unit
        self._logLik = new_ll_unit.sum(axis=1)

    @overload
    def params(
        self, as_list: Literal[True] = True
    ) -> list[dict[str, pd.DataFrame | None]]: ...

    @overload
    def params(self, as_list: Literal[False]) -> xr.Dataset: ...

    @overload
    def params(
        self, as_list: bool = True
    ) -> list[dict[str, pd.DataFrame | None]] | xr.Dataset: ...

    def params(
        self, as_list: bool = True
    ) -> list[dict[str, pd.DataFrame | None]] | xr.Dataset:
        """
        Get the parameters in this set.

        Parameters
        ----------
        as_list : bool, default True
            If True, returns the parameters as a list of dictionaries with keys "shared" and "unit_specific".
            If False, returns the internal xarray Dataset.

        Returns
        -------
        list[dict[str, pd.DataFrame | None]] | xr.Dataset
            The parameters either as a list of dictionaries or as a Dataset.
        """
        return super().params(as_list)

    def set_params(
        self,
        value: dict[str, pd.DataFrame | None]
        | list[dict[str, pd.DataFrame | None]]
        | xr.Dataset,
    ) -> None:
        """
        Set or overwrite the parameter values.

        Parameters
        ----------
        value : dict[str, pd.DataFrame | None] | list[dict[str, pd.DataFrame | None]] | xr.Dataset
            The new panel parameter values. Accepts:
            - A single dictionary with ``"shared"`` and ``"unit_specific"`` keys (each containing a DataFrame).
            - A list of such dictionaries.
            - An existing :class:`xarray.Dataset` of panel parameters.
        """
        if value is None:
            raise ValueError("theta cannot be None")
        if isinstance(value, xr.Dataset):
            self._data = value.copy(deep=True)
            raw_s = self._data.attrs.get("shared_names")
            if raw_s is None:
                raw_s = (
                    list(self._data["shared"].coords["parameter"].values)
                    if "shared" in self._data
                    else []
                )
            s_names = [str(x) for x in raw_s]

            raw_u = self._data.attrs.get("unit_specific_names")
            if raw_u is None:
                raw_u = (
                    list(self._data["unit_specific"].coords["parameter"].values)
                    if "unit_specific" in self._data
                    else []
                )
            u_names = [str(x) for x in raw_u]
        else:
            self._data, s_names, u_names = _standardize_panel_theta(value)
            s_names = [str(x) for x in s_names]
            u_names = [str(x) for x in u_names]

        self._canonical_shared_param_names = s_names
        self._canonical_unit_param_names = u_names
        self._canonical_param_names = list(set(s_names + u_names))
        self._logLik_unit = self._format_logLik_unit(None, self.num_replicates())
        self._logLik = self._logLik_unit.sum(axis=1)

    def _to_list(self) -> list[dict[str, pd.DataFrame | None]]:
        """Return the parameter sets as a list of dictionaries with 'shared' and 'unit_specific' DataFrames."""
        reps = self.num_replicates()
        if reps == 0:
            return []

        shared_names = self._canonical_shared_param_names
        unit_specific_names = self._canonical_unit_param_names
        unit_names = self.get_unit_names()

        out = []
        for j in range(reps):
            t_dict = {}

            if shared_names:
                t_dict["shared"] = pd.DataFrame(
                    self._data["shared"]
                    .isel(theta_idx=j)
                    .sel(parameter=shared_names)
                    .values,
                    index=pd.Index(shared_names),
                    columns=["shared"],
                )
            else:
                t_dict["shared"] = None

            if unit_specific_names and unit_names:
                t_dict["unit_specific"] = pd.DataFrame(
                    self._data["unit_specific"]
                    .isel(theta_idx=j)
                    .sel(parameter=unit_specific_names, unit=unit_names)
                    .values.T,
                    index=pd.Index(unit_specific_names),
                    columns=pd.Index(unit_names),
                )
            else:
                t_dict["unit_specific"] = None

            out.append(t_dict)
        return out

    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        if self._logLik_unit.size > 0:
            new_ll_unit = np.tile(self._logLik_unit, (n, 1))
        else:
            new_ll_unit = np.empty((n * self.num_replicates(), 0))
        return {"logLik_unit": new_ll_unit}

    def _slice_logLik(self, indices: np.ndarray) -> None:
        self._logLik_unit = self._logLik_unit[indices]
        self._logLik = self._logLik_unit.sum(axis=1)

    def _eq_logLik(self, other: "PanelParameters") -> bool:
        if self._canonical_shared_param_names != other._canonical_shared_param_names:
            return False
        if self._canonical_unit_param_names != other._canonical_unit_param_names:
            return False
        return np.array_equal(self._logLik_unit, other._logLik_unit, equal_nan=True)

    def _getitem_int(self, index: int) -> dict[str, pd.DataFrame | None]:
        return self._to_list()[index]

    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        transformed_list = par_trans._panel_transform_list(
            param_list, direction=direction
        )
        ds, s_names, u_names = _standardize_panel_theta(transformed_list)
        self._data = ds
        self._canonical_shared_param_names = [str(x) for x in s_names]
        self._canonical_unit_param_names = [str(x) for x in u_names]


    @staticmethod
    def merge(*param_objs: "PanelParameters") -> "PanelParameters":
        """Merge replications from multiple PanelParameters objects."""
        if len(param_objs) == 0:
            raise ValueError("At least one PanelParameters object must be provided.")
        first = param_objs[0]

        for obj in param_objs:
            if not isinstance(obj, PanelParameters):
                raise TypeError("All merged objects must be of type PanelParameters.")
            if obj._canonical_shared_param_names != first._canonical_shared_param_names:
                raise ValueError(
                    "All PanelParameters objects must have the same canonical shared parameter names."
                )
            if obj._canonical_unit_param_names != first._canonical_unit_param_names:
                raise ValueError(
                    "All PanelParameters objects must have the same canonical unit parameter names."
                )
            if obj.estimation_scale != first.estimation_scale:
                raise ValueError(
                    "All PanelParameters objects must have the same estimation scale."
                )
            if obj.get_unit_names() != first.get_unit_names():
                raise ValueError(
                    "All PanelParameters objects must have the same unit names."
                )

        merged_data = xr.concat([obj._data for obj in param_objs], dim="theta_idx")
        merged_data.coords["theta_idx"] = np.arange(merged_data.sizes["theta_idx"])
        merged_data.attrs["shared_names"] = first._canonical_shared_param_names
        merged_data.attrs["unit_specific_names"] = first._canonical_unit_param_names

        all_logLik_unit = [obj._logLik_unit for obj in param_objs]
        merged_logLik_unit = (
            np.concatenate(all_logLik_unit, axis=0) if all_logLik_unit else np.array([])
        )
        return PanelParameters(
            merged_data,
            logLik_unit=merged_logLik_unit,
            estimation_scale=first.estimation_scale,
        )
