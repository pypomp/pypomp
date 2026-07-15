from __future__ import annotations
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
import xarray as xr
from typing import (
    Mapping,
    Sequence,
    Literal,
    Any,
    cast,
    overload,
)

from .base import ParameterSet
from ..par_trans import ParTrans


def _empty_unit_specific(n_reps: int) -> xr.DataArray:
    """Return the empty ``unit_specific`` array used by single-unit parameters."""
    return xr.DataArray(
        np.empty((n_reps, 0, 0)),
        dims=["theta_idx", "unit", "parameter"],
        coords={"theta_idx": np.arange(n_reps), "unit": [], "parameter": []},
    )


def _pomp_dataset(shared_da: xr.DataArray) -> xr.Dataset:
    """Wrap a ``(theta_idx, parameter)`` shared DataArray into the canonical
    parameter Dataset.

    A standard (non-panel) POMP has no units, so every parameter is stored in
    the ``shared`` variable and ``unit_specific`` is empty.  This is the same
    two-variable Dataset that :class:`PanelParameters` uses, which lets both
    classes share all replicate/log-likelihood machinery in the base class.
    """
    n_reps = shared_da.sizes["theta_idx"]
    param_names = [str(x) for x in shared_da.coords["parameter"].values]
    ds = xr.Dataset(
        data_vars={
            "shared": shared_da,
            "unit_specific": _empty_unit_specific(n_reps),
        }
    )
    ds.attrs["shared_names"] = param_names
    ds.attrs["unit_specific_names"] = []
    return ds


def _shared_from_dataarray(theta: xr.DataArray) -> xr.DataArray:
    """Normalize a user-supplied DataArray to a ``(theta_idx, parameter)`` array."""
    da = theta.astype(float)
    if da.ndim == 3:
        if set(da.dims) != {"theta_idx", "unit", "parameter"}:
            raise ValueError(
                "3D DataArray must have dims ('theta_idx', 'unit', 'parameter')"
            )
        da = da.transpose("theta_idx", "unit", "parameter").isel(unit=0, drop=True)
    elif da.ndim > 3:
        raise ValueError("DataArray must be 1D, 2D, or 3D")

    da = da.drop_vars(["theta_idx"], errors="ignore")
    if da.ndim == 1:
        if "parameter" not in da.dims:
            if len(da.dims) == 1:
                da = da.rename({da.dims[0]: "parameter"})
            else:
                raise ValueError("1D DataArray must have 'parameter' dimension")
        da = da.expand_dims(dim={"theta_idx": [0]}, axis=0)
    elif da.ndim == 2:
        if "parameter" not in da.dims:
            raise ValueError("2D DataArray must have 'parameter' dimension")
        if "theta_idx" not in da.dims:
            other_dim = [d for d in da.dims if d != "parameter"][0]
            da = da.rename({other_dim: "theta_idx"})
        da = da.transpose("theta_idx", "parameter")
    else:
        raise ValueError("DataArray must be 1D, 2D, or 3D")

    if "parameter" in da.coords:
        param_names: list[Any] = [str(x) for x in da.coords["parameter"].values]
    else:
        param_names = list(range(da.sizes["parameter"]))
    return xr.DataArray(
        np.asarray(da.values, dtype=float),
        dims=["theta_idx", "parameter"],
        coords={
            "theta_idx": np.arange(da.sizes["theta_idx"]),
            "parameter": param_names,
        },
    )


def _standardize_pomp_theta(
    theta: Mapping[str, float] | Sequence[Mapping[str, float]] | None,
) -> xr.Dataset:
    if theta is None:
        raise ValueError("theta cannot be None")

    theta_dicts: list[dict[str, float]] = []
    if isinstance(theta, Mapping):
        theta_dicts = [dict(theta)]
    elif isinstance(theta, (list, tuple)):
        theta_dicts = [dict(t) for t in cast(Any, theta)]
    else:
        try:
            theta_dicts = [dict(t) for t in cast(Any, theta)]
        except (TypeError, ValueError):
            raise TypeError(
                "theta must be a Mapping, Sequence of Mappings, or PompParameters"
            )

    # Validate elements are dictionaries and not empty
    if len(theta_dicts) == 0:
        raise ValueError("theta cannot be empty")
    if not all(isinstance(t, dict) for t in theta_dicts):
        raise TypeError("All elements in theta must be dictionaries")

    # Cast to floats (making copy to prevent side-effects)
    clean_dicts = []
    for i, t in enumerate(theta_dicts):
        t_copy = {}
        for key, value in t.items():
            if isinstance(value, bool):
                raise TypeError(
                    f"Parameter '{key}' at index {i} is not a float: got bool"
                )
            try:
                t_copy[key] = float(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Parameter '{key}' at index {i} is not a float: got {type(value).__name__}"
                )
        clean_dicts.append(t_copy)

    # Ensure all dicts have identical keys
    first_keys = set(clean_dicts[0].keys())
    for i, t in enumerate(clean_dicts[1:]):
        if set(t.keys()) != first_keys:
            raise ValueError(
                f"Parameter set at index {i + 1} has different keys than the first set. "
                f"Expected {first_keys}, got {set(t.keys())}"
            )

    reps = len(clean_dicts)
    param_names = list(clean_dicts[0].keys())

    # Contiguous array of shape (J, P), preserving insertion order of the keys.
    values = pd.DataFrame(clean_dicts)[param_names].values

    shared_da = xr.DataArray(
        values,
        dims=["theta_idx", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "parameter": param_names,
        },
    )
    return _pomp_dataset(shared_da)


class PompParameters(ParameterSet):
    """Parameter set for standard POMP models.

    Internally wraps an ``xarray.Dataset`` with two variables, ``shared``
    (dims ``("theta_idx", "parameter")``) and an empty ``unit_specific``.  A
    standard POMP has no units, so all parameters live in ``shared``.  This is
    the same representation used by :class:`PanelParameters`, allowing both to
    share the replicate/log-likelihood machinery defined in the base class.

    Parameters
    ----------
    theta : mapping or sequence of mapping or PompParameters or xr.DataArray or None
        Parameters for the model.  Accepts:

        - A single dictionary: ``dict[str, float]``
        - A sequence of dictionaries: ``list[dict[str, float]]``
        - An existing :class:`PompParameters` object
        - An ``xarray.DataArray`` (1D over parameters, 2D over
          ``(theta_idx, parameter)``, or 3D with a singleton ``unit``)
    logLik : np.ndarray, optional
        Log-likelihood values associated with each parameter set.
    estimation_scale : bool, optional
        Whether the parameters are on the estimation scale.  Defaults to
        ``False``.
    """

    _data: xr.Dataset
    estimation_scale: bool
    _logLik: np.ndarray

    def __init__(
        self,
        theta: Mapping[str, float]
        | Sequence[Mapping[str, float]]
        | PompParameters
        | xr.DataArray
        | xr.Dataset
        | None,
        logLik: np.ndarray | None = None,
        estimation_scale: bool = False,
    ):
        if theta is None:
            empty_shared = xr.DataArray(
                np.empty((0, 0)),
                dims=["theta_idx", "parameter"],
                coords={"theta_idx": [], "parameter": []},
            )
            self._data = _pomp_dataset(empty_shared)
            self._logLik = np.full(0, np.nan)
            self.estimation_scale = False
            return

        if isinstance(theta, PompParameters):
            self._data = theta._data.copy(deep=True)
            self._logLik = (
                theta.logLik.copy()
                if logLik is None
                else self._format_logLik(logLik, self._data.sizes["theta_idx"])
            )
            self.estimation_scale = theta.estimation_scale
            return

        if isinstance(theta, xr.Dataset):
            self._data = theta.copy(deep=True)
        elif isinstance(theta, xr.DataArray):
            self._data = _pomp_dataset(_shared_from_dataarray(theta))
        else:
            self._data = _standardize_pomp_theta(theta)

        self.estimation_scale = estimation_scale
        self._logLik = self._format_logLik(logLik, self._data.sizes["theta_idx"])

    def get_param_names(self) -> list[str]:
        """Return the parameter names in their original (insertion) order."""
        return [str(x) for x in self._data["shared"].coords["parameter"].values]

    def _format_logLik(self, ll: np.ndarray | None, n_reps: int) -> np.ndarray:
        """Helper to standardize logLik input."""
        if ll is None:
            return np.full(n_reps, np.nan)

        ll = np.array(ll, dtype=float)

        if ll.ndim == 0:  # Handle single scalar input (broadcast)
            return np.full(n_reps, ll)

        if len(ll) != n_reps:
            raise ValueError(
                f"Length of logLik ({len(ll)}) must match parameters ({n_reps})"
            )
        return ll

    def to_jax_array(self, param_names: list[str] | None = None, **kwargs) -> jax.Array:
        """Convert parameter values to a JAX array matching ``param_names``.

        Parameters
        ----------
        param_names : list of str, optional
            Parameter names in the desired order.  If ``None`` (default),
            returns the array in the canonical parameter order.
        **kwargs : dict
            Unused in standard pomp parameters.

        Returns
        -------
        jax.Array
            JAX array of shape ``(n_reps, n_params)``.
        """
        if param_names is None:
            param_names = self.get_param_names()
        try:
            ordered_values = self._data["shared"].sel(parameter=param_names).values
        except KeyError as e:
            raise KeyError(
                f"Parameter {e} expected by model but missing from parameter set."
            )

        return jnp.array(ordered_values)

    @property
    def logLik(self) -> np.ndarray:
        """
        Get or set the log-likelihoods for each parameter set (theta_idx).
        """
        return self._logLik

    @logLik.setter
    def logLik(self, value):
        self._logLik = self._format_logLik(value, self.num_replicates())

    @overload
    def params(self, as_list: Literal[True]) -> list[dict[str, float]]: ...

    @overload
    def params(self, as_list: Literal[False] = False) -> xr.DataArray: ...

    @overload
    def params(
        self, as_list: bool = False
    ) -> list[dict[str, float]] | xr.DataArray: ...

    def params(self, as_list: bool = False) -> list[dict[str, float]] | xr.DataArray:
        """Get the parameter values in this set.

        Parameters
        ----------
        as_list : bool, optional
            If ``True``, returns the parameters as a list of Python
            dictionaries.  If ``False`` (default), returns a 3D
            ``xarray.DataArray`` with dims ``("theta_idx", "unit", "parameter")``
            and a single ``"shared"`` unit.

        Returns
        -------
        list of dict or xr.DataArray
            The parameters either as a list of dictionaries mapping parameter
            names to floats, or as a DataArray.
        """
        if as_list:
            return cast(list[dict[str, float]], self._to_list())
        return self._data["shared"].expand_dims(dim={"unit": ["shared"]}, axis=1)

    def set_params(
        self,
        value: Mapping[str, float] | Sequence[Mapping[str, float]] | xr.DataArray,
    ) -> None:
        """Overwrite parameter values.

        Parameters
        ----------
        value : Mapping[str, float] | Sequence[Mapping[str, float]] | xr.DataArray
            The new parameter values. Accepts:
            - A single dictionary: ``dict[str, float]``
            - A list of dictionaries: ``list[dict[str, float]]`` (must have identical keys)
            - An ``xarray.DataArray`` of shape ``(theta_idx, unit, parameter)``
        """
        if value is None:
            raise ValueError("theta cannot be None")
        temp = PompParameters(value, estimation_scale=self.estimation_scale)
        self._data = temp._data
        self._logLik = self._format_logLik(None, self.num_replicates())

    def _to_list(self) -> list[dict[str, float]]:
        """Return the parameter sets as a list of dictionaries."""
        return cast(
            list[dict[str, float]],
            pd.DataFrame(
                self._data["shared"].values, columns=self.get_param_names()
            ).to_dict(orient="records"),
        )

    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        return {"logLik": np.tile(self._logLik, n)}

    def _slice_logLik(self, indices: np.ndarray) -> None:
        self._logLik = self._logLik[indices]

    def _eq_logLik(self, other: "PompParameters") -> bool:
        return np.array_equal(self._logLik, other._logLik, equal_nan=True)

    def _check_merge_compatible(self, other: Any) -> None:
        if other.get_param_names() != self.get_param_names():
            raise ValueError(
                "All PompParameters objects must have the same canonical parameter names."
            )

    @staticmethod
    def _concat_logLik(param_objs: tuple[Any, ...]) -> dict[str, np.ndarray]:
        return {"logLik": np.concatenate([obj._logLik for obj in param_objs])}

    def _getitem_int(self, index: int) -> dict[str, float]:
        return dict(zip(self.get_param_names(), self._data["shared"].values[index]))

    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        transformed_list = [
            par_trans._to_floats(theta_i, direction) for theta_i in param_list
        ]
        self._data = _standardize_pomp_theta(transformed_list)
