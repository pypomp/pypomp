from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import numpy as np
import jax
import xarray as xr
from typing import (
    Union,
    Literal,
    Iterator,
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from ..par_trans import ParTrans

T_data = TypeVar("T_data", xr.DataArray, xr.Dataset)


class ParameterSet(ABC, Generic[T_data]):
    """
    Abstract base class for parameter sets used in POMP models.

    All parameter sets store parameters internally as a 3D ``xarray.DataArray``
    with dimensions ``("theta_idx", "unit", "parameter")``:

    - ``theta_idx``: Coordinate indexing each parameter set/replicate.
    - ``unit``: Coordinate indexing model units ("shared" or specific unit names).
    - ``parameter``: Coordinate indexing parameter names.
    """

    _data: T_data
    estimation_scale: bool

    @abstractmethod
    def to_jax_array(self, param_names: list[str] | None = None, **kwargs) -> jax.Array:
        """Convert the parameters to a JAX array suitable for model functions.

        Parameters
        ----------
        param_names : list of str or None, optional
            Canonical parameter names expected by the model.  If ``None``,
            defaults to the canonical order of parameters in the set.
        **kwargs : dict
            Additional context required for conversion (e.g. unit names).

        Returns
        -------
        jax.Array
            JAX array representing the parameters.  For ``Pomp`` models,
            shape is ``(n_reps, n_params)``.  For ``PanelPomp`` models, shape
            is ``(n_reps, n_units, n_params)``.
        """
        pass

    def num_replicates(self) -> int:
        """Returns the number of parameter sets/replicates."""
        return len(self)

    def num_params(self) -> int:
        """Return the number of canonical parameters."""
        return len(self.get_param_names())

    def get_param_names(self) -> list[str]:
        """Return the list of parameter names contained in this set."""
        if isinstance(self._data, xr.Dataset):
            shared = (
                list(self._data["shared"].coords["parameter"].values)
                if "shared" in self._data
                else []
            )
            unit_spec = (
                list(self._data["unit_specific"].coords["parameter"].values)
                if "unit_specific" in self._data
                else []
            )
            return sorted(list(set(shared + unit_spec)))
        return list(self._data.coords["parameter"].values)

    def __len__(self) -> int:
        """Return the number of parameter sets/replicates."""
        return self._data.sizes["theta_idx"]

    def __iter__(self) -> Iterator[Any]:
        """Support iteration over parameter sets."""
        return iter(self._to_list())

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k == "_data":
                setattr(new_obj, k, v.copy(deep=False))
            else:
                setattr(new_obj, k, v)
        return new_obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == "_data":
                setattr(new_obj, k, v.copy(deep=True))
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj

    def __mul__(self, n: int) -> Self:
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("Multiplication factor must be non-negative")
        if n == 0:
            raise ValueError("Cannot create empty ParameterSet")

        new_data = xr.concat(cast(Any, [self._data] * n), dim="theta_idx")
        new_data.coords["theta_idx"] = np.arange(new_data.sizes["theta_idx"])

        extra_kwargs = self._replicated_logLik(n)
        cls = cast(Any, self.__class__)
        return cls(new_data, estimation_scale=self.estimation_scale, **extra_kwargs)

    def __rmul__(self, n: int) -> Self:
        return self.__mul__(n)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{self._data.__repr__()}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(\n{self._data.__str__()}\n)"

    def pruned(self, n: int = 1, refill: bool = True) -> Self:
        """Return a new parameter set with the top `n` replicates by log-likelihood.

        Parameters
        ----------
        n : int, optional
            Number of top-performing parameter sets to keep.  Defaults to ``1``.
        refill : bool, optional
            If ``True``, duplicate the top ``n`` sets to restore the original
            number of replicates.  Defaults to ``True``.

        Returns
        -------
        Self
            A new parameter set containing the pruned replicates.
        """
        n_reps = self.num_replicates()
        if n_reps == 0:
            raise ValueError("No parameter sets available to prune.")
        if n < 1:
            raise ValueError("n must be at least 1.")

        log_lik = self.logLik
        if log_lik is None or np.all(np.isnan(log_lik)):
            if self.__class__.__name__ == "PompParameters":
                raise ValueError(
                    "No valid log-likelihoods available to prune (all nan)."
                )
            log_lik = np.zeros(n_reps)

        top_indices = log_lik.argsort()[-n:][::-1]

        if refill:
            prev_len = n_reps
            repeats = (prev_len + n - 1) // n
            new_indices = np.tile(top_indices, repeats)[:prev_len]
        else:
            new_indices = top_indices

        new_obj = copy.deepcopy(self)
        new_obj._data = new_obj._data.isel(theta_idx=new_indices)
        new_obj._data.coords["theta_idx"] = np.arange(len(new_indices))
        new_obj._slice_logLik(new_indices)
        return new_obj

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.estimation_scale != other.estimation_scale:
            return False
        if self.get_param_names() != other.get_param_names():
            return False
        if not self._data.equals(other._data):
            return False
        return self._eq_logLik(other)

    def __getitem__(self, index: int | slice | list[int]) -> Any:
        if isinstance(index, (slice, list, np.ndarray)):
            return self.subset(index)
        return self._getitem_int(int(index))

    def transformed(
        self,
        par_trans: ParTrans,
        direction: Literal["to_est", "from_est"] | None = None,
    ) -> Self:
        """Transform parameters between natural and estimation scales.

        Parameters
        ----------
        par_trans : ParTrans
            Parameter transformation mapping.
        direction : {"to_est", "from_est"} or None, optional
            Direction of the transformation.  If ``None`` (default), toggle the
            scale relative to the current ``estimation_scale`` attribute.

        Returns
        -------
        Self
            A new parameter set with the transformed parameters.
        """
        auto = direction is None
        if auto:
            direction = "from_est" if self.estimation_scale else "to_est"

        if (direction == "to_est" and not self.estimation_scale) or (
            direction == "from_est" and self.estimation_scale
        ):
            new_obj = copy.deepcopy(self)
            param_list = new_obj._to_list()
            new_obj._transform_and_load(par_trans, param_list, direction)
            new_obj.estimation_scale = not new_obj.estimation_scale
            return new_obj

        return copy.deepcopy(self)

    @overload
    def params(self, as_list: Literal[True]) -> list[Any]: ...

    @overload
    def params(self, as_list: Literal[False] = False) -> T_data: ...

    @overload
    def params(self, as_list: bool = False) -> list[Any] | T_data: ...

    def params(self, as_list: bool = False) -> list[Any] | T_data:
        """Get the parameter values in this parameter set.

        Parameters
        ----------
        as_list : bool, optional
            If ``True``, returns the parameters as a list of Python
            dictionaries.  If ``False`` (default), returns the internal xarray
            representation (DataArray or Dataset).

        Returns
        -------
        list of dict or xr.DataArray or xr.Dataset
            The parameters either as a list of dictionaries or as an xarray
            object.
        """
        if as_list:
            return self._to_list()
        return self._data

    @abstractmethod
    def set_params(self, value: Any) -> None:
        """
        Set or overwrite the parameter values.
        """
        pass

    @property
    @abstractmethod
    def logLik(self) -> np.ndarray:
        pass

    @abstractmethod
    def _to_list(self) -> list[Any]:
        pass

    @abstractmethod
    def subset(self, indices: Union[int, list[int], slice]) -> Self:
        pass

    @abstractmethod
    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def _slice_logLik(self, indices: np.ndarray) -> None:
        pass

    @abstractmethod
    def _eq_logLik(self, other: Any) -> bool:
        pass

    @abstractmethod
    def _getitem_int(self, index: int) -> Any:
        pass

    @abstractmethod
    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        pass
