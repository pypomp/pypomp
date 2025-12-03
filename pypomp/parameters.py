"""
This module defines the parameter classes for Pomp and PanelPomp models.
It handles input validation, standardization, and conversion to JAX arrays.
"""

from abc import ABC, abstractmethod
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
from typing import Any, Union, Optional, Literal
from .ParTrans_class import ParTrans


class ParameterSet(ABC):
    """
    Abstract base class for parameter sets used in POMP models.
    """

    @abstractmethod
    def to_jax_array(self, param_names: list[str], **kwargs) -> jax.Array:
        """
        Converts the parameters to a JAX array suitable for model functions.

        Args:
            param_names: A list of canonical parameter names expected by the model.
            **kwargs: Additional context required for conversion (e.g. unit names).

        Returns:
            A JAX array representing the parameters.
            - For Pomp: Shape (reps, n_params)
            - For PanelPomp: Shape (reps, n_units, n_params)
        """
        pass

    @abstractmethod
    def num_replicates(self) -> int:
        """Returns the number of parameter replicates (J)."""
        pass

    @abstractmethod
    def subset(self, indices: Union[int, list[int], slice]) -> "ParameterSet":
        """
        Returns a new ParameterSet containing only the specified replicate indices.
        """
        pass

    @abstractmethod
    def get_param_names(self) -> list[str] | tuple[list[str], list[str]]:
        """Returns the list of parameter names contained in this set."""
        pass


class PompParameters(ParameterSet):
    """
    Manages parameters for a standard Pomp model.
    Internal storage is a list of dictionaries.
    """

    def __init__(
        self,
        theta: Union[dict, list[dict], "PompParameters"] | None,
        logLik: np.ndarray | None = None,
        estimation_scale: bool = False,
    ):
        """
        Args:
            theta: A single dictionary, a list of dictionaries, or an existing
                   PompParameters object containing the parameter values.
            logLik: A numpy array of log-likelihoods.
            estimation_scale: Whether the parameters are in the estimation scale.
        """
        if theta is None:
            self._params = []
            self._logLik = np.full(0, np.nan)
            self._canonical_param_names = []
            self.estimation_scale = False
            return

        if isinstance(theta, PompParameters):
            # Copy constructor behavior (shallow copy of list)
            self._params = list(theta._params)
            self._logLik = (
                theta.logLik.copy()
                if logLik is None
                else self._format_logLik(logLik, len(self._params))
            )
            self._canonical_param_names = theta._canonical_param_names
            self.estimation_scale = theta.estimation_scale
            return

        # Normalize input to list of dicts
        if isinstance(theta, dict):
            theta = [theta]

        self._validate_raw(theta)
        self._params: list[dict] = theta
        self._canonical_param_names: list[str] = list(self._params[0].keys())
        self.estimation_scale: bool = estimation_scale
        self._logLik = self._format_logLik(logLik, len(self._params))

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

    def _validate_raw(self, theta: list[dict]):
        if not isinstance(theta, list):
            raise TypeError("theta must be a list of dictionaries")

        if len(theta) == 0:
            raise ValueError("theta cannot be empty")

        if not all(isinstance(t, dict) for t in theta):
            raise TypeError("All elements in theta must be dictionaries")

        # Check that all values of all dictionaries are single floats
        for i, t in enumerate(theta):
            for key, value in t.items():
                if not isinstance(value, float):
                    raise TypeError(
                        f"Parameter '{key}' at index {i} is not a float: got {type(value).__name__}"
                    )

        # Ensure all dicts have identical keys
        first_keys = set(theta[0].keys())
        for i, t in enumerate(theta[1:]):
            if set(t.keys()) != first_keys:
                raise ValueError(
                    f"Parameter set at index {i + 1} has different keys than the first set. "
                    f"Expected {first_keys}, got {set(t.keys())}"
                )

    def _child_PompParameters(
        self,
        theta: Union[dict, list[dict], "PompParameters"] | None = None,
        logLik: np.ndarray | None = None,
        estimation_scale: bool | None = None,
    ):
        """
        Make a new PompParameters object with current attributes as the default.
        """
        # Explicitly handle None to avoid ambiguous truth-value checks on arrays
        theta_sel = self._params if theta is None else theta
        estimation_scale_sel = (
            self.estimation_scale if estimation_scale is None else estimation_scale
        )
        logLik_sel = self._logLik if logLik is None else logLik
        return PompParameters(
            theta=theta_sel, logLik=logLik_sel, estimation_scale=estimation_scale_sel
        )

    def to_jax_array(self, param_names: list[str], **kwargs) -> jax.Array:
        """
        Convert to JAX array matching the order of param_names.
        Returns shape (n_reps, n_params).
        """
        # Logic formerly in _theta_dict_to_array
        try:
            ordered_values = [[t[name] for name in param_names] for t in self._params]
        except KeyError as e:
            raise KeyError(
                f"Parameter {e} expected by model but missing from parameter set."
            )

        return jnp.array(ordered_values)

    @property
    def logLik(self) -> np.ndarray:
        return self._logLik

    @logLik.setter
    def logLik(self, value):
        self._logLik = self._format_logLik(value, self.num_replicates())

    def to_jax_array_canonical(self) -> jax.Array:
        return self.to_jax_array(self._canonical_param_names)

    def num_replicates(self) -> int:
        return len(self._params)

    def num_params(self) -> int:
        return len(self._canonical_param_names)

    def subset(self, indices: Union[int, list[int], slice]) -> "PompParameters":
        if isinstance(indices, int):
            indices = [indices]

        # Determine subset based on type
        if isinstance(indices, slice):
            subset_params = self._params[indices]
            subset_logLik = self._logLik[indices]
        else:
            subset_params = [self._params[i] for i in indices]
            subset_logLik = self._logLik[indices]

        return self._child_PompParameters(subset_params, logLik=subset_logLik)

    def get_param_names(self) -> list[str]:
        if not self._params:
            return []
        return list(self._canonical_param_names)

    def to_list(self) -> list[dict]:
        """Returns the internal list of dictionaries."""
        return self._params

    def transform(
        self,
        par_trans: ParTrans,
        direction: Literal["to_est", "from_est"] | None = None,
    ):
        """
        Transform the parameters to or from the estimation parameter space.

        Args:
            par_trans: The parameter transformation object.
            direction: The direction of transformation. If None, the direction is determined by the estimation_scale attribute.
        """
        auto = direction is None
        if auto:
            direction = "from_est" if self.estimation_scale else "to_est"
        if direction not in ["to_est", "from_est"]:
            raise ValueError(f"Invalid direction: {direction}")
        if direction == "to_est" and self.estimation_scale is False:
            self._params = [par_trans.to_est(theta_i) for theta_i in self._params]
            self.estimation_scale = True
        elif direction == "from_est" and self.estimation_scale is True:
            self._params = [par_trans.from_est(theta_i) for theta_i in self._params]
            self.estimation_scale = False
        else:
            # If this statement is reached, the parameters are already in the correct estimation space. Nothing needs to be done.
            pass

    def prune(self, n: int = 1, refill: bool = True) -> None:
        """
        Replace internal parameter sets with the top `n` based on stored log-likelihoods.

        Args:
            n: Number of top parameter sets to keep.
            refill: If True, repeat the top `n` parameter sets to match the
                previous number of replicates. If False, keep only the `n` sets.
        """
        n_reps = self.num_replicates()
        if n_reps == 0:
            raise ValueError("No parameter sets available to prune.")
        if n < 1:
            raise ValueError("n must be at least 1.")

        if self._logLik is None or np.all(np.isnan(self._logLik)):
            raise ValueError("No valid log-likelihoods available to prune.")

        # Indices of top-n log-likelihoods (descending order)
        top_indices = self._logLik.argsort()[-n:][::-1]

        top_params = [self._params[i] for i in top_indices]
        top_logLik = self._logLik[top_indices]

        if refill:
            prev_len = n_reps
            repeats = (prev_len + n - 1) // n  # Ceiling division
            new_params = (top_params * repeats)[:prev_len]
            new_logLik = np.tile(top_logLik, repeats)[:prev_len]
        else:
            new_params = top_params
            new_logLik = top_logLik

        self._params = new_params
        self._logLik = new_logLik

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[dict, "PompParameters", float]:
        """
        Support indexing like theta[0] or theta[0:2] or theta[0]["param_name"].
        - Integer index: returns the dict at that position
        - Slice: returns a new PompParameters object
        """
        if isinstance(index, int):
            # Integer index: return the dict directly
            return self._params[index]
        elif isinstance(index, slice):
            # Slice: return a new PompParameters object
            return self._child_PompParameters(self._params[index])
        else:
            raise TypeError(f"Invalid index: {index}. Must be an integer or slice.")

    def __iter__(self):
        """Support iteration over parameter sets."""
        return iter(self._params)

    def __len__(self) -> int:
        """Return the number of parameter replicates."""
        return len(self._params)

    def __mul__(self, n: int) -> "PompParameters":
        """
        Support replication like theta * 3.
        Returns a new PompParameters with n copies of the parameter sets.
        """
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("Multiplication factor must be non-negative")
        if n == 0:
            raise ValueError("Cannot create empty PompParameters")
        # Replicate the parameter sets n times
        replicated_params = self._params * n
        replicated_logLik = np.tile(self._logLik, n)
        return self._child_PompParameters(replicated_params, logLik=replicated_logLik)

    def __rmul__(self, n: int) -> "PompParameters":
        """Support left multiplication like 3 * theta."""
        return self.__mul__(n)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PompParameters(n_replicates={len(self._params)}, n_params={len(self._canonical_param_names)})"

    def __eq__(self, other) -> bool:
        """
        Check equality with another PompParameters object.
        Two PompParameters are equal if their canonical parameter names and parameter sets are equal.
        """
        if not isinstance(other, type(self)):
            return False
        # Compare canonical parameter names
        if self._canonical_param_names != other._canonical_param_names:
            return False
        # Compare parameter lists
        if len(self._params) != len(other._params):
            return False
        for p1, p2 in zip(self._params, other._params):
            if p1 != p2:
                return False
        # Check same scale
        if self.estimation_scale != other.estimation_scale:
            return False
        return True


class PanelParameters(ParameterSet):
    """
    Manages parameters for PanelPomp models.
    Internal storage is a list of dictionaries, always containing "shared" and "unit_specific" keys mapping to DataFrames (which may be empty).
    """

    def __init__(
        self,
        theta: Union[
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            "PanelParameters",
            None,
        ],
        logLik_unit: np.ndarray | None = None,
        estimation_scale: bool = False,
    ):
        self._theta: list[dict[str, pd.DataFrame | None]]
        self.estimation_scale: bool
        self._logLik_unit: np.ndarray
        self._canonical_shared_param_names: list[str]
        self._canonical_unit_param_names: list[str]
        self._canonical_param_names: list[str]

        if isinstance(theta, PanelParameters):
            self._theta = [
                {k: v.copy() if v is not None else None for k, v in t.items()}
                for t in theta._theta
            ]
            self.estimation_scale = theta.estimation_scale
            self._validate_none_consistency()
            self._validate_df_consistency()
            self._logLik_unit = (
                theta.logLik_unit.copy()
                if logLik_unit is None
                else self._format_logLik_unit(logLik_unit, len(self._theta))
            )
        else:
            self._theta = self._normalize_input(theta)
            self.estimation_scale = estimation_scale
            self._validate_none_consistency()
            self._validate_df_consistency()
            self._logLik_unit = self._format_logLik_unit(logLik_unit, len(self._theta))

        self._logLik = self._logLik_unit.sum(axis=1)

        shared_df = self._theta[0]["shared"]
        unit_df = self._theta[0]["unit_specific"]
        if shared_df is not None:
            self._canonical_shared_param_names = list(shared_df.index)
        else:
            self._canonical_shared_param_names = []
        if unit_df is not None:
            self._canonical_unit_param_names = list(unit_df.index)
        else:
            self._canonical_unit_param_names = []

        self._canonical_param_names = list(
            set(self._canonical_shared_param_names + self._canonical_unit_param_names)
        )

    def _normalize_input(
        self,
        theta: None
        | dict[str, pd.DataFrame | None]
        | list[dict[str, pd.DataFrame | None]],
    ) -> list[dict[str, pd.DataFrame | None]]:
        """
        Normalize input to list of dicts with valid DataFrames or None.
        Checks that all dictionaries have the keys "shared" and "unit_specific" and that all values are None or pd.DataFrames.
        """
        if theta is None:
            return []

        if isinstance(theta, dict):
            theta = [theta]

        if not isinstance(theta, list):
            raise TypeError("theta must be a dictionary or a list of dictionaries")

        for i, t in enumerate(theta):
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

        return theta.copy()

    def _validate_none_consistency(self):
        """
        Sets internal flags for whether all or only some 'shared'/'unit_specific' are None.
        """
        shared_none = [t["shared"] is None for t in self._theta]
        unit_none = [t["unit_specific"] is None for t in self._theta]

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

    def _format_logLik_unit(
        self, ll_unit: np.ndarray | None, n_reps: int
    ) -> np.ndarray:
        """Standardize logLik dimensions."""
        # Determine n_units from the first valid unit_specific dataframe
        n_units = 0
        if n_reps > 0 and self._theta[0]["unit_specific"] is not None:
            n_units = self._theta[0]["unit_specific"].shape[1]

        if ll_unit is None:
            return np.full((n_reps, n_units), np.nan)

        ll_unit = np.array(ll_unit, dtype=float)
        if ll_unit.ndim == 1 and n_reps == 1:
            return ll_unit.reshape(1, -1)
        if ll_unit.shape != (n_reps, n_units):
            # Allow shape (n_reps, 0) if n_units is 0
            if n_units == 0 and ll_unit.size == 0:
                return np.empty((n_reps, 0))
            raise ValueError(
                f"logLik_unit shape mismatch: {ll_unit.shape} vs ({n_reps}, {n_units})"
            )
        return ll_unit

    def _validate_df_consistency(self):
        """
        Ensure all replicates have consistent data frames:
        - Shared parameters must have the same index and exactly one column.
        - Unit-specific parameters must have the same index and columns.
        - If a parameter is in shared, it must not be in unit-specific and vice-versa.
        """
        if not self.theta:
            return

        ref = self.theta[0]
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

        for i, t in enumerate(self.theta[1:], 1):
            if t["shared"] is not None:
                if not t["shared"].index.equals(ref_s_idx):
                    raise ValueError(
                        f"Shared parameter index mismatch at replicate {i}."
                    )
            if t["unit_specific"] is not None:
                if not t["unit_specific"].index.equals(ref_u_idx):
                    raise ValueError(f"Unit parameter index mismatch at replicate {i}.")
                if not t["unit_specific"].columns.equals(ref_u_cols):
                    raise ValueError(f"Unit columns mismatch at replicate {i}.")

    @property
    def logLik(self) -> np.ndarray:
        return self._logLik

    @logLik.setter
    def logLik(self, value):
        # We generally don't set full logLik directly for panels, but strictly:
        # We can't infer unit contribution, so this setter is ambiguous
        # unless we just broadcast/reset. For now, assume read-only derived.
        pass

    @property
    def logLik_unit(self) -> np.ndarray:
        return self._logLik_unit

    @logLik_unit.setter
    def logLik_unit(self, value):
        self._logLik_unit = self._format_logLik_unit(value, len(self.theta))
        self._logLik = self._logLik_unit.sum(axis=1)

    @property
    def theta(self):
        return self._theta.copy()

    @theta.setter
    def theta(
        self,
        value: dict[str, pd.DataFrame | None] | list[dict[str, pd.DataFrame | None]],
    ):
        self._theta = self._normalize_input(value)
        self._validate_none_consistency()
        self._validate_df_consistency()
        n_reps = len(value)
        self._logLik_unit = self._format_logLik_unit(None, n_reps)
        self._logLik = self._logLik_unit.sum(axis=1)

    def num_replicates(self) -> int:
        return len(self._theta)

    def get_param_names(self) -> list[str]:
        return self._canonical_param_names

    def get_shared_param_names(self) -> list[str]:
        return self._canonical_shared_param_names

    def get_unit_param_names(self) -> list[str]:
        return self._canonical_unit_param_names

    def subset(self, indices: Union[int, list[int], slice]) -> "PanelParameters":
        if isinstance(indices, int):
            indices = [indices]

        # Slicing handled by list/array slicing logic
        if isinstance(indices, slice):
            sub_theta = self._theta[indices]
            sub_ll = self._logLik_unit[indices]
        else:
            sub_theta = [self._theta[i] for i in indices]
            sub_ll = self._logLik_unit[indices]

        return PanelParameters(
            sub_theta, logLik_unit=sub_ll, estimation_scale=self.estimation_scale
        )

    def to_jax_array(
        self, param_names: list[str], unit_names: list[str] | None = None, **kwargs
    ) -> jax.Array:
        reps = len(self._theta)
        if reps == 0:
            return jnp.empty((0, 0, 0))

        # Infer unit names if needed
        if unit_names is None:
            if self._theta[0]["unit_specific"] is not None:
                unit_names = list(self._theta[0]["unit_specific"].columns)
            else:
                raise ValueError(
                    "unit_names required when no unit_specific parameters exist"
                )

        n_units = len(unit_names)
        n_params = len(param_names)

        # Identify source of each parameter
        ref = self._theta[0]
        if ref["shared"] is not None:
            shared_keys = set(ref["shared"].index)
        else:
            shared_keys = set()
        if ref["unit_specific"] is not None:
            specific_keys = set(ref["unit_specific"].index)
        else:
            specific_keys = set()

        full_array = np.zeros((reps, n_units, n_params))

        for j, t in enumerate(self._theta):
            if t["shared"] is not None:
                s_df = t["shared"]
            else:
                s_df = pd.DataFrame()
            if t["unit_specific"] is not None:
                u_df = t["unit_specific"]
            else:
                u_df = pd.DataFrame()

            for p_idx, p_name in enumerate(param_names):
                if p_name in specific_keys:
                    # u_df cannot be empty if p_name is in specific_keys
                    try:
                        full_array[j, :, p_idx] = u_df.loc[p_name, unit_names].values
                    except KeyError:
                        raise KeyError(f"Unit mismatch for parameter {p_name}")
                elif p_name in shared_keys:
                    # s_df cannot be empty if p_name is in shared_keys
                    val = s_df.loc[p_name].iloc[0]
                    full_array[j, :, p_idx] = val
                else:
                    raise KeyError(f"Parameter '{p_name}' not found.")

        return jnp.array(full_array)

    def transform(
        self,
        par_trans: ParTrans,
        direction: Literal["to_est", "from_est"] | None = None,
    ):
        auto = direction is None
        if auto:
            direction = "from_est" if self.estimation_scale else "to_est"

        if (direction == "to_est" and not self.estimation_scale) or (
            direction == "from_est" and self.estimation_scale
        ):
            self._theta = par_trans.panel_transform_list(
                self._theta, direction=direction
            )
            self.estimation_scale = not self.estimation_scale

    def prune(self, n: int = 1, refill: bool = True) -> None:
        if not self.theta:
            return

        # Sort by total log likelihood
        top_indices = self._logLik.argsort()[-n:][::-1]

        top_theta = [self.theta[i] for i in top_indices]
        top_ll_unit = self._logLik_unit[top_indices]

        if refill:
            n_reps = len(self.theta)
            repeats = (n_reps + n - 1) // n
            self.theta = (top_theta * repeats)[:n_reps]
            self._logLik_unit = np.tile(top_ll_unit, (repeats, 1))[:n_reps]
        else:
            self.theta = top_theta
            self._logLik_unit = top_ll_unit

        self._logLik = self._logLik_unit.sum(axis=1)

    def mix_and_match(self, unit_names: list[str]) -> None:
        if not self.theta:
            return

        # Rank by shared logLik (total)
        shared_ranks = self._logLik.argsort()[::-1]

        # Rank by unit-specific logLik
        unit_ranks = {}
        for u_idx, unit in enumerate(unit_names):
            unit_ranks[unit] = self._logLik_unit[:, u_idx].argsort()[::-1]

        new_theta = []
        new_ll_unit = np.zeros_like(self._logLik_unit)

        for i in range(len(self.theta)):
            # 1. Best shared params for this position
            s_idx = shared_ranks[i]
            best_shared = self.theta[s_idx]["shared"].copy()

            # 2. Best unit params for each unit for this position
            new_u_data = {}
            for u_idx, unit in enumerate(unit_names):
                u_best_idx = unit_ranks[unit][i]

                # Copy the logLik for this unit/replicate combo
                new_ll_unit[i, u_idx] = self._logLik_unit[u_best_idx, u_idx]

                # Extract the unit specific column
                src_df = self.theta[u_best_idx]["unit_specific"]
                if not src_df.empty and unit in src_df.columns:
                    new_u_data[unit] = src_df[unit].copy()

            # Construct new unit dataframe
            if new_u_data:
                # Use index from the first theta (guaranteed consistent)
                if self._theta[0]["unit_specific"] is not None:
                    new_u_df = pd.DataFrame(
                        new_u_data, index=self._theta[0]["unit_specific"].index
                    )
                else:
                    new_u_df = None
            else:
                new_u_df = None

            new_theta.append({"shared": best_shared, "unit_specific": new_u_df})

        self._theta = new_theta
        self._logLik_unit = new_ll_unit
        self._logLik = new_ll_unit.sum(axis=1)

    def to_list(self) -> list[dict[str, pd.DataFrame | None]]:
        return self._theta.copy()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._theta[index]
        return self.subset(index)

    def __iter__(self):
        return iter(self._theta)

    def __len__(self):
        return len(self._theta)

    def __mul__(self, n: int) -> "PanelParameters":
        """Replicate the parameter set n times."""
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("n must be non-negative")

        # Replicate the internal list of dicts
        new_theta = self._theta * n

        # Replicate the logLik array
        if self._logLik_unit.size > 0:
            new_ll_unit = np.tile(self._logLik_unit, (n, 1))
        else:
            # Handle edge case of empty params or 0 replicates
            n_cols = self._logLik_unit.shape[1] if self._logLik_unit.ndim > 1 else 0
            new_ll_unit = np.empty((len(new_theta), n_cols))

        return PanelParameters(
            new_theta, logLik_unit=new_ll_unit, estimation_scale=self.estimation_scale
        )

    def __rmul__(self, n: int) -> "PanelParameters":
        """Support left multiplication (e.g. 5 * params)."""
        return self.__mul__(n)
