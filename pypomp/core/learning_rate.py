from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, Mapping, Sequence, Any


class LearningRate:
    """Represent the learning rate schedule for model parameters during training.

    This class encapsulates learning rate values for each parameter, which can be
    either constant values or time-varying schedules (1D arrays of length M).
    It provides utility methods to generate common decay schedules such as
    cosine, geometric, and linear decay.

    Parameters
    ----------
    rates : Mapping[str, Union[float, list[float], np.ndarray]]
        Learning rates keyed by parameter name. Can be a single float,
        a list of floats, or a numpy array.

    Examples
    --------
    >>> import pypomp as pp
    >>> rates = pp.LearningRate({"beta": 0.1, "rho": 0.01})
    >>> rates = pp.LearningRate({"beta": [0.1, 0.2], "rho": [0.01, 0.02]})
    >>> rates = pp.LearningRate({"beta": np.array([0.1, 0.2]), "rho": np.array([0.01, 0.02])})
    """

    param_names: tuple[str, ...]
    """Tuple of parameter names, defining the array column ordering (a PyTree aux metadata)."""
    rates_all_arr: np.ndarray
    """Array of learning rate values or schedules (a PyTree leaf)."""

    def __init__(self, rates: Mapping[str, Union[float, list[float], np.ndarray]]):
        param_names, rates_arr = self._validate_rates(rates)
        object.__setattr__(self, "param_names", param_names)
        object.__setattr__(self, "rates_all_arr", rates_arr)

    @classmethod
    def _from_leaves(
        cls, param_names: tuple[str, ...], rates_all_arr: Any
    ) -> LearningRate:
        """Rebuild an instance directly from leaves + aux (bypasses validation).

        Used by PyTree unflattening and array transformation helpers.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "param_names", tuple(param_names))
        object.__setattr__(obj, "rates_all_arr", rates_all_arr)
        return obj

    @staticmethod
    def _validate_rates(
        rates: Mapping[str, Union[float, list[float], np.ndarray]],
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Validates the learning rates and returns prepared parameter names and array."""
        if not isinstance(rates, Mapping):
            raise ValueError("rates must be a Mapping (e.g., dict)")

        param_names = tuple(rates.keys())
        parsed_vals: list[float | np.ndarray] = []
        lengths: set[int] = set()

        for name, value in rates.items():
            if not isinstance(name, str):
                raise ValueError("All keys in rates must be strings (parameter names)")
            if isinstance(value, (int, float, np.number)):
                parsed_vals.append(float(value))
            elif isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"Learning rate schedule for '{name}' must be 1D")
                parsed_vals.append(arr)
                lengths.add(arr.size)
            else:
                raise TypeError(
                    f"Learning rate for '{name}' must be float or 1D sequence, "
                    f"got {type(value).__name__}"
                )

        if len(lengths) > 1:
            raise ValueError(
                f"All 1D learning rate schedules must have the same length. Got lengths: {sorted(lengths)}"
            )

        if not parsed_vals:
            return (), np.empty((0,), dtype=float)

        if lengths:
            M = lengths.pop()
            cols = [
                np.full(M, val, dtype=float) if isinstance(val, float) else val
                for val in parsed_vals
            ]
            rates_arr = np.column_stack(cols)
        else:
            rates_arr = np.asarray(parsed_vals, dtype=float)

        return param_names, rates_arr

    def _mismatched_param(self, target_M: int) -> str:
        """Find parameter with time-varying schedule for error messages."""
        arr = np.asarray(self.rates_all_arr)
        for i, name in enumerate(self.param_names):
            if arr.ndim == 2 and not np.all(arr[:, i] == arr[0, i]):
                return name
        return self.param_names[0] if self.param_names else "parameter"

    def _canonicalize(self, canonical_names: Sequence[str]) -> LearningRate:
        """Reorder learning rates to match canonical parameter names.

        Parameters
        ----------
        canonical_names : sequence of str
            The model's canonical parameter names.

        Returns
        -------
        LearningRate
            A new instance whose array columns match ``canonical_names``.
        """
        names = tuple(canonical_names)
        idx = {n: i for i, n in enumerate(self.param_names)}
        for n in names:
            if n not in idx:
                raise ValueError(f"Parameter '{n}' not found in learning rates")

        order = [idx[n] for n in names]
        arr = self.rates_all_arr
        new_arr = arr[order] if arr.ndim == 1 else arr[:, order]
        return LearningRate._from_leaves(names, new_arr)

    def to_array(self, param_names: list[str], M: int) -> jax.Array:
        """Convert the learning rates into a JAX array.

        Parameters
        ----------
        param_names : list of str
            Parameter names in canonical order.
        M : int
            Number of iterations in the training schedule.

        Returns
        -------
        jax.Array
            A 2D array of shape ``(M, n_params)`` where each column is the
            learning rate schedule for a parameter.
        """
        M_eff = max(M, 1)
        canonical = self._canonicalize(param_names)
        arr = canonical.rates_all_arr

        if arr.ndim == 1:
            schedule = jnp.tile(arr, (M_eff, 1))
        else:
            if arr.shape[0] != M:
                p_name = canonical._mismatched_param(M)
                raise ValueError(
                    f"Learning rate schedule for '{p_name}' has length {arr.shape[0]}, expected M={M}"
                )
            schedule = arr

        return jnp.asarray(schedule)

    def _apply_decay(self, factor: np.ndarray, M: int, decay_name: str) -> LearningRate:
        """Apply a 1D decay factor across all parameter rates."""
        arr = np.asarray(self.rates_all_arr)
        if arr.ndim == 2 and arr.shape[0] != M:
            p_name = self._mismatched_param(M)
            raise ValueError(
                f"Cannot apply {decay_name} decay of length {M} to schedule of length {arr.shape[0]} for '{p_name}'"
            )

        new_arr = factor[:, None] * (arr[None, :] if arr.ndim == 1 else arr)
        return LearningRate._from_leaves(self.param_names, new_arr)

    def cosine_decay(self, final_factor: float, M: int) -> LearningRate:
        """Apply a cosine cooling schedule to all current rates.

        Parameters
        ----------
        final_factor : float
            Multiplier to reach at the end of the schedule.  Must be in the
            interval :math:`[0, 1]`.
        M : int
            Number of iterations for the schedule.

        Returns
        -------
        LearningRate
            A new learning rate object with cosine decay applied.
        """
        if not (0 <= final_factor <= 1):
            raise ValueError("final_factor should be between 0 and 1")
        iterations = np.arange(M)
        factor = final_factor + (1.0 - final_factor) * 0.5 * (
            1.0 + np.cos(np.pi * iterations / M)
        )
        return self._apply_decay(factor, M, "cosine")

    def geometric_decay(self, decay_rate: float, M: int) -> LearningRate:
        """Apply a geometric decay schedule.

        The rate at step ``t`` is ``eta_t = eta_0 * (decay_rate ^ t)``.

        Parameters
        ----------
        decay_rate : float
            Decay factor per iteration.  Must be in the interval :math:`[0, 1]`.
        M : int
            Number of iterations for the schedule.

        Returns
        -------
        LearningRate
            A new learning rate object with geometric decay applied.
        """
        if not (0 <= decay_rate <= 1):
            raise ValueError("decay_rate should be between 0 and 1")
        return self._apply_decay(decay_rate ** np.arange(M), M, "geometric")

    def linear_decay(self, final_factor: float, M: int) -> LearningRate:
        """Apply a linear decay schedule.

        Linearly interpolates learning rates from their initial values down to
        initial values multiplied by ``final_factor``.

        Parameters
        ----------
        final_factor : float
            Multiplier to reach at the end of the schedule.  Must be in the
            interval :math:`[0, 1]`.
        M : int
            Number of iterations for the schedule.

        Returns
        -------
        LearningRate
            A new learning rate object with linear decay applied.
        """
        if not (0 <= final_factor <= 1):
            raise ValueError("final_factor should be between 0 and 1")
        return self._apply_decay(np.linspace(1.0, final_factor, M), M, "linear")

    @property
    def rates(self) -> dict[str, Union[float, np.ndarray]]:
        """Dictionary mapping parameter names to learning rate values or schedules."""
        arr = np.asarray(self.rates_all_arr)
        if arr.ndim == 1:
            return {n: float(v) for n, v in zip(self.param_names, arr)}
        return {
            n: (
                float(arr[0, i])
                if arr.shape[0] > 0 and np.all(arr[:, i] == arr[0, i])
                else arr[:, i]
            )
            for i, n in enumerate(self.param_names)
        }

    def __getitem__(self, param_name: str) -> Union[float, np.ndarray]:
        if param_name not in self.param_names:
            raise KeyError(f"Parameter '{param_name}' not found in learning rates.")
        idx = self.param_names.index(param_name)
        val = self.rates_all_arr
        if val.ndim == 1:
            return float(val[idx])
        return (
            float(val[0, idx])
            if val.shape[0] > 0 and np.all(val[:, idx] == val[0, idx])
            else val[:, idx]
        )

    def __contains__(self, param_name: str) -> bool:
        return param_name in self.param_names

    def __len__(self) -> int:
        return len(self.param_names)

    def __iter__(self):
        return iter(self.param_names)

    def keys(self):
        return self.rates.keys()

    def values(self):
        return self.rates.values()

    def items(self):
        return self.rates.items()

    def get(
        self, param_name: str, default: Union[float, np.ndarray, None] = None
    ) -> Union[float, np.ndarray, None]:
        if param_name in self.param_names:
            return self[param_name]
        return default

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.param_names != other.param_names:
            return False
        return bool(
            np.array_equal(
                np.asarray(self.rates_all_arr), np.asarray(other.rates_all_arr)
            )
        )

    def __str__(self) -> str:
        rate_strs = []
        for name, val in self.rates.items():
            if isinstance(val, (int, float, np.number)):
                rate_strs.append(f"'{name}': {val:.4g}")
            elif isinstance(val, np.ndarray):
                if val.size == 0:
                    rate_strs.append(f"'{name}': []")
                elif val.size <= 5:
                    vals_str = ", ".join(f"{x:.4g}" for x in val)
                    rate_strs.append(f"'{name}': [{vals_str}]")
                else:
                    rate_strs.append(
                        f"'{name}': [{val[0]:.4g} ... {val[-1]:.4g}] (len={val.size})"
                    )
            else:
                rate_strs.append(f"'{name}': {val}")

        indented_rates = "\n    ".join(rate_strs)
        return f"LearningRate(\n    {indented_rates}\n)"

    def __repr__(self) -> str:
        return self.__str__()


jax.tree_util.register_pytree_node(
    LearningRate,
    lambda lr: ((lr.rates_all_arr,), lr.param_names),
    lambda aux, children: LearningRate._from_leaves(aux, children[0]),
)
