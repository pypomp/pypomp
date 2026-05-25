import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, Mapping


class LearningRate:
    """
    Represent the learning rate schedule for model parameters during training.

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

    rates: dict[str, Union[float, np.ndarray]]
    """Dictionary mapping parameter names to learning rate values or schedules."""

    def __init__(self, rates: Mapping[str, Union[float, list[float], np.ndarray]]):
        self.rates = self._validate_rates(rates)

    def _validate_rates(
        self, rates: Mapping[str, Union[float, list[float], np.ndarray]]
    ) -> dict[str, Union[float, np.ndarray]]:
        """
        Validates the learning rates and returns a prepared dictionary.
        """
        if not isinstance(rates, Mapping):
            raise ValueError("rates must be a Mapping (e.g., dict)")

        validated = {}
        for param_name, value in rates.items():
            if not isinstance(param_name, str):
                raise ValueError("All keys in rates must be strings (parameter names)")

            if isinstance(value, (int, float, np.number)):
                validated[param_name] = float(value)
            elif isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Learning rate schedule for '{param_name}' must be 1D"
                    )
                validated[param_name] = arr
            else:
                raise TypeError(
                    f"Learning rate for '{param_name}' must be float or 1D sequence, "
                    f"got {type(value).__name__}"
                )

        return validated

    def to_array(self, param_names: list[str], M: int) -> jax.Array:
        """
        Convert the learning rates into a JAX array of shape (M, n_params).

        Parameters
        ----------
        param_names : list[str]
            List of parameter names in canonical order.
        M : int
            Number of iterations in the training schedule.

        Returns
        -------
        jax.Array
            A 2D array where each column is the learning rate schedule for a parameter.
        """
        n_params = len(param_names)
        M_eff = max(M, 1)
        schedule = np.zeros((M_eff, n_params), dtype=float)

        for i, name in enumerate(param_names):
            if name not in self.rates:
                raise ValueError(f"Parameter '{name}' not found in learning rates")

            val = self.rates[name]
            if isinstance(val, (float, int)):
                schedule[:, i] = float(val)
            elif isinstance(val, np.ndarray):
                if val.size != M:
                    raise ValueError(
                        f"Learning rate schedule for '{name}' has length {val.size}, expected M={M}"
                    )
                schedule[:, i] = val

        return jnp.array(schedule)

    def cosine_decay(self, final_factor: float, M: int) -> "LearningRate":
        """
        Apply a cosine cooling schedule to all current rates.

        Parameters
        ----------
        final_factor : float
            The factor to reach at the end of the schedule (between 0 and 1).
        M : int
            Number of iterations for the schedule.

        Returns
        -------
        LearningRate
            A new LearningRate object with cosine decay applied.
        """
        if not (0 <= final_factor <= 1):
            raise ValueError("final_factor should be between 0 and 1")

        iterations = np.arange(M)
        factor = final_factor + (1.0 - final_factor) * 0.5 * (
            1.0 + np.cos(np.pi * iterations / M)
        )

        new_rates = {}
        for name, val in self.rates.items():
            if isinstance(val, (float, int)):
                new_rates[name] = float(val) * factor
            elif isinstance(val, np.ndarray):
                # If it's already a schedule, multiply element-wise (assuming same M)
                if val.size != M:
                    raise ValueError(
                        f"Cannot apply cosine decay of length {M} to schedule of length {val.size} for '{name}'"
                    )
                new_rates[name] = val * factor

        return LearningRate(new_rates)

    def geometric_decay(self, decay_rate: float, M: int) -> "LearningRate":
        """
        Apply a geometric decay schedule: eta_t = eta_0 * (decay_rate ^ t).

        Parameters
        ----------
        decay_rate : float
            The decay rate per iteration (between 0 and 1).
        M : int
            Number of iterations for the schedule.

        Returns
        -------
        LearningRate
            A new LearningRate object with geometric decay applied.
        """
        if not (0 <= decay_rate <= 1):
            raise ValueError("decay_rate should be between 0 and 1")

        iterations = np.arange(M)
        factor = decay_rate**iterations

        new_rates = {}
        for name, val in self.rates.items():
            if isinstance(val, (float, int)):
                new_rates[name] = float(val) * factor
            elif isinstance(val, np.ndarray):
                if val.size != M:
                    raise ValueError(
                        f"Cannot apply geometric decay of length {M} to schedule of length {val.size} for '{name}'"
                    )
                new_rates[name] = val * factor

        return LearningRate(new_rates)

    def linear_decay(self, final_factor: float, M: int) -> "LearningRate":
        """
        Apply a linear decay schedule from 1.0 down to final_factor.

        Parameters
        ----------
        final_factor : float
            The factor to reach at the end of the schedule (between 0 and 1).
        M : int
            Number of iterations for the schedule.

        Returns
        -------
        LearningRate
            A new LearningRate object with linear decay applied.
        """
        if not (0 <= final_factor <= 1):
            raise ValueError("final_factor should be between 0 and 1")

        factor = np.linspace(1.0, final_factor, M)

        new_rates = {}
        for name, val in self.rates.items():
            if isinstance(val, (float, int)):
                new_rates[name] = float(val) * factor
            elif isinstance(val, np.ndarray):
                if val.size != M:
                    raise ValueError(
                        f"Cannot apply linear decay of length {M} to schedule of length {val.size} for '{name}'"
                    )
                new_rates[name] = val * factor

        return LearningRate(new_rates)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.rates.keys() != other.rates.keys():
            return False
        for k in self.rates:
            if not np.array_equal(self.rates[k], other.rates[k]):
                return False
        return True

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
                    rate_strs.append(f"'{name}': [{val[0]:.4g} ... {val[-1]:.4g}] (len={val.size})")
            else:
                rate_strs.append(f"'{name}': {val}")
        
        indented_rates = "\n    ".join(rate_strs)
        return f"LearningRate(\n    {indented_rates}\n)"


    def __repr__(self) -> str:
        return self.__str__()

