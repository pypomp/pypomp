import jax
import jax.numpy as jnp


class RWSigma:
    sigmas: dict[str, float]
    init_names: list[str]
    not_init_names: list[str]
    all_names: list[str]

    """
    Class for representing the random walk sigma parameters for the iterated filtering (IF2) algorithm.

    Attributes:
        sigmas (dict[str, float]): Dictionary mapping parameter names to sigma values.
        init_names (list[str]): List of parameter names that are considered initial parameters.
        not_init_names (list[str]): List of parameter names that are not considered initial parameters.
        all_names (list[str]): List of all parameter names.
    """

    def __init__(self, sigmas: dict[str, float], init_names: list[str] = []):
        self.sigmas, self.init_names, self.not_init_names, self.all_names = (
            self._validate_attributes(sigmas, init_names)
        )

    def _validate_attributes(
        self, sigmas: dict[str, float], init_names: list[str]
    ) -> tuple[dict[str, float], list[str], list[str], list[str]]:
        """
        Validates the attributes of the RWSigma object and returns prepared attributes.
        """
        if not isinstance(sigmas, dict):
            raise ValueError("sigmas must be a dictionary")
        if not all(
            isinstance(sigmas[param_name], float) for param_name in sigmas.keys()
        ):
            raise ValueError("All values in sigmas dictionary must be floats")
        if not isinstance(init_names, list):
            raise ValueError("init_names must be a list")
        if not all(isinstance(param_name, str) for param_name in init_names):
            raise ValueError("All values in init_names list must be strings")
        if not all(param_name in sigmas.keys() for param_name in init_names):
            raise ValueError("All init_names names must be in sigmas dictionary")
        if len(init_names) != len(set(init_names)):
            raise ValueError("Duplicate names found in init_names")
        if not all(sigmas[param_name] >= 0 for param_name in sigmas.keys()):
            raise ValueError("All values in sigmas dictionary must be non-negative")

        not_init_names = [name for name in sigmas.keys() if name not in init_names]
        if len(not_init_names) != len(set(not_init_names)):
            raise ValueError("Duplicate names found in not_init_names")

        all_names = not_init_names + init_names
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate names found in all_names")

        return sigmas, init_names, not_init_names, all_names

    def _return_arrays(
        self, param_names: list[str] | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns the sigmas and sigmas_init arrays. If param_names is provided, only
        returns the arrays if the parameter names in the object match those in the
        param_names argument.

        Returns:
            sigmas_array: Array of sigmas for non-initial parameters. Shape (d,).
            Contains 0 for initial parameters.
            sigmas_init_array: Array of sigmas for initial parameters. Shape (d,).
            Contains 0 for non-initial parameters.
        """
        if param_names is None:
            param_names = self.all_names
        else:
            if not (
                all(param_name in self.all_names for param_name in param_names)
                and all(param_name in param_names for param_name in self.all_names)
            ):
                raise ValueError("All param_names must be in all_names and vice versa")

        all_sigmas_array = jnp.array(
            [self.sigmas[param_name] for param_name in param_names]
        )
        not_init_mask = jnp.array(
            [
                1 if param_name in self.not_init_names else 0
                for param_name in param_names
            ]
        )
        init_mask = jnp.array(
            [1 if param_name in self.init_names else 0 for param_name in param_names]
        )
        sigmas_array = all_sigmas_array * not_init_mask
        sigmas_init_array = all_sigmas_array * init_mask
        return sigmas_array, sigmas_init_array

    def cool(self, factor: float) -> None:
        """
        Reduces all sigmas by multiplying them by the specified factor in place.

        Args:
            factor (float): Value by which to multiply each sigma.

        Returns:
            None
        """
        if not (0 <= factor <= 1):
            raise ValueError("factor should be between 0 and 1")
        for key in self.sigmas:
            self.sigmas[key] *= factor

    def __setitem__(self, param_name: str, value: float) -> None:
        """
        Set the value of a sigma for a given parameter name using the indexing syntax.

        Args:
            param_name (str): The name of the parameter whose sigma value you wish to set.
            value (float): The new sigma value.

        Raises:
            KeyError: If param_name is not found in sigmas.
            TypeError: If value cannot be coerced to a float.
            ValueError: If the value is negative.
        """
        if param_name not in self.sigmas:
            raise KeyError(f"Parameter '{param_name}' not found in sigmas.")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                "Sigma value must be a float or numeric type that can be coerced to float."
            )
        if value < 0:
            raise ValueError("Sigma value must be non-negative.")

        self.sigmas[param_name] = value

    def __eq__(self, other) -> bool:
        """
        Check equality with another RWSigma object.

        Two RWSigma instances are equal if they have the same sigmas
        and init_names.
        """
        if not isinstance(other, type(self)):
            return False
        if self.sigmas != other.sigmas:
            return False
        if self.init_names != other.init_names:
            return False
        return True
