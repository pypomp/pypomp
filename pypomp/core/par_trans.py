from typing import Any, Callable, Literal, Mapping, cast, TypeVar
from ..types import ParamDict
import importlib
import jax
import jax.numpy as jnp
import numpy as np

TArray = TypeVar("TArray", bound=np.ndarray | jax.Array)
TShared = TypeVar("TShared", bound=np.ndarray | jax.Array | None)
TUnit = TypeVar("TUnit", bound=np.ndarray | jax.Array | None)


class ParTrans:
    """Parameter transformations between natural and estimation scales.

    Enables numerical algorithms to switch between operating in an unconstrained
    estimation parameter space (e.g. log-transformed positive parameters,
    logit-transformed probabilities) and running the model in the natural parameter
    space.

    Parameters
    ----------
    to_est : callable or None, optional
        A function mapping a parameter dictionary to the estimation scale.
        If ``None``, defaults to the identity transformation.
    from_est : callable or None, optional
        A function mapping a parameter dictionary from the estimation scale
        back to the natural scale.  If ``None``, defaults to the identity
        transformation.
    """

    to_est: Callable[[ParamDict], ParamDict]
    """The parameter transformation function to the estimation parameter space."""
    from_est: Callable[[ParamDict], ParamDict]
    """The parameter transformation function from the estimation parameter space to the natural parameter space."""

    def __init__(
        self,
        to_est: Callable[[ParamDict], ParamDict] | None = None,
        from_est: Callable[[ParamDict], ParamDict] | None = None,
    ):
        self.to_est: Callable[[ParamDict], ParamDict] = to_est or _to_est_default
        self.from_est: Callable[[ParamDict], ParamDict] = from_est or _from_est_default

    def _get_transform_fn(
        self, direction: Literal["to_est", "from_est"]
    ) -> Callable[[ParamDict], ParamDict]:
        """
        Validate direction and return the corresponding transformation function.
        """
        if direction not in ("to_est", "from_est"):
            raise ValueError(f"Invalid direction: {direction}")
        return self.to_est if direction == "to_est" else self.from_est

    def _to_floats(
        self,
        theta: Mapping[str, float | jax.Array],
        direction: Literal["to_est", "from_est"],
    ) -> dict[str, float]:
        """
        Convert the theta dictionary values from jax.Array to float.
        """
        func = self._get_transform_fn(direction)
        theta_out = func(dict(theta))
        return {k: float(v) for k, v in theta_out.items()}

    def _transform_array(
        self,
        param_array: TArray,
        param_names: list[str],
        direction: Literal["to_est", "from_est"],
    ) -> TArray:
        """
        Transform a parameter array to or from the estimation parameter space.
        Supports both NumPy and JAX arrays, preserving the input type.

        Args:
            param_array: Array of parameter values with shape (..., n_params)
            param_names: List of parameter names in the order in which the parameters appear in the array
            direction: Direction of transformation ("to_est" or "from_est")

        Returns:
            Transformed parameter array with the same shape and type as input
        """
        if direction not in ("to_est", "from_est"):
            raise ValueError(f"Invalid direction: {direction}")

        is_numpy = isinstance(param_array, np.ndarray)
        arr = jnp.asarray(param_array) if is_numpy else param_array

        original_shape = arr.shape
        n_params = original_shape[-1]
        if n_params == 0:
            return param_array

        param_array_2d = arr.reshape(-1, n_params)
        transform_fn = self._get_transform_fn(direction)

        def transform_single_row(row):
            param_dict = dict(zip(param_names, row))
            transformed_dict = transform_fn(param_dict)
            return jnp.stack([transformed_dict[name] for name in param_names])

        transform_vectorized = jax.vmap(transform_single_row)
        transformed_jax = transform_vectorized(param_array_2d)
        res = transformed_jax.reshape(original_shape)

        return cast(Any, np.array(res) if is_numpy else res)

    def _transform_panel_array(
        self,
        shared_array: TShared,
        unit_array: TUnit,
        shared_names: list[str],
        unit_specific_names: list[str],
        direction: Literal["to_est", "from_est"],
    ) -> tuple[TShared, TUnit]:
        """
        Transform shared and unit-specific parameters for a panel model.

        Supports both NumPy and JAX arrays, automatically preserving the input type
        (e.g., returning NumPy arrays if input contains NumPy arrays).

        Args:
            shared_array: Array of shared parameters. Shape: (..., n_shared).
                Can be None if there are no shared parameters.
            unit_array: Array of unit-specific parameters. Shape: (..., n_units, n_spec).
                Can be None if there are no unit-specific parameters.
            shared_names: List of shared parameter names in the order in which they appear in the array.
            unit_specific_names: List of unit-specific parameter names in the order in which they appear in the array.
            direction: Direction of transformation, either "to_est" (natural to estimation
                space) or "from_est" (estimation to natural space).

        Returns:
            A tuple (transformed_shared, transformed_unit) of the same types and shapes
            as the inputs. If an input array is None, the corresponding output is also None.
        """
        if direction not in ("to_est", "from_est"):
            raise ValueError(f"Invalid direction: {direction}")

        n_shared = len(shared_names)
        n_spec = len(unit_specific_names)

        if (n_shared == 0 and n_spec == 0) or (
            shared_array is None and unit_array is None
        ):
            return shared_array, unit_array

        is_numpy = (
            shared_array is not None and isinstance(shared_array, np.ndarray)
        ) or (unit_array is not None and isinstance(unit_array, np.ndarray))

        shared_jax = jnp.asarray(shared_array) if shared_array is not None else None
        unit_jax = jnp.asarray(unit_array) if unit_array is not None else None

        if unit_jax is not None:
            U = unit_jax.shape[-2]
            batch_shape = unit_jax.shape[:-2]
        elif shared_jax is not None:
            U = 1
            batch_shape = shared_jax.shape[:-1]
        else:
            U = 1
            batch_shape = ()

        # Shared broadcast array of shape (..., U, n_shared)
        if n_shared > 0:
            if shared_jax is not None:
                shared_jax_sliced = shared_jax[..., :n_shared]
                shared_broadcast = jnp.expand_dims(shared_jax_sliced, axis=-2)
                shared_broadcast = jnp.repeat(shared_broadcast, U, axis=-2)
            else:
                shared_broadcast = jnp.zeros(batch_shape + (U, n_shared))
        else:
            shared_broadcast = jnp.zeros(batch_shape + (U, 0))

        # Unit-specific array of shape (..., U, n_spec)
        if n_spec > 0:
            if unit_jax is not None:
                unit_broadcast = unit_jax[..., :n_spec]
            else:
                unit_broadcast = jnp.zeros(batch_shape + (U, n_spec))
        else:
            unit_broadcast = jnp.zeros(batch_shape + (U, 0))

        combined = jnp.concatenate([shared_broadcast, unit_broadcast], axis=-1)
        all_names = shared_names + unit_specific_names

        combined_transformed = self._transform_array(combined, all_names, direction)

        res_shared = None
        if shared_array is not None:
            res_shared = (
                combined_transformed[..., 0, :n_shared]
                if n_shared > 0
                else shared_array
            )

        res_unit = None
        if unit_array is not None:
            res_unit = (
                combined_transformed[..., n_shared:] if n_spec > 0 else unit_array
            )

        if is_numpy:
            res_shared = np.array(res_shared) if res_shared is not None else None
            res_unit = np.array(res_unit) if res_unit is not None else None

        return cast(Any, (res_shared, res_unit))

    def __eq__(self, other):
        """
        Check equality with another ParTrans object.

        Two ParTrans instances are equal if they use the same function objects
        for to_est and from_est. Note that functionally identical lambda functions
        will not be considered equal unless they are the same object.
        """
        if not isinstance(other, type(self)):
            return False
        if self.to_est != other.to_est:
            return False
        if self.from_est != other.from_est:
            return False
        return True

    def __getstate__(self):
        """
        Custom pickling method to preserve function identity.

        Stores module and function names for module-level functions.
        Lambdas/closures cannot be reliably reconstructed and will fall back
        to defaults on unpickling.
        """
        return {
            **_serialize_func(self.to_est, "to_est"),
            **_serialize_func(self.from_est, "from_est"),
        }

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct functions.

        Reconstructs module-level functions by importing them.
        Falls back to defaults for lambdas/closures.
        """
        self.to_est = _restore_func(state, "to_est", _to_est_default)
        self.from_est = _restore_func(state, "from_est", _from_est_default)


def _serialize_func(func, name: str) -> dict[str, str | bool]:
    if (
        hasattr(func, "__module__")
        and hasattr(func, "__name__")
        and func.__module__ is not None
    ):
        return {f"_{name}_module": func.__module__, f"_{name}_name": func.__name__}
    return {f"_{name}_is_lambda": True}


def _restore_func(state: dict, name: str, default_func) -> Callable:
    if f"_{name}_is_lambda" in state:
        return default_func
    try:
        module = importlib.import_module(state[f"_{name}_module"])
        return getattr(module, state[f"_{name}_name"])
    except (ImportError, AttributeError):
        return default_func


def _to_est_default(
    theta: ParamDict,
) -> ParamDict:
    return dict(theta)


def _from_est_default(
    theta: ParamDict,
) -> ParamDict:
    return dict(theta)
