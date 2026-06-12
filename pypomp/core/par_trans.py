from typing import Callable, Literal, Mapping, cast
from ..types import ParamDict
import importlib
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


class ParTrans:
    """
    Handles parameter transformations between natural and estimation parameter spaces.
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

    def _panel_transform(
        self,
        theta: dict[str, pd.DataFrame | None],
        direction: Literal["to_est", "from_est"],
    ) -> dict[str, pd.DataFrame | None]:
        """
        Transform shared and unit-specific parameters for a single replicate.
        Input theta contains 'shared' and/or 'unit_specific' DataFrames.
        """
        func = self.to_est if direction == "to_est" else self.from_est

        s_df = theta.get("shared")
        u_df = theta.get("unit_specific")

        res: dict[str, pd.DataFrame | None] = {"shared": None, "unit_specific": None}

        # Pre-calculate shared dictionary (param -> value)
        s_vals = cast(dict, s_df.iloc[:, 0].to_dict()) if s_df is not None else {}

        # 1. Transform Shared Parameters
        if s_df is not None:
            # Context: Shared values + First unit's specific values (if any)
            ctx = s_vals.copy()
            if u_df is not None:
                first_unit = u_df.columns[0]
                ctx.update(cast(dict, u_df[first_unit].to_dict()))

            trans: ParamDict = func(cast(ParamDict, ctx))

            # Filter output back to just shared keys
            new_s_vals = {k: trans[k] for k in s_vals}
            res["shared"] = pd.DataFrame(
                list(new_s_vals.values()),
                index=s_df.index,
                columns=s_df.columns,
            )

        # 2. Transform Unit-Specific Parameters
        if u_df is not None:
            new_u_data = {}
            for unit in u_df.columns:
                # Context: Shared values + This unit's specific values
                ctx = s_vals.copy()
                ctx.update(cast(dict, u_df[unit].to_dict()))

                trans = func(cast(ParamDict, ctx))

                # Filter output back to specific keys (maintaining order)
                new_u_data[unit] = [trans[k] for k in u_df.index]

            res["unit_specific"] = pd.DataFrame(new_u_data, index=u_df.index)

        return res

    def _panel_transform_list(
        self,
        theta_list: list[dict[str, pd.DataFrame | None]],
        direction: Literal["to_est", "from_est"],
    ) -> list[dict[str, pd.DataFrame | None]]:
        """
        Apply transform to a list of parameter sets.
        """
        return [self._panel_transform(t, direction) for t in theta_list]

    def _to_floats(
        self,
        theta: Mapping[str, float | jax.Array],
        direction: Literal["to_est", "from_est"],
    ) -> dict[str, float]:
        """
        Convert the theta dictionary values from jax.Array to float.
        """
        if direction not in ("to_est", "from_est"):
            raise ValueError(f"Invalid direction: {direction}")

        func = self.to_est if direction == "to_est" else self.from_est
        theta_out = func(dict(theta))
        return {k: float(v) for k, v in theta_out.items()}

    def _transform_array(
        self,
        param_array: np.ndarray,
        param_names: list[str],
        direction: Literal["to_est", "from_est"],
    ) -> np.ndarray:
        """
        Transform a parameter array to or from the (unconstrained) estimation parameter space.

        This wrapper converts an array of parameters to a dict, applies the
        dict-to-dict transformation function, and converts back to an array.

        Args:
            param_array: Array of parameter values with shape (..., n_params)
            param_names: List of parameter names in the same order as the array
            direction: Direction of transformation ("to_est" or "from_est")

        Returns:
            Transformed parameter array with the same shape as input
        """
        if direction not in ["to_est", "from_est"]:
            raise ValueError(f"Invalid direction: {direction}")

        transform_fn = self.to_est if direction == "to_est" else self.from_est

        original_shape = param_array.shape

        if len(original_shape) == 1:
            param_array_2d = param_array.reshape(1, -1)
        else:
            param_array_2d = param_array.reshape(-1, original_shape[-1])

        def transform_single_row(row):
            param_dict = dict(zip(param_names, row))
            transformed_dict = transform_fn(param_dict)
            return jnp.array([transformed_dict[name] for name in param_names])

        transform_vectorized = jax.vmap(transform_single_row)

        param_jax = jnp.array(param_array_2d)
        transformed_jax = transform_vectorized(param_jax)
        transformed_array = np.array(transformed_jax)

        return transformed_array.reshape(original_shape)

    def _transform_panel_traces(
        self,
        shared_traces: np.ndarray | None,
        unit_traces: np.ndarray | None,
        shared_param_names: list[str],
        unit_param_names: list[str],
        direction: Literal["to_est", "from_est"],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Transform panel traces from estimation space to natural space.

        For panel models, shared and unit-specific parameters may be interdependent
        in the transformation, so they need to be transformed together.

        Args:
            shared_traces: Array of shared parameter traces, shape (n_reps, n_iters, n_shared+1)
                where [:, :, 0] is loglik and [:, :, 1:] are shared params
            unit_traces: Array of unit-specific parameter traces,
                shape (n_reps, n_iters, n_spec+1, n_units) where [:, :, 0, :] is per-unit loglik
                and [:, :, 1:, :] are unit-specific params
            shared_param_names: List of shared parameter names
            unit_param_names: List of unit-specific parameter names
            direction: Direction of transformation ("to_est" or "from_est")

        Returns:
            Tuple of (transformed_shared_traces, transformed_unit_traces) with same shapes as inputs
        """
        if direction not in ["to_est", "from_est"]:
            raise ValueError(f"Invalid direction: {direction}")

        if shared_traces is None and unit_traces is None:
            return None, None

        transform_fn = self.to_est if direction == "to_est" else self.from_est

        n_shared = len(shared_param_names)
        n_spec = len(unit_param_names)

        shared_out = None
        unit_out = None

        if shared_traces is not None and n_shared > 0:
            n_reps, n_iters, _ = shared_traces.shape
            shared_out = shared_traces.copy()

            def transform_shared_single(shared_vals, unit_vals_for_context):
                param_dict = dict(zip(shared_param_names, shared_vals))
                if n_spec > 0:
                    param_dict.update(zip(unit_param_names, unit_vals_for_context))
                transformed = transform_fn(param_dict)
                return jnp.array([transformed[name] for name in shared_param_names])

            transform_shared_vectorized = jax.vmap(jax.vmap(transform_shared_single))

            shared_params_only = jnp.array(shared_traces[:, :, 1:])

            if unit_traces is not None and n_spec > 0:
                unit_context = jnp.array(unit_traces[:, :, 1:, 0])
            else:
                unit_context = jnp.zeros((n_reps, n_iters, max(1, n_spec)))

            transformed_shared = transform_shared_vectorized(
                shared_params_only, unit_context
            )
            shared_out[:, :, 1:] = np.array(transformed_shared)

        if unit_traces is not None and n_spec > 0:
            n_reps, n_iters, _, n_units = unit_traces.shape
            unit_out = unit_traces.copy()

            def transform_unit_single(shared_vals_for_context, unit_vals):
                param_dict = dict(zip(unit_param_names, unit_vals))
                if n_shared > 0:
                    param_dict.update(zip(shared_param_names, shared_vals_for_context))
                transformed = transform_fn(param_dict)
                return jnp.array([transformed[name] for name in unit_param_names])

            # At the per-iteration slice, unit_vals has shape (n_spec, n_units),
            # so we need to vmap over axis=1 (units axis) here.
            vmap_over_units = jax.vmap(transform_unit_single, in_axes=(None, 1))
            vmap_over_iters = jax.vmap(vmap_over_units, in_axes=(0, 0))
            transform_unit_vectorized = jax.vmap(vmap_over_iters, in_axes=(0, 0))

            if shared_traces is not None and n_shared > 0:
                shared_context = jnp.array(shared_traces[:, :, 1:])
            else:
                shared_context = jnp.zeros((n_reps, n_iters, max(1, n_shared)))

            unit_params_only = jnp.array(unit_traces[:, :, 1:, :])

            transformed_unit = transform_unit_vectorized(
                shared_context, unit_params_only
            )
            # transformed shape: (n_reps, n_iters, n_units, n_spec)
            # target slice shape: (n_reps, n_iters, n_spec, n_units)
            transformed_unit = jnp.transpose(transformed_unit, (0, 1, 3, 2))
            unit_out[:, :, 1:, :] = np.array(transformed_unit)
        elif unit_traces is not None:
            unit_out = unit_traces.copy()

        return shared_out, unit_out

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
