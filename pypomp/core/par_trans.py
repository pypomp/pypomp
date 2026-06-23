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

    def _get_transform_fn(
        self, direction: Literal["to_est", "from_est"]
    ) -> Callable[[ParamDict], ParamDict]:
        """
        Validate direction and return the corresponding transformation function.
        """
        if direction not in ("to_est", "from_est"):
            raise ValueError(f"Invalid direction: {direction}")
        return self.to_est if direction == "to_est" else self.from_est

    def _panel_transform(
        self,
        theta: dict[str, pd.DataFrame | None],
        direction: Literal["to_est", "from_est"],
    ) -> dict[str, pd.DataFrame | None]:
        """
        Transform shared and unit-specific parameters for a single replicate.
        Input theta contains 'shared' and/or 'unit_specific' DataFrames.
        """
        func = self._get_transform_fn(direction)

        s_df = theta.get("shared")
        u_df = theta.get("unit_specific")

        res: dict[str, pd.DataFrame | None] = {"shared": None, "unit_specific": None}

        # Pre-calculate shared dictionary (param -> value)
        s_vals = cast(dict, s_df.iloc[:, 0].to_dict()) if s_df is not None else {}

        def get_trans(u_vals_dict: dict) -> ParamDict:
            ctx = s_vals.copy()
            ctx.update(u_vals_dict)
            return func(cast(ParamDict, ctx))

        # 1. Transform Shared Parameters
        if s_df is not None:
            # Context: Shared values + First unit's specific values (if any)
            first_u_vals = (
                cast(dict, u_df.iloc[:, 0].to_dict()) if u_df is not None else {}
            )
            trans = get_trans(first_u_vals)

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
                unit_u_vals = cast(dict, u_df[unit].to_dict())
                trans = get_trans(unit_u_vals)

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
        func = self._get_transform_fn(direction)
        theta_out = func(dict(theta))
        return {k: float(v) for k, v in theta_out.items()}

    def _transform_array_jax(
        self,
        param_array: jax.Array,
        param_names: list[str],
        direction: Literal["to_est", "from_est"],
    ) -> jax.Array:
        """
        Transform a JAX parameter array to or from the estimation parameter space.
        """
        transform_fn = self._get_transform_fn(direction)

        original_shape = param_array.shape
        n_params = original_shape[-1]
        if n_params == 0:
            return param_array

        param_array_2d = param_array.reshape(-1, n_params)

        def transform_single_row(row):
            param_dict = dict(zip(param_names, row))
            transformed_dict = transform_fn(param_dict)
            return jnp.stack([transformed_dict[name] for name in param_names])

        transform_vectorized = jax.vmap(transform_single_row)
        transformed_jax = transform_vectorized(param_array_2d)

        return transformed_jax.reshape(original_shape)

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
        transformed_jax = self._transform_array_jax(
            jnp.array(param_array), param_names, direction
        )
        return np.array(transformed_jax)

    def _transform_panel_array_jax(
        self,
        shared_array: jax.Array,  # shape (..., n_shared)
        unit_array: jax.Array,  # shape (..., U, n_spec)
        shared_names: list[str],
        unit_specific_names: list[str],
        direction: Literal["to_est", "from_est"],
    ) -> tuple[jax.Array, jax.Array]:
        """
        Transform shared and unit-specific JAX array parameters using user-defined ParTrans.
        Handles arbitrary batch dimensions by flattening.
        """
        n_shared = len(shared_names)
        n_spec = len(unit_specific_names)

        if n_shared == 0 and n_spec == 0:
            return shared_array, unit_array

        orig_shared_shape = shared_array.shape
        orig_unit_shape = unit_array.shape

        # Flatten leading dimensions
        batch_size = 1
        for dim in orig_shared_shape[:-1]:
            batch_size *= dim

        shared_flat = (
            shared_array.reshape(-1, n_shared)
            if n_shared > 0
            else jnp.zeros((batch_size, 0))
        )
        unit_flat = (
            unit_array.reshape(-1, orig_unit_shape[-2], n_spec)
            if n_spec > 0
            else jnp.zeros((batch_size, orig_unit_shape[-2], 0))
        )

        func = self._get_transform_fn(direction)

        def transform_single_rep(s_val, u_vals):
            # s_val: shape (n_shared,)
            # u_vals: shape (U, n_spec)
            def get_trans(u_val):
                ctx = {}
                if n_shared > 0:
                    ctx.update(zip(shared_names, s_val))
                if n_spec > 0:
                    ctx.update(zip(unit_specific_names, u_val))
                return func(ctx)

            if n_shared > 0:
                first_u_val = u_vals[0] if n_spec > 0 else jnp.zeros(0)
                trans = get_trans(first_u_val)
                s_val_new = jnp.stack([trans[name] for name in shared_names])
            else:
                s_val_new = s_val

            if n_spec > 0:

                def transform_unit(u_val):
                    trans = get_trans(u_val)
                    return jnp.stack([trans[name] for name in unit_specific_names])

                u_vals_new = jax.vmap(transform_unit)(u_vals)
            else:
                u_vals_new = u_vals

            return s_val_new, u_vals_new

        shared_transformed, unit_transformed = jax.vmap(transform_single_rep)(
            shared_flat, unit_flat
        )

        return (
            shared_transformed.reshape(orig_shared_shape)
            if n_shared > 0
            else shared_array,
            unit_transformed.reshape(orig_unit_shape) if n_spec > 0 else unit_array,
        )

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

        n_shared = len(shared_param_names)
        n_spec = len(unit_param_names)

        shared_out = None
        unit_out = None

        # Determine batch dimensions
        if shared_traces is not None:
            n_reps = shared_traces.shape[0]
            n_iters = shared_traces.shape[1]
        elif unit_traces is not None:
            n_reps = unit_traces.shape[0]
            n_iters = unit_traces.shape[1]
        else:
            n_reps = 1
            n_iters = 1

        n_units = unit_traces.shape[2] if unit_traces is not None else 1

        shared_params = None
        if shared_traces is not None and n_shared > 0:
            shared_params = jnp.array(shared_traces[:, :, 1:])
        else:
            shared_params = jnp.zeros((n_reps, n_iters, n_shared))

        unit_params = None
        if unit_traces is not None and n_spec > 0:
            unit_params_only = unit_traces[:, :, :, 1:]
            unit_params = jnp.array(unit_params_only)
        else:
            unit_params = jnp.zeros((n_reps, n_iters, n_units, n_spec))

        shared_transformed, unit_transformed = self._transform_panel_array_jax(
            shared_params,
            unit_params,
            shared_param_names,
            unit_param_names,
            direction,
        )

        if shared_traces is not None:
            shared_out = shared_traces.copy()
            if n_shared > 0:
                shared_out[:, :, 1:] = np.array(shared_transformed)

        if unit_traces is not None:
            unit_out = unit_traces.copy()
            if n_spec > 0:
                unit_out[:, :, :, 1:] = np.array(unit_transformed)

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
