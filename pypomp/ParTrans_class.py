from typing import Callable, Literal
import importlib
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


class ParTrans:
    """
    Class that handles the parameter transformation to and from the natural parameter space.

    Attributes:
        to_est: Function that transforms the parameters to the estimation parameter space.
        from_est: Function that transforms the parameters from the estimation parameter space to the natural parameter space.
    """

    def __init__(
        self,
        to_est: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None,
        from_est: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None,
    ):
        self.to_est = to_est or _to_est_default
        self.from_est = from_est or _from_est_default

    def panel_transform(
        self,
        theta: dict[str, pd.DataFrame | None],
        direction: Literal["to_est", "from_est"],
    ) -> dict[str, pd.DataFrame | None]:
        """
        Transform shared and unit-specific parameters for a single replicate.
        Input theta contains 'shared' and/or 'unit_specific' DataFrames.
        """
        func = self.to_est if direction == "to_est" else self.from_est

        # Normalize to empty DFs if None or missing for cleaner logic
        s_df = theta.get("shared")
        if s_df is None:
            s_df = None

        u_df = theta.get("unit_specific")
        if u_df is None:
            u_df = None

        res: dict[str, pd.DataFrame | None] = {"shared": None, "unit_specific": None}

        # Pre-calculate shared dictionary (param -> value)
        s_vals = s_df.iloc[:, 0].to_dict() if s_df is not None else {}

        # 1. Transform Shared Parameters
        if s_df is not None:
            # Context: Shared values + First unit's specific values (if any)
            ctx = s_vals.copy()
            if u_df is not None:
                first_unit = u_df.columns[0]
                ctx.update(u_df[first_unit].to_dict())

            trans = func(ctx)

            # Filter output back to just shared keys
            new_s_vals = {k: trans[k] for k in s_vals}
            res["shared"] = pd.DataFrame(
                new_s_vals.values(),
                index=pd.Index(new_s_vals.keys()),
                columns=pd.Index(["shared"]),
            )

        # 2. Transform Unit-Specific Parameters
        if u_df is not None:
            new_u_data = {}
            for unit in u_df.columns:
                # Context: Shared values + This unit's specific values
                ctx = s_vals.copy()
                ctx.update(u_df[unit].to_dict())

                trans = func(ctx)

                # Filter output back to specific keys (maintaining order)
                new_u_data[unit] = [trans[k] for k in u_df.index]

            res["unit_specific"] = pd.DataFrame(new_u_data, index=u_df.index)

        return res

    def panel_transform_list(
        self,
        theta_list: list[dict[str, pd.DataFrame | None]],
        direction: Literal["to_est", "from_est"],
    ) -> list[dict[str, pd.DataFrame | None]]:
        """
        Apply transform to a list of parameter sets.
        """
        return [self.panel_transform(t, direction) for t in theta_list]

    def to_floats(
        self, theta: dict[str, jax.Array], direction: Literal["to_est", "from_est"]
    ) -> dict[str, float]:
        """
        Convert the theta dictionary values from jax.Array to float.
        """
        if direction == "to_est":
            theta = self.to_est(theta)
            return {k: float(v) for k, v in theta.items()}
        elif direction == "from_est":
            theta = self.from_est(theta)
            return {k: float(v) for k, v in theta.items()}
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def transform_array(
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
            param_dict = {name: row[i] for i, name in enumerate(param_names)}
            transformed_dict = transform_fn(param_dict)
            return jnp.array([transformed_dict[name] for name in param_names])

        transform_vectorized = jax.vmap(transform_single_row)

        param_jax = jnp.array(param_array_2d)
        transformed_jax = transform_vectorized(param_jax)
        transformed_array = np.array(transformed_jax)

        if len(original_shape) == 1:
            return transformed_array.reshape(original_shape)
        else:
            return transformed_array.reshape(original_shape)

    def transform_panel_traces(
        self,
        shared_traces: np.ndarray | None,
        unit_traces: np.ndarray | None,
        shared_param_names: list[str],
        unit_param_names: list[str],
        unit_names: list[str],
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
            unit_names: List of unit names
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
                param_dict = {
                    name: shared_vals[i] for i, name in enumerate(shared_param_names)
                }
                if n_spec > 0:
                    param_dict.update(
                        {
                            name: unit_vals_for_context[i]
                            for i, name in enumerate(unit_param_names)
                        }
                    )
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
                param_dict = {}
                if n_shared > 0:
                    param_dict.update(
                        {
                            name: shared_vals_for_context[i]
                            for i, name in enumerate(shared_param_names)
                        }
                    )
                param_dict.update(
                    {name: unit_vals[i] for i, name in enumerate(unit_param_names)}
                )
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
        state = {}

        # Store function information for reconstruction
        # Check if to_est is a module-level function
        if (
            hasattr(self.to_est, "__module__")
            and hasattr(self.to_est, "__name__")
            and self.to_est.__module__ is not None
        ):
            state["_to_est_module"] = self.to_est.__module__
            state["_to_est_name"] = self.to_est.__name__
        else:
            # Lambda or closure - can't reliably pickle, will use default
            state["_to_est_is_lambda"] = True

        # Check if from_est is a module-level function
        if (
            hasattr(self.from_est, "__module__")
            and hasattr(self.from_est, "__name__")
            and self.from_est.__module__ is not None
        ):
            state["_from_est_module"] = self.from_est.__module__
            state["_from_est_name"] = self.from_est.__name__
        else:
            # Lambda or closure - can't reliably pickle, will use default
            state["_from_est_is_lambda"] = True

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct functions.

        Reconstructs module-level functions by importing them.
        Falls back to defaults for lambdas/closures.
        """
        # Reconstruct to_est
        if "_to_est_is_lambda" in state:
            # Can't reconstruct lambdas - use default
            self.to_est = _to_est_default
        else:
            module = importlib.import_module(state["_to_est_module"])
            self.to_est = getattr(module, state["_to_est_name"])

        # Reconstruct from_est
        if "_from_est_is_lambda" in state:
            # Can't reconstruct lambdas - use default
            self.from_est = _from_est_default
        else:
            module = importlib.import_module(state["_from_est_module"])
            self.from_est = getattr(module, state["_from_est_name"])


def _to_est_default(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return theta


def _from_est_default(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return theta
