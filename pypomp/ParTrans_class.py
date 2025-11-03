from typing import Callable, Literal
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


class ParTrans:
    """
    Class that handles the parameter transformation to and from the natural parameter space.
    """

    def __init__(
        self,
        to_est: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None,
        from_est: Callable[[dict[str, jax.Array]], dict[str, jax.Array]] | None = None,
    ):
        self.to_est = to_est or to_est_default
        self.from_est = from_est or from_est_default

    def panel_transform(
        self,
        shared: pd.DataFrame | None,
        unit_specific: pd.DataFrame | None,
        direction: Literal["to_est", "from_est"],
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Transform shared and unit-specific parameters to or from the estimation parameter space.
        """
        if shared is None and unit_specific is None:
            return None, None

        shared_out = None
        unit_specific_out = None

        # Process shared parameters
        if shared is not None:
            shared_names = shared.index.tolist()
            shared_dict = shared.to_dict("index")
            # Extract the single column value for each parameter
            shared_values = {
                name: list(data.values())[0] for name, data in shared_dict.items()
            }

            complete_input = shared_values.copy()
            if unit_specific is not None:
                unit_specific_names = unit_specific.index.tolist()
                # Fill in actual values from the first unit for unit-specific parameters
                first_unit = unit_specific.columns[0]
                unit_values = unit_specific[first_unit].to_dict()
                complete_input.update(unit_values)

            shared_transformed = (
                self.to_est(complete_input)
                if direction == "to_est"
                else self.from_est(complete_input)
            )
            shared_out = pd.DataFrame(
                index=pd.Index(shared_names),
                data={"shared": [shared_transformed[name] for name in shared_names]},
            )

        # Process unit-specific parameters
        if unit_specific is not None:
            unit_specific_names = unit_specific.index.tolist()
            unit_names = unit_specific.columns.tolist()
            unit_specific_out = pd.DataFrame(index=pd.Index(unit_specific_names))

            for unit in unit_names:
                input_dict = {}
                if shared is not None:
                    shared_dict = shared.to_dict("index")
                    shared_values = {
                        name: list(data.values())[0]
                        for name, data in shared_dict.items()
                    }
                    input_dict.update(shared_values)

                unit_values = unit_specific[unit].to_dict()
                input_dict.update(unit_values)

                output_dict = (
                    self.to_est(input_dict)
                    if direction == "to_est"
                    else self.from_est(input_dict)
                )

                unit_specific_transformed = {
                    name: output_dict[name] for name in unit_specific_names
                }
                unit_specific_out[unit] = [
                    unit_specific_transformed[name] for name in unit_specific_names
                ]

        return shared_out, unit_specific_out

    def panel_transform_list(
        self,
        shared_list: list[pd.DataFrame] | None,
        unit_specific_list: list[pd.DataFrame] | None,
        direction: Literal["to_est", "from_est"],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        if shared_list is None and unit_specific_list is None:
            return [], []

        # Convert None inputs to list of Nones with appropriate length
        if shared_list is None and unit_specific_list is not None:
            length = len(unit_specific_list)
            shared_list = [None] * length  # type: ignore
        elif shared_list is not None and unit_specific_list is None:
            length = len(shared_list)
            unit_specific_list = [None] * length  # type: ignore

        # Both lists should now be non-None for easy iteration
        assert shared_list is not None
        assert unit_specific_list is not None

        if len(shared_list) != len(unit_specific_list):
            raise ValueError(
                "shared_list and unit_specific_list must have the same length"
            )

        param_trans_list: list[tuple[pd.DataFrame | None, pd.DataFrame | None]] = [
            self.panel_transform(shared, spec, direction=direction)
            for shared, spec in zip(shared_list, unit_specific_list)
        ]
        shared_trans_list = [
            shared_trans.apply(pd.to_numeric, errors="coerce").astype(float)
            for shared_trans, _ in param_trans_list
            if shared_trans is not None
        ]
        spec_trans_list = [
            spec_trans.apply(pd.to_numeric, errors="coerce").astype(float)
            for _, spec_trans in param_trans_list
            if spec_trans is not None
        ]
        return shared_trans_list, spec_trans_list  # type: ignore

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
        Transform a parameter array to or from the estimation parameter space.
        
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
                param_dict = {name: shared_vals[i] for i, name in enumerate(shared_param_names)}
                if n_spec > 0:
                    param_dict.update({name: unit_vals_for_context[i] for i, name in enumerate(unit_param_names)})
                transformed = transform_fn(param_dict)
                return jnp.array([transformed[name] for name in shared_param_names])
            
            transform_shared_vectorized = jax.vmap(jax.vmap(transform_shared_single))
            
            shared_params_only = jnp.array(shared_traces[:, :, 1:])
            
            if unit_traces is not None and n_spec > 0:
                unit_context = jnp.array(unit_traces[:, :, 1:, 0])
            else:
                unit_context = jnp.zeros((n_reps, n_iters, max(1, n_spec)))
            
            transformed_shared = transform_shared_vectorized(shared_params_only, unit_context)
            shared_out[:, :, 1:] = np.array(transformed_shared)
        
        if unit_traces is not None and n_spec > 0:
            n_reps, n_iters, _, n_units = unit_traces.shape
            unit_out = unit_traces.copy()
            
            def transform_unit_single(shared_vals_for_context, unit_vals):
                param_dict = {}
                if n_shared > 0:
                    param_dict.update({name: shared_vals_for_context[i] for i, name in enumerate(shared_param_names)})
                param_dict.update({name: unit_vals[i] for i, name in enumerate(unit_param_names)})
                transformed = transform_fn(param_dict)
                return jnp.array([transformed[name] for name in unit_param_names])
            
            vmap_over_units = jax.vmap(transform_unit_single, in_axes=(None, 2))
            vmap_over_iters = jax.vmap(vmap_over_units, in_axes=(0, 0))
            transform_unit_vectorized = jax.vmap(vmap_over_iters, in_axes=(0, 0))
            
            if shared_traces is not None and n_shared > 0:
                shared_context = jnp.array(shared_traces[:, :, 1:])
            else:
                shared_context = jnp.zeros((n_reps, n_iters, max(1, n_shared)))
            
            unit_params_only = jnp.array(unit_traces[:, :, 1:, :])
            
            transformed_unit = transform_unit_vectorized(shared_context, unit_params_only)
            unit_out[:, :, 1:, :] = np.array(transformed_unit)
        elif unit_traces is not None:
            unit_out = unit_traces.copy()
        
        return shared_out, unit_out



def to_est_default(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return theta


def from_est_default(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return theta
