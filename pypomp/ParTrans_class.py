from typing import Callable, Literal
import pandas as pd
import jax


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
        param_trans_list: list[tuple[pd.DataFrame | None, pd.DataFrame | None]] = [
            self.panel_transform(shared, spec, direction=direction)
            for shared, spec in zip(shared_list, spec_list)
        ]
        shared_trans_list: list[pd.DataFrame] = [
            shared_trans for shared_trans, _ in param_trans_list
        ]
        spec_trans_list: list[pd.DataFrame] = [
            spec_trans for _, spec_trans in param_trans_list
        ]
        return shared_trans_list, spec_trans_list

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


def to_est_default(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return theta


def from_est_default(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return theta
