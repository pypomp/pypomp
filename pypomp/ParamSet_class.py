import pandas as pd


class ParamSet:
    def __init__(
        self,
        global_params: dict | list[dict] | None = None,
        local_params: pd.DataFrame | list[pd.DataFrame] | None = None,
    ):
        """
        Initializes a ParamSet object.
        """
        self.global_params = global_params
        self.local_params = local_params

    @staticmethod
    def _enlist(x) -> list:
        if x is None:
            return []
        elif not isinstance(x, list):
            return [x]
        else:
            return x

    def _validate_params_and_units(
        self,
        global_params: dict | list[dict] | None = None,
        local_params: pd.DataFrame | list[pd.DataFrame] | None = None,
    ):
        """
        Validates the global and local parameters.
        """
        gp: list[dict] = (
            ParamSet._enlist(global_params)
            if global_params is not None
            else ParamSet._enlist(self.global_params)
        )
        lp: list[pd.DataFrame] = (
            ParamSet._enlist(local_params)
            if local_params is not None
            else ParamSet._enlist(self.local_params)
        )
        if len(gp) == 0 and len(lp) == 0:
            raise ValueError("Either global_params or local_params must be provided.")
        if len(gp) > 0:
            if not all(isinstance(param, dict) for param in gp):
                raise ValueError(
                    "global_params must be a dictionary or a list of dictionaries"
                )
            if not all(param.keys() == gp[0].keys() for param in gp):
                raise ValueError(
                    "All global_params dictionaries must have the same keys"
                )
            if not all(
                isinstance(val, float) for param in gp for val in param.values()
            ):
                raise ValueError(
                    "All values in global_params dictionaries must be floats"
                )
        if len(lp) > 0:
            if not all(isinstance(param, pd.DataFrame) for param in lp):
                raise ValueError(
                    "local_params must be a pandas DataFrame or a list of pandas DataFrames"
                )

        if len(gp) == 0 and len(lp) > 0:
            gp = [{} for _ in range(len(lp))]
        if len(lp) == 0 and len(gp) > 0:
            lp = [pd.DataFrame() for _ in range(len(gp))]

        if len(gp) != len(lp):
            raise ValueError("global_params and local_params must have the same length")
        return gp, lp
