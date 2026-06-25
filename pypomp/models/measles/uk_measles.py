from typing import Literal
import numpy as np
import pandas as pd
import os
import pickle
import pypomp.models.measles.model_001 as m001
import pypomp.models.measles.model_001b as m001b
import pypomp.models.measles.model_001c as m001c
import pypomp.models.measles.model_001d as m001d
import pypomp.models.measles.model_002 as m002
import pypomp.models.measles.model_002d as m002d
import pypomp.models.measles.model_003 as m003
from scipy.interpolate import make_smoothing_spline
from pypomp.core.pomp import Pomp
from pypomp.panel.panel import PanelPomp
from pypomp.core.par_trans import ParTrans
from pypomp.core.parameters import PompParameters, PanelParameters


def evaluate_spline_with_linear_extrapolation(spline, x, x_min, x_max):
    """
    Evaluates a SciPy BSpline with linear extrapolation outside [x_min, x_max],
    matching R's predict.smooth.spline behavior.
    """
    x = np.atleast_1d(x)
    y = spline(x)

    deriv = spline.derivative()

    left_mask = x < x_min
    if np.any(left_mask):
        y_min = float(spline(x_min))
        dy_min = float(deriv(x_min))
        y[left_mask] = y_min + dy_min * (x[left_mask] - x_min)

    right_mask = x > x_max
    if np.any(right_mask):
        y_max = float(spline(x_max))
        dy_max = float(deriv(x_max))
        y[right_mask] = y_max + dy_max * (x[right_mask] - x_max)

    return y


# Not sure if this is the best way to implement this.
class UKMeasles:
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    _data_dir = os.path.join(_module_dir, os.pardir, os.pardir, "data/uk_measles")
    _data_file = os.path.join(_data_dir, "uk_measles.pkl")
    _data = None

    MODELS = {
        "001": m001,
        "001b": m001b,
        "001c": m001c,
        "001d": m001d,
        "002": m002,
        "002d": m002d,
        "003": m003,
    }

    @classmethod
    def _get_data(cls):
        if cls._data is None:
            with open(cls._data_file, "rb") as f:
                cls._data = pickle.load(f)
        return cls._data

    @classmethod
    def subset(cls, units=None, clean=False):
        """
        Return a subset of the UKMeasles data, filtered by the given units.

        Parameters
        ----------
        units : list of str, optional
            A list of unit names to subset the data by. If None, the entire
            dataset is returned.

        clean : bool, optional
            If True, returns a copy of the data with suspicious values set to np.nan.

        Returns
        -------
        A dictionary with the same structure as UKMeasles.data, but with the data subsetted to only include the given units.
        """
        raw_data = cls._get_data()
        data = {k: v.copy() for k, v in raw_data.items()}

        if clean:
            # London 1955-08-12   124
            # London 1955-08-19    82
            # London 1955-08-26     0
            # London 1955-09-02    58
            # London 1955-09-09    38
            data["measles"].loc[
                (data["measles"]["unit"] == "London")
                & (data["measles"]["date"] == "1955-08-26"),
                "cases",
            ] = np.nan
            # The value 76 was used in He10.

            # 13770 Liverpool 1955-11-04    10
            # 13771 Liverpool 1955-11-11    25
            # 13772 Liverpool 1955-11-18   116
            # 13773 Liverpool 1955-11-25    17
            # 13774 Liverpool 1955-12-02    18
            data["measles"].loc[
                (data["measles"]["unit"] == "Liverpool")
                & (data["measles"]["date"] == "1955-11-18"),
                "cases",
            ] = np.nan

            # 13950 Liverpool 1959-04-17   143
            # 13951 Liverpool 1959-04-24   115
            # 13952 Liverpool 1959-05-01   450
            # 13953 Liverpool 1959-05-08    96
            # 13954 Liverpool 1959-05-15   157
            data["measles"].loc[
                (data["measles"]["unit"] == "Liverpool")
                & (data["measles"]["date"] == "1959-05-01"),
                "cases",
            ] = np.nan

            # 19552 Nottingham 1961-08-18     6
            # 19553 Nottingham 1961-08-25     7
            # 19554 Nottingham 1961-09-01    66
            # 19555 Nottingham 1961-09-08     8
            # 19556 Nottingham 1961-09-15     7
            data["measles"].loc[
                (data["measles"]["unit"] == "Nottingham")
                & (data["measles"]["date"] == "1961-09-01"),
                "cases",
            ] = np.nan

            # Sheffield 1961-05-05   266
            # Sheffield 1961-05-12   346
            # Sheffield 1961-05-19     0
            # Sheffield 1961-05-26   314
            # Sheffield 1961-06-02   297
            data["measles"].loc[
                (data["measles"]["unit"] == "Sheffield")
                & (data["measles"]["date"] == "1961-05-19"),
                "cases",
            ] = np.nan

            # Hull 1956-06-22    72
            # Hull 1956-06-29    94
            # Hull 1956-07-06     0
            # Hull 1956-07-13    91
            # Hull 1956-07-20    87

            data["measles"].loc[
                (data["measles"]["unit"] == "Hull")
                & (data["measles"]["date"] == "1956-07-06"),
                "cases",
            ] = np.nan

        if units is None:
            return data
        else:
            return {
                k: v[v["unit"].isin(units)].reset_index(drop=True)
                for k, v in data.items()
            }

    @classmethod
    def AK_mles(cls) -> pd.DataFrame:
        """
        Returns a data frame of Aaron King's MLEs from https://kingaa.github.io/sbied/measles/index.html
        """
        data_file = os.path.join(cls._data_dir, "AK_mles.csv")
        df = pd.read_csv(data_file, index_col="town")
        df.drop(columns=["loglik", "loglik.sd", "mu", "delay"], inplace=True)
        return df[
            [
                "R0",
                "sigma",
                "gamma",
                "iota",
                "rho",
                "sigmaSE",
                "psi",
                "cohort",
                "amplitude",
                "S_0",
                "E_0",
                "I_0",
                "R_0",
            ]
        ].T

    @classmethod
    def Pomp(
        cls,
        unit: str,
        theta: PompParameters,
        model: Literal["001", "001b", "001c", "001d", "002", "002d", "003"] = "001b",
        interp_method: Literal["shifted_splines", "linear"] = "shifted_splines",
        first_year: int = 1950,
        last_year: int = 1963,
        dt: float = 1 / 365.25,
        clean=False,
    ):
        """
        Returns a Pomp object for the UK Measles data.

        Parameters
        ----------
        unit : str
            The name of the unit to use.
        theta : PompParameters
            Parameters for the model.
        model : Literal["001", "001b", "001c", "001d", "002", "002d", "003"]
            The model to use.
        interp_method : Literal["shifted_splines", "linear"]
            The method to use to interpolate the covariates.
        first_year : int
            The first year of the data to use.
        last_year : int
            The last year of the data to use.
        dt : float
            The time step size to use for the model.
        clean : bool
            If True, uses a copy of the data with suspicious values set to np.nan.

        Returns
        -------
        Pomp:
            A Pomp object for the UK Measles data.
        """

        data = cls.subset([unit], clean)
        measles = data["measles"]
        demog = data["demog"]

        # ----prep-data-------------------------------------------------
        dat = measles.copy()
        dat["year"] = dat["date"].dt.year
        dat_filtered = dat[
            (dat["year"] >= first_year) & (dat["year"] <= last_year)
        ].copy()
        dat_filtered["time"] = (
            (dat_filtered["date"] - pd.Timestamp(f"{first_year}-01-01")).dt.days
            / 365.25
        ) + first_year
        dat_filtered = dat_filtered[
            (dat_filtered["time"] > first_year) & (dat_filtered["time"] < last_year + 1)
        ][["time", "cases"]]
        dat_filtered.set_index("time", inplace=True)

        # ----prep-covariates-------------------------------------------------
        demog = demog.drop(columns=["unit"])
        times = np.arange(demog["year"].min(), demog["year"].max() + 1 / 12, 1 / 12)
        if interp_method == "shifted_splines":
            pop_bspl = make_smoothing_spline(demog["year"], demog["pop"])
            births_bspl = make_smoothing_spline(demog["year"] + 0.5, demog["births"])
            pop_interp = evaluate_spline_with_linear_extrapolation(
                pop_bspl, times, x_min=demog["year"].min(), x_max=demog["year"].max()
            )
            births_interp = evaluate_spline_with_linear_extrapolation(
                births_bspl,
                times - 4,
                x_min=demog["year"].min() + 0.5,
                x_max=demog["year"].max() + 0.5,
            )
        elif interp_method == "linear":
            pop_interp = np.interp(times, demog["year"], demog["pop"])
            births_interp = np.interp(times - 4, demog["year"], demog["births"])
        else:
            raise ValueError(f"interp_method {interp_method} not recognized")

        covar_df = pd.DataFrame(
            {"time": times, "pop": pop_interp, "birthrate": births_interp}
        )
        covar_df.set_index("time", inplace=True)

        # Add log(pop_1950) as constant covariate for iota log-log linear models
        pop_1950_row = demog.loc[demog["year"] == 1950, "pop"]
        if len(pop_1950_row) > 0:
            covar_df["log_pop_1950"] = np.log(float(pop_1950_row.values[0]))
        else:
            # Fallback: use earliest available year
            covar_df["log_pop_1950"] = np.log(float(demog["pop"].iloc[0]))

        # Placeholder for standardized log(pop_1950); must be overwritten
        # at the panel level with correct z-score across all units.
        covar_df["std_log_pop_1950"] = covar_df["log_pop_1950"]

        # ----pomp-construction-----------------------------------------------

        mod = cls.MODELS[model]

        missing_params = [
            p for p in mod.param_names if p not in theta.get_param_names()
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for model '{model}': {missing_params}"
            )

        t0 = float(2 * dat_filtered.index[0] - dat_filtered.index[1])
        return Pomp(
            ys=dat_filtered,
            theta=theta,
            covars=covar_df,
            t0=t0,
            nstep=None,
            dt=dt,
            accumvars=mod.accumvars,
            statenames=mod.statenames,
            rinit=mod.rinit,
            rproc=mod.rproc,
            dmeas=mod.dmeas,
            rmeas=mod.rmeas,
            par_trans=ParTrans(to_est=mod.to_est, from_est=mod.from_est),
        )

    @classmethod
    def PanelPomp(
        cls,
        units: list[str],
        theta: PanelParameters,
        model: Literal["001", "001b", "001c", "001d", "002", "002d", "003"] = "001b",
        interp_method: Literal["shifted_splines", "linear"] = "shifted_splines",
        first_year: int = 1950,
        last_year: int = 1963,
        dt: float = 1 / 365.25,
        clean: bool = False,
    ):
        """
        Returns a PanelPomp object for the UK Measles data.

        Parameters
        ----------
        units : list of str
            List of units to include in the panel.
        theta : PanelParameters
            Parameters for the panel model.
        model : Literal["001", "001b", "001c", "001d", "002", "002d", "003"]
            The model to use.
        interp_method : Literal["shifted_splines", "linear"]
            The method to use to interpolate the covariates.
        first_year : int
            The first year of the data to use.
        last_year : int
            The last year of the data to use.
        dt : float
            The time step size to use for the model.
        clean : bool
            If True, uses a copy of the data with suspicious values set to np.nan.

        Returns
        -------
        PanelPomp:
            A PanelPomp object for the UK Measles data.
        """
        if not isinstance(theta, PanelParameters):
            raise TypeError("theta must be a PanelParameters instance")

        mod = cls.MODELS[model]
        param_names = mod.param_names

        pomp_dict = {}
        theta_list = theta.params()

        for unit in units:
            unit_theta_dict = {}
            if len(theta_list) > 0:
                theta_dict = theta_list[0]
                if theta_dict["shared"] is not None:
                    unit_theta_dict.update(theta_dict["shared"].iloc[:, 0].to_dict())
                if theta_dict["unit_specific"] is not None:
                    unit_theta_dict.update(theta_dict["unit_specific"][unit].to_dict())

            missing_params = [p for p in param_names if p not in unit_theta_dict]
            if missing_params:
                raise ValueError(
                    f"Missing required parameters for unit '{unit}': {missing_params}"
                )

            unit_theta = PompParameters(unit_theta_dict)

            pomp_dict[unit] = cls.Pomp(
                unit=unit,
                theta=unit_theta,
                model=model,
                interp_method=interp_method,
                first_year=first_year,
                last_year=last_year,
                dt=dt,
                clean=clean,
            )

        log_pops = {
            unit: float(pomp_obj.covars["log_pop_1950"].iloc[0])
            for unit, pomp_obj in pomp_dict.items()
        }
        log_pop_values = list(log_pops.values())
        mean_log_pop = np.mean(log_pop_values)
        if len(log_pop_values) > 1:
            sd_log_pop = np.std(log_pop_values, ddof=1)
        else:
            sd_log_pop = 1.0

        if sd_log_pop == 0.0:
            sd_log_pop = 1.0

        for unit, pomp_obj in pomp_dict.items():
            std_val = (log_pops[unit] - mean_log_pop) / sd_log_pop
            pomp_obj.covars["std_log_pop_1950"] = std_val

        return PanelPomp(Pomp_dict=pomp_dict, theta=theta)
