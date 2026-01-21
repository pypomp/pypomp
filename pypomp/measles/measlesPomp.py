import numpy as np
import pandas as pd
import os
import pickle
import pypomp.measles.model_001b as m001b
import pypomp.measles.model_001c as m001c
import pypomp.measles.model_002 as m002
from scipy.interpolate import make_smoothing_spline
from pypomp.pomp_class import Pomp
import copy
from pypomp.ParTrans_class import ParTrans


# Not sure if this is the best way to implement this.
class UKMeasles:
    _module_dir = os.path.dirname(os.path.abspath(__file__))
    _data_dir = os.path.join(_module_dir, os.pardir, "data/uk_measles")
    _data_file = os.path.join(_data_dir, "uk_measles.pkl")
    with open(_data_file, "rb") as _f:
        _data = pickle.load(_f)

    @staticmethod
    def subset(units=None, clean=False):
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
        data = copy.deepcopy(UKMeasles._data)

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

    @staticmethod
    def AK_mles():
        """
        MLEs from https://kingaa.github.io/sbied/measles/index.html
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(module_dir, os.pardir, "data/uk_measles/AK_mles.csv")
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

    @staticmethod
    def Pomp(
        unit: list[str],
        theta: dict | list[dict],
        model: str = "001b",
        interp_method: str = "shifted_splines",
        first_year: int = 1950,
        last_year: int = 1963,
        dt: float = 1 / 365.25,
        clean=False,
    ):
        """
        Returns a Pomp object for the UK Measles data.

        Parameters
        ----------
        unit : list[str]
            Which unit to use. Currently only supports one unit.
        theta : dict | list[dict]
            Parameters for the model. Can be a single dict or a list of dicts.
        model : str
            The model to use. Can be "001b" or "001c", currently.
        interp_method : str
            The method to use to interpolate the covariates. Can be "shifted_splines" or "linear".
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

        data = UKMeasles.subset(unit, clean)
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
            # TODO fix exploding birthrate below year 1950
            pop_bspl = make_smoothing_spline(demog["year"], demog["pop"])
            births_bspl = make_smoothing_spline(demog["year"] + 0.5, demog["births"])
            pop_interp = pop_bspl(times)
            births_interp = births_bspl(times - 4)
        elif interp_method == "linear":
            pop_interp = np.interp(times, demog["year"], demog["pop"])
            births_interp = np.interp(times - 4, demog["year"], demog["births"])
        else:
            raise ValueError(f"interp_method {interp_method} not recognized")

        covar_df = pd.DataFrame(
            {"time": times, "pop": pop_interp, "birthrate": births_interp}
        )
        covar_df.set_index("time", inplace=True)

        # ----pomp-construction-----------------------------------------------

        mod = {
            "001b": m001b,
            "001c": m001c,
            "002": m002,
        }[model]
        t0 = float(2 * dat_filtered.index[0] - dat_filtered.index[1])
        return Pomp(
            ys=dat_filtered,
            theta=theta,
            covars=covar_df,
            t0=t0,
            nstep=None,
            dt=dt,
            ydim=1,
            accumvars=mod.accumvars,
            statenames=mod.statenames,
            rinit=mod.rinit,
            rproc=mod.rproc,
            dmeas=mod.dmeas,
            rmeas=mod.rmeas,
            par_trans=ParTrans(to_est=mod.to_est, from_est=mod.from_est),
        )
