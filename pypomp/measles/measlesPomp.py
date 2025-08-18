import numpy as np
import pandas as pd
import os
import pickle
import pypomp.measles.model_001b as m001b
from scipy.interpolate import make_splrep
from scipy.interpolate import splev
from pypomp.pomp_class import Pomp
from pypomp.model_struct import RInit
from pypomp.model_struct import RProc
from pypomp.model_struct import DMeas
from pypomp.model_struct import RMeas


# Not sure if this is the best way to implement this.
class UKMeasles:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(module_dir, os.pardir, "data/uk_measles")
    data_file = os.path.join(data_dir, "uk_measles.pkl")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    @staticmethod
    def subset(units=None):
        """
        Return a subset of the UKMeasles data, filtered by the given units.

        Parameters
        ----------
        units : list of str, optional
            A list of unit names to subset the data by. If None, the entire
            dataset is returned.

        Returns
        -------
        A dictionary with the same structure as UKMeasles.data, but with the
        data subsetted to only include the given units.
        """
        if units is None:
            return UKMeasles.data
        else:
            return {
                k: v[v["unit"].isin(units)].reset_index(drop=True)
                for k, v in UKMeasles.data.items()
            }

    # TODO: add method or argument to return the cleaned copy of the data

    @staticmethod
    def Pomp(
        unit,
        theta,
        interp_method="shifted_splines",
        first_year=1950,
        last_year=1963,
        dt=1 / 365.25,
    ):
        data = UKMeasles.subset(unit)
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
            pop_bspl = make_splrep(demog["year"], demog["pop"])
            births_bspl = make_splrep(demog["year"] + 0.5, demog["births"])
            pop_interp = splev(times, pop_bspl)
            births_interp = splev(times - 4, births_bspl)
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
        t0 = float(2 * dat_filtered.index[0] - dat_filtered.index[1])
        return Pomp(
            ys=dat_filtered,
            theta=theta,
            covars=covar_df,
            rinit=RInit(m001b.rinit, t0=t0),
            rproc=RProc(m001b.rproc, step_type="euler", dt=dt, accumvars=(4, 5)),
            dmeas=DMeas(m001b.dmeas),
            rmeas=RMeas(m001b.rmeas, ydim=1),
        )
