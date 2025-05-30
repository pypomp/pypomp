import numpy as np
import pandas as pd
import pypomp.measles.model_001b as m001b
from pypomp.pomp_class import Pomp
from pypomp.data.uk_measles.uk_measles import UKMeasles
from pypomp.model_struct import RInit
from pypomp.model_struct import RProc
from pypomp.model_struct import DMeas
from pypomp.model_struct import RMeas


def measles_Pomp(
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
    dat_filtered = dat[(dat["year"] >= first_year) & (dat["year"] <= last_year)]
    dat_filtered["time"] = (
        (dat_filtered["date"] - pd.Timestamp(f"{first_year}-01-01")).dt.days / 365.25
    ) + first_year
    dat_filtered = dat_filtered[
        (dat_filtered["time"] > first_year) & (dat_filtered["time"] < last_year + 1)
    ][["time", "cases"]]

    # ----prep-covariates-------------------------------------------------
    demog = demog.drop(columns=["unit"])
    times = np.arange(demog["year"].min(), demog["year"].max() + 1 / 12, 1 / 12)
    if interp_method == "shifted_splines":
        # TODO apply splines properly
        pop_interp = np.interp(times, demog["year"], demog["pop"])
        births_interp = np.interp(times - 4, demog["year"] + 0.5, demog["births"])
    elif interp_method == "linear":
        pop_interp = np.interp(times, demog["year"], demog["pop"])
        births_interp = np.interp(times - 4, demog["year"], demog["births"])

    covar_df = pd.DataFrame(
        {"time": times, "pop": pop_interp, "birthrate": births_interp}
    )

    # ----pomp-construction-----------------------------------------------
    # time = covar_df["time"]
    return Pomp(
        ys=dat_filtered["cases"],
        theta=theta,
        covars=covar_df,
        rinit=RInit(m001b.rinit),
        rproc=RProc(m001b.rproc, time_helper="euler", dt=dt),
        dmeas=DMeas(m001b.dmeas),
        rmeas=RMeas(m001b.rmeas),
        # t0=2 * time.iloc[0] - time.iloc[1],
        # times="time",
    )
