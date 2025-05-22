import pandas as pd
import numpy as np
import os

raw_location = "pypomp/data/uk_measles/raw"
urr = {}
for file in os.listdir(raw_location):
    if file.endswith(".csv"):
        urr[file.split(".")[0]] = pd.read_csv(
            os.path.join(raw_location, file), index_col=False
        )

# The raw data stores the observation dates as years with a decimal indicating
# how far into the year the observation was recorded. Rounding issues cause the
# conversion from numeric to date to produce dates which are not all spaced 7
# days apart, so we instead assign dates by adding multiples of 7 days to the
# first date.
dates = pd.to_datetime("1944-01-07") + 7 * pd.to_timedelta(
    np.arange(0, len(urr["measles_urban"]), 1), unit="D"
)

measles = pd.concat([urr["measles_rural"], urr["measles_urban"]], axis=1)
measles = pd.concat([measles, pd.DataFrame({"date": dates})], axis=1)
measles = pd.melt(measles, id_vars=["date"], var_name="unit", value_name="cases")
measles = measles.sort_values(["unit", "date"])

demog_pop = pd.concat([urr["pop_rural"], urr["pop_urban"]], axis=1)
demog_pop["year"] = demog_pop.index + 1944
demog_pop = pd.melt(demog_pop, id_vars=["year"], var_name="unit", value_name="pop")

# TODO: fix code below this line

demog_births = pd.concat([urr["births_rural"], urr["births_urban"]], axis=1)
demog_births["year"] = demog_births.index + 1944
demog_births = pd.melt(
    demog_births, id_vars=["year"], var_name="unit", value_name="births"
)

demog = pd.merge(demog_pop, demog_births, on=["unit", "year"], how="outer")
demog = demog.sort_values(["unit", "year"])

coord = pd.concat([urr["coord_rural"], urr["coord_urban"]], axis=0)
coord = pd.DataFrame({"unit": coord["X"], "long": coord["Long"], "lat": coord["Lat"]})
coord = coord.sort_values(["unit"])

# TODO: IIRC the he10 data has coordinates that the UK data doesn't have. These could be
# added in later.
# preexisting_units = uk_measles["coord"]["unit"][
#     uk_measles["coord"]["unit"].isin(twentycities["coord"]["unit"])
# ]
# uk_measles["coord"] = pd.concat(
#     [
#         uk_measles["coord"],
#         twentycities["coord"][~twentycities["coord"]["unit"].isin(preexisting_units)],
#     ]
# )
# ur_measles = {k: v.to_dict() for k, v in uk_measles.items()}


# Not sure if this is the best way to implement this.
class UKMeasles:
    data = {
        "measles": measles.reset_index(drop=True),
        "demog": demog.reset_index(drop=True),
        "coord": coord.reset_index(drop=True),
    }

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


# with open("pypomp/data/uk_measles/uk_measles.pkl", "wb") as f:
#     pickle.dump(uk_measles, f)
