"""
UK Measles Data Preparation Script (data-raw).

This script processes the raw epidemiological and demographic CSV files from
`data-raw/uk_measles/raw/` and exports clean, pre-processed datasets into
`pypomp/data/uk_measles/`.

Data Provenance
---------------
The raw data files in the `raw/` subdirectory originate from:
    Korevaar, Hannah, C. Jessica Metcalf, and Bryan T. Grenfell.
    "Structure, space and size: competing drivers of variation in urban and rural measles transmission."
    Journal of The Royal Society Interface 17, no. 168 (2020): 20200010.
    https://doi.org/10.1098/rsif.2020.0010

except `raw/coord_lau2020.csv`, which is `data/formatted/prevac/coordinates_urban.csv`
from https://github.com/msylau/measles_competing_risks (commit b7a6758). It derives
from the Grenfell/Bjørnstad England and Wales dataset and accompanies:
    Lau, Max S. Y., et al. "A competing-risks model explains hierarchical spatial
    coupling of measles epidemics en route to national elimination."
    Nature Ecology & Evolution 4 (2020): 934–939.
    https://doi.org/10.1038/s41559-020-1186-6
Despite its name, it covers every urban and rural unit, plus the 29 London
metropolitan boroughs that the case data merge into a single London unit.

Transformations
---------------
1. Case Reports:
   - Raw observation dates are recorded as decimal fractions of years. To avoid floating-point
     rounding discrepancies and ensure exact weekly spacing (7 days apart), dates are synthesized
     by adding multiples of 7 days starting from 1944-01-07.
   - Urban and rural case reports are combined and melted from wide format into long format
     with columns: `['date', 'unit', 'cases']`.
   - Exported as `measles.csv.gz` (gzipped CSV for space efficiency).

2. Demographics:
   - Annual population and birth counts for urban and rural units (1944 onwards) are merged
     into a single table with columns: `['unit', 'year', 'pop', 'births']`.
   - Exported as `demog.csv`.

3. Spatial Coordinates:
   - The raw file stores one unit per column; it is transposed into a table with columns
     `['unit', 'long', 'lat']`, restricted to units in the case data (dropping the London
     boroughs).
   - Exported as `coord.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def process_raw_measles_data(
    raw_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Read and transform raw UK measles CSV files into structured DataFrames.

    Parameters
    ----------
    raw_dir : Path or str, optional
        Path to the directory containing raw CSV files. Defaults to the
        sibling `raw/` directory.

    Returns
    -------
    dict of str to pd.DataFrame
        Dictionary containing:
        - `'measles'`: Case reports with columns `['date', 'unit', 'cases']`.
        - `'demog'`: Demographic data with columns `['unit', 'year', 'pop', 'births']`.
        - `'coord'`: Spatial coordinates with columns `['unit', 'long', 'lat']`.
    """
    if raw_dir is None:
        raw_path = Path(__file__).resolve().parent / "raw"
    else:
        raw_path = Path(raw_dir).resolve()

    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    urr: dict[str, pd.DataFrame] = {}
    for csv_file in raw_path.glob("*.csv"):
        urr[csv_file.stem] = pd.read_csv(csv_file, index_col=False)

    if "measles_urban" not in urr or "measles_rural" not in urr:
        raise ValueError(
            f"Missing required measles CSV files in {raw_path}. Found: {list(urr.keys())}"
        )

    # 1. Weekly case reports
    num_weeks = len(urr["measles_urban"])
    dates = pd.to_datetime("1944-01-07") + 7 * pd.to_timedelta(
        np.arange(0, num_weeks, 1), unit="D"
    )

    measles = pd.concat([urr["measles_rural"], urr["measles_urban"]], axis=1)
    measles = pd.concat([measles, pd.DataFrame({"date": dates})], axis=1)
    measles = pd.melt(measles, id_vars=["date"], var_name="unit", value_name="cases")
    measles = measles.sort_values(["unit", "date"]).reset_index(drop=True)

    # 2. Annual demographics (population and births)
    demog_pop = pd.concat([urr["pop_rural"], urr["pop_urban"]], axis=1).copy()
    demog_pop["year"] = demog_pop.index + 1944
    demog_pop = pd.melt(demog_pop, id_vars=["year"], var_name="unit", value_name="pop")

    demog_births = pd.concat([urr["births_rural"], urr["births_urban"]], axis=1).copy()
    demog_births["year"] = demog_births.index + 1944
    demog_births = pd.melt(
        demog_births, id_vars=["year"], var_name="unit", value_name="births"
    )

    demog = pd.merge(demog_pop, demog_births, on=["unit", "year"], how="outer")
    demog = demog.sort_values(["unit", "year"]).reset_index(drop=True)

    # 3. Spatial coordinates
    coord = urr["coord_lau2020"].set_index(urr["coord_lau2020"].columns[0]).T
    coord = pd.DataFrame(
        {"unit": coord.index, "long": coord["Long"], "lat": coord["Lat"]}
    )
    units = set(measles["unit"])
    missing = units - set(coord["unit"])
    if missing:
        raise ValueError(f"Units missing coordinates: {sorted(missing)}")
    coord = coord[coord["unit"].isin(units)]
    coord = coord.sort_values(["unit"]).reset_index(drop=True)

    return {
        "measles": measles,
        "demog": demog,
        "coord": coord,
    }


def main() -> None:
    """Run data processing and export pre-processed tables to pypomp/data/uk_measles/."""
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir.parent.parent / "pypomp" / "data" / "uk_measles"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing raw UK Measles data...")
    data = process_raw_measles_data()

    measles_path = output_dir / "measles.csv.gz"
    demog_path = output_dir / "demog.csv"
    coord_path = output_dir / "coord.csv"

    print(f"Writing {measles_path}...")
    data["measles"].to_csv(measles_path, index=False, compression="gzip")

    print(f"Writing {demog_path}...")
    data["demog"].to_csv(demog_path, index=False)

    print(f"Writing {coord_path}...")
    data["coord"].to_csv(coord_path, index=False)

    print("Export complete:")
    for name, path in [
        ("measles", measles_path),
        ("demog", demog_path),
        ("coord", coord_path),
    ]:
        size_kb = path.stat().st_size / 1024
        print(f"  - {name}: {path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
