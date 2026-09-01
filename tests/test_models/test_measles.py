import jax
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
import pypomp.models.measles.uk_measles as uk_measles_mod

# import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)


BASE_THETA = {
    "R0": 56.8,
    "sigma": 28.9,
    "gamma": 30.4,
    "iota": 2.9,
    "rho": 0.488,
    "sigmaSE": 0.0878,
    "psi": 0.116,
    "cohort": 0.557,
    "amplitude": 0.554,
    "S_0": 2.97e-02,
    "E_0": 5.17e-05,
    "I_0": 5.14e-05,
    "R_0": 9.70e-01,
    "mu": 0.02,
    "alpha": 1.0,
}

DEFAULT_J = 3
DEFAULT_KEY = jax.random.key(1)
DEFAULT_M = 2
DEFAULT_A = 0.5


@pytest.fixture(scope="function")
def london():
    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]
    measles = pp.models.UKMeasles.pomp(
        unit="London",
        theta=pp.PompParameters(theta),
        clean=True,
        model="001b",
        # dt=7 / 365.25,
    )

    return measles


@pytest.fixture(scope="function")
def default_rw_sd():
    rw_sd = pp.RWSigma(
        sigmas={
            "R0": 0.02,
            "sigma": 0.02,
            "gamma": 0.02,
            "iota": 0.02,
            "rho": 0.02,
            "sigmaSE": 0.02,
            "psi": 0.02,
            "cohort": 0.02,
            "amplitude": 0.02,
            "S_0": 0.01,
            "E_0": 0.01,
            "I_0": 0.01,
            "R_0": 0.01,
        },
        init_names=["S_0", "E_0", "I_0", "R_0"],
    )
    return rw_sd


@pytest.fixture(scope="function")
def london_003():
    theta = BASE_THETA.copy()
    measles = pp.models.UKMeasles.pomp(
        unit="London",
        theta=pp.PompParameters(theta),
        model="003",
    )
    return measles


@pytest.mark.parametrize(
    "model,theta",
    [
        ("001", BASE_THETA),
        ("003", BASE_THETA),
        (
            "002",
            {
                "R0": BASE_THETA["R0"],
                "sigma": BASE_THETA["sigma"],
                "gamma": BASE_THETA["gamma"],
                "iota1": np.log(BASE_THETA["iota"]),
                "iota2": 0.1,
                "rho": BASE_THETA["rho"],
                "sigmaSE": BASE_THETA["sigmaSE"],
                "psi": BASE_THETA["psi"],
                "cohort": BASE_THETA["cohort"],
                "amplitude": BASE_THETA["amplitude"],
                "S_0": BASE_THETA["S_0"],
                "E_0": BASE_THETA["E_0"],
                "I_0": BASE_THETA["I_0"],
                "R_0": BASE_THETA["R_0"],
            },
        ),
    ],
)
def test_other_models(model, theta):
    key = jax.random.key(0)
    mod_obj = pp.models.UKMeasles.pomp(
        unit="London",
        theta=pp.PompParameters(theta),
        model=model,
        clean=True,
    )
    mod_obj.simulate(key=key, nsim=1)
    mod_obj.pfilter(J=2, key=key)
    assert not np.isnan(mod_obj.results()["logLik"]).any()


def test_measles_sim(london):
    """Simulated measles case counts must be non-negative and finite."""
    measles = london
    _, obs = measles.simulate(key=DEFAULT_KEY, nsim=1)

    values = obs.drop(columns=["theta_idx", "sim", "time"]).to_numpy()
    assert np.all(np.isfinite(values)), "simulated observations must be finite"
    assert np.all(values >= 0), "measles case counts cannot be negative"


def test_measles_pfilter(london):
    """The filter returns a finite negative log-likelihood in both precisions."""
    measles = london

    measles.pfilter(J=DEFAULT_J, key=DEFAULT_KEY)
    ll_f32 = np.asarray(measles.results_history[-1].payload["logLiks"])
    assert np.all(np.isfinite(ll_f32)), f"non-finite logLik: {ll_f32}"
    assert np.all(ll_f32 < 0)

    # x64 is global state, so restore it even if the filter raises; otherwise
    # every later test in this worker would silently run in double precision.
    jax.config.update("jax_enable_x64", True)
    try:
        measles.pfilter(J=DEFAULT_J, key=DEFAULT_KEY)
    finally:
        jax.config.update("jax_enable_x64", False)

    ll_f64 = np.asarray(measles.results_history[-1].payload["logLiks"])
    assert np.all(np.isfinite(ll_f64)), f"non-finite logLik in x64: {ll_f64}"


def test_measles_mif(london, default_rw_sd):
    """mif yields one finite trace row per iteration."""
    measles = london
    measles.mif(
        J=DEFAULT_J,
        key=DEFAULT_KEY,
        M=DEFAULT_M,
        rw_sd=default_rw_sd.geometric_cooling(a=DEFAULT_A),
    )

    traces = measles.results_history[-1].payload["traces"]
    assert traces.sizes["iteration"] == DEFAULT_M + 1
    for name in measles.canonical_param_names:
        values = np.asarray(traces.sel(variable=name))
        assert np.all(np.isfinite(values)), f"non-finite trace for {name}"


def test_measles_clean():
    data = pp.models.UKMeasles.subset(clean=True)
    london_cleaned = np.isnan(
        data["measles"]
        .loc[
            (data["measles"]["unit"] == "London")
            & (data["measles"]["date"] == "1955-08-26"),
            "cases",
        ]
        .values
    )
    assert london_cleaned
    london_cleaned2 = np.isnan(
        data["measles"]
        .loc[
            (data["measles"]["unit"] == "London")
            & (data["measles"]["date"] == "1955-08-19"),
            "cases",
        ]
        .values
    )
    assert not london_cleaned2


def test_measles_invalid_interp_method():
    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]
    with pytest.raises(ValueError, match="interp_method invalid_method not recognized"):
        pp.models.UKMeasles.pomp(
            unit="London",
            theta=pp.PompParameters(theta),
            interp_method="invalid_method",  # type: ignore
        )


def test_measles_covariates_r_alignment():
    import os

    import pandas as pd

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "measles", "r_covariates.csv"
    )
    r_covars = pd.read_csv(csv_path)

    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]

    for unit in ["London", "Halesworth"]:
        measles = pp.models.UKMeasles.pomp(
            unit=unit,
            theta=pp.PompParameters(theta),
            clean=True,
            model="001b",
            interp_method="shifted_splines",
        )
        covars_df = measles.covars
        assert covars_df is not None
        py_covars = covars_df.reset_index()
        r_unit = r_covars[r_covars["unit"] == unit]

        # Merge on time to align evaluation points
        merged = pd.merge(py_covars, r_unit, on="time", suffixes=("_py", "_r"))
        assert len(merged) > 0

        # Check alignment: pop is extremely close, birthrate is within ~1.5% due to
        # small differences in spline libraries between Python/SciPy and R
        pop_py = np.asarray(merged["pop_py"])
        pop_r = np.asarray(merged["pop_r"])
        np.testing.assert_allclose(pop_py, pop_r, rtol=1e-5, atol=0.0)

        # Print statements for debugging
        diffs = np.abs(pop_py - pop_r)
        print(f"{unit} pop max difference: {diffs.max()}")
        print(f"{unit} pop mean difference: {diffs.mean()}")

        birthrate_py = np.asarray(merged["birthrate_py"])
        birthrate_r = np.asarray(merged["birthrate_r"])
        np.testing.assert_allclose(birthrate_py, birthrate_r, rtol=1.5e-2, atol=0.0)

        # Print statements for debugging
        diffs = np.abs(birthrate_py - birthrate_r)
        print(f"{unit} birthrate max difference: {diffs.max()}")
        print(f"{unit} birthrate mean difference: {diffs.mean()}")


def test_measles_panel_pomp():
    AK_mles = pp.models.UKMeasles.AK_mles()
    unit_specific = AK_mles[["London", "Hastings"]]
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])

    panel = pp.models.UKMeasles.panel_pomp(
        units=["London", "Hastings"],
        theta=theta,
        clean=True,
    )

    assert isinstance(panel, pp.PanelPomp)
    assert "London" in panel.unit_objects
    assert "Hastings" in panel.unit_objects

    london_covars = panel.unit_objects["London"].covars
    hastings_covars = panel.unit_objects["Hastings"].covars
    assert london_covars is not None
    assert hastings_covars is not None

    london_log_pop = float(london_covars["log_pop_1950"].iloc[0])
    hastings_log_pop = float(hastings_covars["log_pop_1950"].iloc[0])

    mean_val = (london_log_pop + hastings_log_pop) / 2.0
    sd_val = np.std([london_log_pop, hastings_log_pop], ddof=1)

    london_std_val = (london_log_pop - mean_val) / sd_val
    hastings_std_val = (hastings_log_pop - mean_val) / sd_val

    np.testing.assert_allclose(
        float(london_covars["std_log_pop_1950"].iloc[0]),
        london_std_val,
    )
    np.testing.assert_allclose(
        float(hastings_covars["std_log_pop_1950"].iloc[0]),
        hastings_std_val,
    )


def test_uk_measles_units():
    units = pp.models.UKMeasles.units()
    assert isinstance(units, list)
    assert len(units) > 0
    assert all(isinstance(u, str) for u in units)
    assert "London" in units
    assert "Liverpool" in units
    assert units == sorted(units)


def test_evaluate_spline_right_extrapolation():
    """evaluate_spline_with_linear_extrapolation should linearly extrapolate
    beyond x_max (the right_mask branch), matching the spline's value/slope
    at x_max."""
    from scipy.interpolate import make_smoothing_spline

    from pypomp.models.measles.uk_measles import (
        evaluate_spline_with_linear_extrapolation,
    )

    x = np.arange(0, 12, dtype=float)
    y = np.sin(x / 2.0) + 0.1 * x
    spline = make_smoothing_spline(x, y)
    x_min, x_max = float(x.min()), float(x.max())

    x_eval = np.array([x_max + 1.0, x_max + 3.0])
    y_eval = evaluate_spline_with_linear_extrapolation(spline, x_eval, x_min, x_max)

    deriv = spline.derivative()
    y_at_max = float(spline(x_max))
    dy_at_max = float(deriv(x_max))
    expected = y_at_max + dy_at_max * (x_eval - x_max)

    np.testing.assert_allclose(y_eval, expected)

    # Also cover the no-extrapolation-needed path for completeness.
    x_inside = np.array([x_min, (x_min + x_max) / 2.0, x_max])
    y_inside = evaluate_spline_with_linear_extrapolation(spline, x_inside, x_min, x_max)
    np.testing.assert_allclose(y_inside, spline(x_inside))


def test_measles_linear_interp_method():
    """The 'linear' interp_method branch (np.interp) should produce finite
    covariates, as an alternative to the default shifted_splines method."""
    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]
    measles = pp.models.UKMeasles.pomp(
        unit="London",
        theta=pp.PompParameters(theta),
        clean=True,
        model="001b",
        interp_method="linear",
    )
    covars_df = measles.covars
    assert covars_df is not None
    assert np.all(np.isfinite(covars_df["pop"]))
    assert np.all(np.isfinite(covars_df["birthrate"]))
    assert len(covars_df) > 0


def test_measles_pomp_missing_params():
    """Pomp should raise a clear ValueError when theta is missing parameters
    required by the selected model variant (here, model '002' needs iota1/
    iota2 instead of 'iota')."""
    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]
    with pytest.raises(ValueError, match="Missing required parameters"):
        pp.models.UKMeasles.pomp(
            unit="London",
            theta=pp.PompParameters(theta),
            model="002",
        )


def test_measles_panelpomp_theta_type_error():
    """PanelPomp should reject a theta that isn't a PanelParameters instance."""
    with pytest.raises(TypeError, match="theta must be a PanelParameters instance"):
        pp.models.UKMeasles.panel_pomp(
            units=["London"],
            theta={"shared": None, "unit_specific": None},  # type: ignore
        )


def test_measles_panelpomp_missing_params():
    """PanelPomp should raise a clear ValueError when a unit's parameters are
    missing entries required by the selected model variant."""
    AK_mles = pp.models.UKMeasles.AK_mles()
    unit_specific = AK_mles[["London"]]  # has 'iota', not 'iota1'/'iota2'
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])

    with pytest.raises(ValueError, match="Missing required parameters for unit"):
        pp.models.UKMeasles.panel_pomp(
            units=["London"],
            theta=theta,
            model="002",
        )


def test_measles_panelpomp_shared_params():
    """PanelPomp should correctly merge shared parameters (not just
    unit-specific ones) when building each unit's Pomp object."""
    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]

    shared_params = ["gamma", "cohort"]
    shared_df = pd.DataFrame(
        {"shared": [theta[p] for p in shared_params]}, index=pd.Index(shared_params)
    )
    specific_params = [p for p in theta if p not in shared_params]
    specific_df = pd.DataFrame(
        {
            "London": [theta[p] for p in specific_params],
            "Hastings": [theta[p] for p in specific_params],
        },
        index=pd.Index(specific_params),
    )
    panel_theta = pp.PanelParameters(
        theta=[{"shared": shared_df, "unit_specific": specific_df}]
    )

    panel = pp.models.UKMeasles.panel_pomp(
        units=["London", "Hastings"],
        theta=panel_theta,
        clean=True,
    )

    assert isinstance(panel, pp.PanelPomp)
    assert set(panel.canonical_shared_param_names) == set(shared_params)
    assert "London" in panel.unit_objects
    assert "Hastings" in panel.unit_objects


def test_measles_panelpomp_single_unit_std_fallback():
    """With only one unit, std(log_pop_1950) is undefined, so PanelPomp
    should fall back to sd=1.0, making std_log_pop_1950 exactly 0."""
    AK_mles = pp.models.UKMeasles.AK_mles()
    unit_specific = AK_mles[["London"]]
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])

    panel = pp.models.UKMeasles.panel_pomp(
        units=["London"],
        theta=theta,
        clean=True,
    )
    london_covars = panel.unit_objects["London"].covars
    assert london_covars is not None
    std_val = float(london_covars["std_log_pop_1950"].iloc[0])
    assert std_val == 0.0


def test_measles_panelpomp_zero_std_fallback(monkeypatch):
    """When the computed std of log_pop_1950 across units is exactly zero,
    PanelPomp should fall back to sd=1.0 rather than dividing by zero."""
    real_std = np.std

    def fake_std(a, *args, **kwargs):
        if kwargs.get("ddof") == 1:
            return 0.0
        return real_std(a, *args, **kwargs)

    monkeypatch.setattr(uk_measles_mod.np, "std", fake_std)

    AK_mles = pp.models.UKMeasles.AK_mles()
    unit_specific = AK_mles[["London", "Hastings"]]
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])

    panel = pp.models.UKMeasles.panel_pomp(
        units=["London", "Hastings"],
        theta=theta,
        clean=True,
    )

    london_covars = panel.unit_objects["London"].covars
    hastings_covars = panel.unit_objects["Hastings"].covars
    assert london_covars is not None
    assert hastings_covars is not None

    london_log_pop = float(london_covars["log_pop_1950"].iloc[0])
    hastings_log_pop = float(hastings_covars["log_pop_1950"].iloc[0])
    mean_val = (london_log_pop + hastings_log_pop) / 2.0

    # sd fallback is 1.0, so std_log_pop_1950 == log_pop - mean
    np.testing.assert_allclose(
        float(london_covars["std_log_pop_1950"].iloc[0]),
        london_log_pop - mean_val,
    )
    np.testing.assert_allclose(
        float(hastings_covars["std_log_pop_1950"].iloc[0]),
        hastings_log_pop - mean_val,
    )


def test_measles_log_pop_1950_fallback(monkeypatch):
    """When a unit's demography table has no row for year 1950 (e.g. data
    starting later than 1950), Pomp should fall back to using the earliest
    available year for log_pop_1950 instead of raising/crashing."""
    demog_years = list(range(1951, 1959))
    fake_demog = pd.DataFrame(
        {
            "year": demog_years,
            "unit": ["TestTown"] * len(demog_years),
            "pop": [100000 + 1000 * i for i in range(len(demog_years))],
            "births": [2000 + 10 * i for i in range(len(demog_years))],
        }
    )

    dates = pd.date_range("1951-01-01", "1956-12-31", freq="7D")
    rng = np.random.default_rng(0)
    fake_measles = pd.DataFrame(
        {
            "date": dates,
            "unit": ["TestTown"] * len(dates),
            "cases": rng.integers(0, 50, size=len(dates)),
        }
    )
    fake_coord = pd.DataFrame({"unit": ["TestTown"], "long": [0.0], "lat": [51.0]})

    fake_data = {"measles": fake_measles, "demog": fake_demog, "coord": fake_coord}

    monkeypatch.setattr(
        pp.models.UKMeasles, "_get_data", classmethod(lambda cls: fake_data)
    )

    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]

    measles = pp.models.UKMeasles.pomp(
        unit="TestTown",
        theta=pp.PompParameters(theta),
        model="001b",
        first_year=1951,
        last_year=1956,
    )

    assert measles.covars is not None
    expected_log_pop = np.log(float(fake_demog["pop"].iloc[0]))
    actual_log_pop = float(measles.covars["log_pop_1950"].iloc[0])
    np.testing.assert_allclose(actual_log_pop, expected_log_pop)
