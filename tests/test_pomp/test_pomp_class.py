import jax
import numpy as np
import pickle
import pypomp as pp
import pytest
import jax.numpy as jnp
import pandas as pd


@pytest.fixture(scope="function")
def setup():
    LG = pp.models.LG()
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in LG.canonical_param_names},
        init_names=[],
    ).geometric_cooling(0.5)
    rw_sd.sigmas["R4"] = 0.0
    return {
        "LG": LG,
        "J": 2,
        "rw_sd": rw_sd,
        "a": 0.5,
        "M": 2,
        "key": jax.random.key(111),
        "theta": LG.theta,
        "fresh_key": LG.fresh_key,
    }


@pytest.fixture(scope="function")
def model(setup):
    LG = setup["LG"]
    LG.results_history.clear()
    LG.theta = setup["theta"]
    LG.fresh_key = setup["fresh_key"]
    return LG, setup


@pytest.fixture(scope="function")
def neapolitan_setup(setup):
    LG = setup["LG"]
    p = setup
    LG.pfilter(J=p["J"], reps=1, key=p["key"])
    LG.mif(J=p["J"], rw_sd=p["rw_sd"], M=p["M"], key=p["key"])
    LG.train(
        J=p["J"],
        M=1,
        eta=pp.LearningRate({n: 0.2 for n in LG.canonical_param_names}),
        key=p["key"],
    )
    return LG, setup


def test_invalid_initialization(model):
    LG, _ = model
    for arg in ["ys", "theta", "rinit", "rproc", "dmeas"]:
        with pytest.raises(Exception):
            kwargs = {
                k: getattr(LG, k) for k in ["ys", "theta", "rinit", "rproc", "dmeas"]
            }
            kwargs[arg] = None
            pp.Pomp(**kwargs)


def test_results(neapolitan_setup):
    LG, _ = neapolitan_setup
    expected_cols = {"theta_idx", "logLik", "se", *LG.theta[0].keys()}
    for i in range(3):
        res = LG.results(i)
        assert res.shape[0] == len(LG.theta)
        assert set(res.columns) == expected_cols


def test_sample_params():
    bounds = {"R0": (0.0, 100.0), "sigma": (0.0, 100.0), "gamma": (0.0, 100.0)}
    params = pp.Pomp.sample_params(bounds, 10, jax.random.key(1))
    assert isinstance(params, pp.PompParameters)
    assert len(params) == 10
    assert list(params[0].keys()) == list(bounds.keys())


@pytest.mark.parametrize("method", ["mif", "train"])
def test_theta_carryover(model, method):
    """Check that the parameters are carried over between method calls."""
    LG, p = model
    theta_order = list(LG.theta[0].keys())
    if method == "mif":
        LG.mif(J=p["J"], rw_sd=p["rw_sd"], M=p["M"], key=p["key"])
    else:
        LG.train(
            J=p["J"],
            M=1,
            eta=pp.LearningRate({n: 0.2 for n in LG.canonical_param_names}),
            key=p["key"],
        )

    assert theta_order == list(LG.theta[0].keys())
    LG.pfilter(J=p["J"], reps=2)
    assert list(LG.results_history[-1].theta[0].keys()) == theta_order

    traces_da = LG.results_history[-2].traces_da
    param_names = traces_da.coords["variable"].values[1:]
    last_row = traces_da.sel(theta_idx=0, iteration=traces_da.sizes["iteration"] - 1)
    last_val = [float(last_row.sel(variable=n).values) for n in param_names]

    assert list(LG.results_history[-1].theta[0].values()) == last_val
    traces = LG.traces()
    assert traces.iloc[-1, 4:].values.tolist() == traces.iloc[-2, 4:].values.tolist()


def test_pickle(model):
    LG, p = model
    LG.pfilter(J=p["J"], reps=1, key=p["key"])
    unpickled = pickle.loads(pickle.dumps(LG))
    assert LG == unpickled
    # Check that pickling works when rmeas is None and pfilter is called
    unpickled.rmeas = None
    pickle.loads(pickle.dumps(unpickled)).pfilter(J=p["J"], reps=1)


def test_pickle_by_value():
    """Test that unpickling works without reference to model mechanics."""

    def rinit_local(theta_, key, covars, t0):
        return {"X": theta_["X0"]}

    def rproc_local(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"] + theta_["sigma"] * jax.random.normal(key, ())}

    def dmeas_local(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=0.1)

    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0, "sigma": 0.1}),
        rinit=rinit_local,
        rproc=rproc_local,
        dmeas=dmeas_local,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp.fresh_key = jax.random.key(1)
    del rinit_local, rproc_local, dmeas_local

    unpickled = pickle.loads(pickle.dumps(pomp))
    assert unpickled.statenames == pomp.statenames
    assert float(unpickled.t0) == float(pomp.t0)
    assert unpickled.theta.params() == pomp.theta.params()
    assert jnp.array_equal(
        jax.random.key_data(unpickled.fresh_key), jax.random.key_data(pomp.fresh_key)
    )
    unpickled.pfilter(J=10, reps=1)
    assert unpickled.results(0)["logLik"].iloc[0] is not None


def test_prune(model):
    LG, p = model
    LG.pfilter(J=p["J"], reps=5, key=p["key"])
    orig_theta, orig_len = LG.theta, len(LG.theta)

    LG.prune(n=2, refill=True)
    assert len(LG.theta) == orig_len
    assert len(set(tuple(sorted(d.items())) for d in LG.theta)) <= 2

    LG.prune(n=1, refill=False)
    assert len(LG.theta) == 1

    LG.theta = orig_theta
    LG.prune(n=10, refill=False)
    assert len(LG.theta) == min(10, orig_len)

    with pytest.raises(ValueError):
        LG.__class__(
            ys=LG.ys.copy(),
            theta=pp.PompParameters(LG.theta.params()[0].copy()),
            rinit=LG.rinit.original_func,
            rproc=LG.rproc.original_func,
            dmeas=LG.dmeas.original_func,
            t0=LG.t0,
            nstep=LG.rproc.nstep,
            statenames=["X1", "X2"],
        ).prune(n=1)


def test_diagnostics(neapolitan_setup):
    LG, _ = neapolitan_setup
    assert len(LG.time()) == 3
    LG.print_summary()
    LG.print_metadata()


def test_merge(setup):
    p = setup
    LG1, LG2 = pp.models.LG(), pp.models.LG()
    k1, k2 = jax.random.split(p["key"])

    for obj, k in [(LG1, k1), (LG2, k2)]:
        obj.pfilter(theta=p["theta"], J=p["J"], reps=1, key=k)
        obj.mif(J=p["J"], M=p["M"], rw_sd=p["rw_sd"])
        obj.train(
            J=p["J"],
            M=1,
            eta=pp.LearningRate({n: 0.2 for n in obj.canonical_param_names}),
        )

    merged = pp.Pomp.merge(LG1, LG2)
    assert len(merged.theta) == len(LG1.theta) + len(LG2.theta)
    assert len(merged.results_history) == len(LG1.results_history)


def test_traces(neapolitan_setup):
    LG, p = neapolitan_setup
    traces = LG.traces()
    assert isinstance(traces, pd.DataFrame)

    expected_cols = {
        "theta_idx",
        "iteration",
        "method",
        "logLik",
        *LG.canonical_param_names,
    }
    assert expected_cols.issubset(set(traces.columns))

    methods = traces["method"].unique().tolist()
    assert "pfilter" in methods
    assert "mif" in methods
    assert "train" in methods

    for theta_idx in traces["theta_idx"].unique():
        sub_df = traces[traces["theta_idx"] == theta_idx]
        iterations = sub_df["iteration"].tolist()
        assert iterations == sorted(iterations)
        assert max(iterations) > 0


def test_pomp_parameters_indexing(model):
    LG, p = model
    # Set up some logLik values to check slicing behavior
    log_liks = np.array([-10.0, -20.0, -30.0, -40.0])
    # Create PompParameters with 4 replicates
    params_dicts = [
        {"R0": 1.0, "sigma": 0.1},
        {"R0": 2.0, "sigma": 0.2},
        {"R0": 3.0, "sigma": 0.3},
        {"R0": 4.0, "sigma": 0.4},
    ]
    params = pp.PompParameters(params_dicts, logLik=log_liks)

    # 1. Test single integer indexing
    assert params[0] == {"R0": 1.0, "sigma": 0.1}
    assert params[2] == {"R0": 3.0, "sigma": 0.3}

    # 2. Test slice indexing (this previously crashed with ValueError)
    slice_params = params[1:3]
    assert isinstance(slice_params, pp.PompParameters)
    assert len(slice_params) == 2
    assert slice_params[0] == {"R0": 2.0, "sigma": 0.2}
    assert slice_params[1] == {"R0": 3.0, "sigma": 0.3}
    np.testing.assert_allclose(slice_params.logLik, [-20.0, -30.0])

    # 3. Test list of indices indexing
    list_params = params[[0, 3]]
    assert isinstance(list_params, pp.PompParameters)
    assert len(list_params) == 2
    assert list_params[0] == {"R0": 1.0, "sigma": 0.1}
    assert list_params[1] == {"R0": 4.0, "sigma": 0.4}
    np.testing.assert_allclose(list_params.logLik, [-10.0, -40.0])

    # 4. Test mutation safety (dicts copy constructors)
    user_dicts = [{"R0": 1.5}]
    pp_params = pp.PompParameters(user_dicts)
    pp_params[0]["R0"] = 9.9
    # Slicing/copying should not mutate original user_dicts input
    assert user_dicts[0]["R0"] == 1.5


def test_pomp_parameters_data_array_dimension_expansion():
    import xarray as xr

    # Test 1D DataArray
    da_1d = xr.DataArray(
        [1.0, 2.0], dims=["parameter"], coords={"parameter": ["R0", "sigma"]}
    )
    p_1d = pp.PompParameters(da_1d)
    assert p_1d.get_param_names() == ["R0", "sigma"]
    assert len(p_1d) == 1
    assert p_1d[0] == {"R0": 1.0, "sigma": 2.0}

    # Test 2D DataArray
    da_2d = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=["theta_idx", "parameter"],
        coords={"theta_idx": [0, 1], "parameter": ["R0", "sigma"]},
    )
    p_2d = pp.PompParameters(da_2d)
    assert p_2d.get_param_names() == ["R0", "sigma"]
    assert len(p_2d) == 2
    assert p_2d[0] == {"R0": 1.0, "sigma": 2.0}
    assert p_2d[1] == {"R0": 3.0, "sigma": 4.0}
