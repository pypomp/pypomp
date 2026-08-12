import pickle
from copy import deepcopy

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp


@pytest.fixture(scope="module")
def setup_module():
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
def setup(setup_module):
    params = deepcopy(setup_module)
    params["LG"] = deepcopy(setup_module["LG"])
    params["rw_sd"] = deepcopy(setup_module["rw_sd"])
    return params


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


def _valid_pomp_kwargs():
    """A complete, valid Pomp argument set built from raw callables.

    Pomp takes unwrapped functions, so these cannot be read back off an
    existing model via getattr (that yields the wrapped _RInit/_RProc/_DMeas).
    """

    def rinit(theta_, key, covars, t0):
        return {"X": 0.0}

    def rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"]}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(
            Y_["Y"], loc=theta_["mu"], scale=theta_["sigma"]
        )

    return {
        "ys": pd.DataFrame({"Y": [1.0]}, index=[1.0]),
        "theta": pp.PompParameters({"mu": 0.0, "sigma": 0.1}),
        "statenames": ["X"],
        "t0": 0.0,
        "rinit": rinit,
        "rproc": rproc,
        "dmeas": dmeas,
        "nstep": 1,
    }


def test_valid_initialization_baseline():
    """The baseline used by test_invalid_initialization must itself construct.

    Without this, a change that made every construction fail would leave the
    invalid-argument test below passing for the wrong reason.
    """
    assert isinstance(pp.Pomp(**_valid_pomp_kwargs()), pp.Pomp)


@pytest.mark.parametrize(
    "arg, exc, message",
    [
        ("ys", TypeError, "ys must be a pandas DataFrame"),
        ("theta", TypeError, "theta must be a PompParameters instance"),
        ("statenames", ValueError, "statenames must be provided"),
        ("t0", TypeError, "must be a string or a real number"),
        ("rinit", TypeError, "not a callable object"),
        ("rproc", TypeError, "not a callable object"),
        ("dmeas", ValueError, "at least one of dmeas or rmeas"),
    ],
)
def test_invalid_initialization(arg, exc, message):
    """Each required argument is rejected with its own specific error."""
    kwargs = _valid_pomp_kwargs()
    kwargs[arg] = None
    with pytest.raises(exc, match=message):
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
    param_cols = list(LG.theta[0].keys())
    assert (
        traces[param_cols].iloc[-1].values.tolist()
        == traces[param_cols].iloc[-2].values.tolist()
    )


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
    assert unpickled.theta.params(as_list=True) == pomp.theta.params(as_list=True)
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

    with pytest.raises(ValueError, match="No valid log-likelihoods available to prune"):
        LG.__class__(
            ys=LG.ys.copy(),
            theta=pp.PompParameters(LG.theta.params(as_list=True)[0].copy()),
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


def dummy_rinit(theta_, key, covars, t0):
    val = next(iter(theta_.values()))
    return {"X": val}


def dummy_rproc(X_, theta_, key, covars, t, dt):
    return {"X": X_["X"] + theta_["sigma"] * jax.random.normal(key, ())}


def dummy_dmeas(Y_, X_, theta_, covars, t):
    return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=0.1)


def dummy_rmeas(X_, theta_, key, covars, t):
    return {"y": X_["X"] + 0.1 * jax.random.normal(key, ())}


def dummy_dprior(theta_):
    return 0.0


@pytest.fixture
def base_pomp():
    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp.fresh_key = jax.random.key(1)
    return pomp


# --- Extra Pomp Tests ---


def test_init_validation():
    """Test invalid arguments during Pomp initialization."""
    ys = pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0])
    theta = pp.PompParameters({"X0": 0.0, "sigma": 0.1})

    # ys must be a DataFrame
    with pytest.raises(TypeError, match="ys must be a pandas DataFrame"):
        pp.Pomp(
            ys="not_df",  # type: ignore
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # covars must be a DataFrame
    with pytest.raises(TypeError, match="covars must be a pandas DataFrame"):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
            covars="not_df",  # type: ignore
        )

    # theta must be a PompParameters instance
    with pytest.raises(TypeError, match="theta must be a PompParameters instance"):
        pp.Pomp(
            ys=ys,
            theta={"X0": 0.0},  # type: ignore
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # statenames must be provided
    with pytest.raises(ValueError, match="statenames must be provided"):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=None,  # type: ignore
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # statenames must be list of strings
    with pytest.raises(
        ValueError, match="statenames must be a tuple or list of strings"
    ):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=[1, 2],  # type: ignore
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # accumvars validation
    with pytest.raises(
        ValueError, match="accumvars must be a tuple or list of strings"
    ):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
            accumvars=[1],  # type: ignore
        )

    with pytest.raises(ValueError, match="all accumvars must be in statenames"):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
            accumvars=["Y"],
        )

    # both dmeas and rmeas cannot be None
    with pytest.raises(
        ValueError, match="You must supply at least one of dmeas or rmeas"
    ):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )


def test_theta_getter_setter(base_pomp):
    """Test getter/setter validation for theta property."""
    # Getter when _theta is None
    base_pomp._theta = None
    with pytest.raises(ValueError, match="Model parameters have not been set"):
        _ = base_pomp.theta

    # Setter type check
    with pytest.raises(TypeError, match="theta must be a PompParameters instance"):
        base_pomp.theta = {"X0": 0.0}


def test_prepare_theta_input(base_pomp):
    """Test _prepare_theta_input validation."""
    # TypeError for wrong class
    with pytest.raises(
        TypeError, match="theta must be a PompParameters object or None"
    ):
        base_pomp._prepare_theta_input("invalid")

    # ValueError for mismatched param names
    bad_theta = pp.PompParameters({"unknown": 1.0})
    with pytest.raises(
        ValueError, match="theta parameter names must match canonical_param_names"
    ):
        base_pomp._prepare_theta_input(bad_theta)


def test_update_fresh_key(base_pomp):
    """Test _update_fresh_key validation."""
    # ValueError when both keys are None
    base_pomp.fresh_key = None
    with pytest.raises(
        ValueError,
        match="Both the key argument and the fresh_key attribute are None",
    ):
        base_pomp._update_fresh_key(None)


def test_simulate_edge_cases(base_pomp):
    """Test simulate with custom times, warnings, and as_pomp=True."""
    # 1. rmeas is None raises ValueError
    pomp_no_rmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="self.rmeas cannot be None"):
        pomp_no_rmeas.simulate(key=jax.random.key(1))

    # 2. nsim > 1 when as_pomp is True triggers UserWarning
    with pytest.warns(UserWarning, match="as_pomp is True, but nsim > 1"):
        pomp_copy = base_pomp.simulate(key=jax.random.key(1), nsim=5, as_pomp=True)
    assert isinstance(pomp_copy, pp.Pomp)

    # 3. simulate with custom times
    times = jnp.array([1.5, 2.5])
    states, obs = base_pomp.simulate(key=jax.random.key(1), times=times)
    assert set(states["time"].unique()) == {0.0, 1.5, 2.5}
    assert set(obs["time"].unique()) == {1.5, 2.5}


def test_filtering_methods_validation(base_pomp):
    # Test validations for pfilter, mif, and train when components are missing.
    pomp_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp_no_dmeas.fresh_key = jax.random.key(1)

    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp_no_dmeas.pfilter(J=10)

    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp_no_dmeas.mif(J=10, M=1, rw_sd=pp.RWSigma({"X0": 0.01, "sigma": 0.01}))

    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp_no_dmeas.train(J=10, M=1, eta=pp.LearningRate({"X0": 0.1, "sigma": 0.1}))


def test_dpop_train_validation(base_pomp):
    """Test input validations for dpop_train."""
    eta = pp.LearningRate({"X0": 0.1, "sigma": 0.1})

    # 1. missing dmeas
    pomp_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp_no_dmeas.fresh_key = jax.random.key(1)
    with pytest.raises(
        ValueError, match="dpop_train requires self.dmeas to be not None"
    ):
        pomp_no_dmeas.dpop_train(J=5, M=1, eta=eta, process_weight_state="logw")

    # 2. invalid eta
    with pytest.raises(TypeError, match="eta must be a LearningRate object"):
        base_pomp.dpop_train(J=5, M=1, eta="not_lr", process_weight_state="logw")

    # 3. missing process_weight_state
    with pytest.raises(ValueError, match="dpop_train requires a process-weight state"):
        base_pomp.dpop_train(J=5, M=1, eta=eta, process_weight_state=None)

    # 4. process_weight_state not in statenames
    with pytest.raises(ValueError, match="not found in statenames"):
        base_pomp.dpop_train(J=5, M=1, eta=eta, process_weight_state="non_existent")

    # 5. Test valid call with optimizer='SGD'
    # Add logw to states
    pomp_dpop = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=lambda theta_, key, covars, t0: {"X": theta_["X0"], "logw": 0.0},
        rproc=lambda X_, theta_, key, covars, t, dt: {
            "X": X_["X"],
            "logw": X_["logw"] + 0.1,
        },
        dmeas=dummy_dmeas,
        statenames=["X", "logw"],
        t0=0.0,
        nstep=1,
    )
    pomp_dpop.fresh_key = jax.random.key(1)
    pomp_dpop.results_history.clear()
    ret = pomp_dpop.dpop_train(
        J=2, M=2, eta=eta, process_weight_state="logw", optimizer=pp.SGD()
    )
    assert ret is None
    res = pomp_dpop.results_history[-1]
    assert res.method == "dpop_train"
    assert res.kind == "trace"
    traces = res.traces()
    assert len(traces) > 0


def test_merge_validation(base_pomp):
    """Test merge validation exceptions."""
    # 1. No arguments
    with pytest.raises(ValueError, match="At least one Pomp object must be provided"):
        pp.Pomp.merge()

    # 2. Different type
    with pytest.raises(TypeError, match="All merged objects must be of type Pomp"):
        pp.Pomp.merge(base_pomp, "not_pomp")  # type: ignore

    # 3. Mismatched parameter names
    p2_diff_params = pp.Pomp(
        ys=base_pomp.ys,
        theta=pp.PompParameters({"diff_name": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same canonical_param_names"):
        pp.Pomp.merge(base_pomp, p2_diff_params)

    # 4. Mismatched statenames
    p2_diff_states = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=lambda theta_, key, covars, t0: {"Y": theta_["X0"]},
        rproc=lambda X_, theta_, key, covars, t, dt: {"Y": X_["Y"]},
        dmeas=lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["Y"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same statenames"):
        pp.Pomp.merge(base_pomp, p2_diff_states)

    # 5. Mismatched ys
    p2_diff_ys = pickle.loads(pickle.dumps(base_pomp))
    p2_diff_ys.ys = pd.DataFrame({"y": [3.0, 4.0]}, index=[1.0, 2.0])
    with pytest.raises(ValueError, match="same ys data"):
        pp.Pomp.merge(base_pomp, p2_diff_ys)

    # 6. Mismatched t0
    p2_diff_t0 = pickle.loads(pickle.dumps(base_pomp))
    p2_diff_t0.t0 = 5.0
    with pytest.raises(ValueError, match="same t0"):
        pp.Pomp.merge(base_pomp, p2_diff_t0)

    # 7. Mismatched dmeas None vs not None
    pomp_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same dmeas"):
        pp.Pomp.merge(base_pomp, pomp_no_dmeas)

    # 8. Mismatched theta is None
    p2_no_theta = pickle.loads(pickle.dumps(base_pomp))
    p2_no_theta._theta = None
    with pytest.raises(
        ValueError, match="Cannot merge Pomp objects with no parameters"
    ):
        pp.Pomp.merge(base_pomp, p2_no_theta)


def test_eq_comparisons(base_pomp):
    """Test all inequality paths in __eq__."""
    assert base_pomp != "not_pomp"

    # 1. canonical_param_names mismatch
    p2_diff_params = pp.Pomp(
        ys=base_pomp.ys,
        theta=pp.PompParameters({"diff_name": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p2_diff_params

    # 2. one theta is None
    p2 = pickle.loads(pickle.dumps(base_pomp))
    p2._theta = None
    assert base_pomp != p2

    # 3. different theta values
    p3 = pickle.loads(pickle.dumps(base_pomp))
    p3.theta = pp.PompParameters({"X0": 1.0, "sigma": 0.1})
    assert base_pomp != p3

    # 4. different ys
    p4 = pickle.loads(pickle.dumps(base_pomp))
    p4.ys = pd.DataFrame({"y": [3.0, 4.0]}, index=[1.0, 2.0])
    assert base_pomp != p4

    # 5. different statenames
    p5 = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=lambda theta_, key, covars, t0: {"Y": theta_["X0"]},
        rproc=lambda X_, theta_, key, covars, t, dt: {"Y": X_["Y"]},
        dmeas=lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["Y"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p5

    # 6. different t0
    p6 = pickle.loads(pickle.dumps(base_pomp))
    p6.t0 = 10.0
    assert base_pomp != p6

    # 7. different fresh_key
    p7 = pickle.loads(pickle.dumps(base_pomp))
    p7.fresh_key = jax.random.key(99)
    assert base_pomp != p7


def test_pickle_setstate_fallback_warning(base_pomp):
    """Test that unpickling issues a UserWarning when a function fails to reconstruct."""
    state = base_pomp.__getstate__()

    # Corrupt the bytes of rinit so unpickling fails
    state["_rinit_func_bytes"] = b"invalid_pickle_bytes"

    with pytest.warns(UserWarning, match="Failed to reconstruct rinit function"):
        pomp_unpickled = pickle.loads(pickle.dumps(base_pomp))
        del pomp_unpickled.rinit
        # Directly trigger __setstate__ with corrupted state
        pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rinit is None


@pytest.fixture
def dprior_pomp():
    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        dprior=dummy_dprior,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp.fresh_key = jax.random.key(1)
    return pomp


def test_dprior_construction_and_pickle_roundtrip(dprior_pomp):
    """Constructing with dprior= wraps it in a _DPrior; pickling round-trips it."""
    assert dprior_pomp.dprior is not None
    unpickled = pickle.loads(pickle.dumps(dprior_pomp))
    assert unpickled.dprior is not None
    assert dprior_pomp == unpickled


def test_eq_covars_and_derived_arrays(base_pomp):
    """Cover __eq__ branches for covars, covars_extended, nstep/dt arrays."""
    covars = pd.DataFrame({"c": [1.0, 2.0]}, index=[1.0, 2.0])

    p_with_covars_a = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
        covars=covars,
    )
    # one has covars, the other does not
    assert base_pomp != p_with_covars_a

    # both have covars, but different values
    p_with_covars_b = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
        covars=pd.DataFrame({"c": [3.0, 4.0]}, index=[1.0, 2.0]),
    )
    assert p_with_covars_a != p_with_covars_b

    # differing _covars_extended directly (bypassing normal construction)
    p_covars_extended_diff = deepcopy(p_with_covars_a)
    p_covars_extended_diff._covars_extended = (
        p_with_covars_a._covars_extended + 1.0
        if p_with_covars_a._covars_extended is not None
        else None
    )
    assert p_with_covars_a != p_covars_extended_diff

    p_covars_extended_none = deepcopy(p_with_covars_a)
    p_covars_extended_none._covars_extended = None
    assert p_with_covars_a != p_covars_extended_none

    # differing nstep_array / dt_array_extended / max_steps_per_interval
    p_nstep_diff = deepcopy(base_pomp)
    p_nstep_diff._nstep_array = base_pomp._nstep_array + 1
    assert base_pomp != p_nstep_diff

    p_dt_diff = deepcopy(base_pomp)
    p_dt_diff._dt_array_extended = base_pomp._dt_array_extended + 1.0
    assert base_pomp != p_dt_diff

    p_max_steps_diff = deepcopy(base_pomp)
    p_max_steps_diff._max_steps_per_interval = base_pomp._max_steps_per_interval + 1
    assert base_pomp != p_max_steps_diff


def test_eq_model_components(base_pomp):
    """Cover __eq__ branches for rinit, rproc, dmeas, rmeas, results_history, par_trans."""

    def other_rinit(theta_, key, covars, t0):
        val = next(iter(theta_.values()))
        return {"X": val + 1.0}

    p_diff_rinit = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=other_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_rinit

    def other_rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"] + 1.0}

    p_diff_rproc = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=other_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_rproc

    # dmeas: one None, other not
    p_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_no_dmeas

    # dmeas: both present but different
    def other_dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=1.0)

    p_diff_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=other_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_dmeas

    # rmeas: one None, other not
    p_no_rmeas = deepcopy(base_pomp)
    p_no_rmeas.rmeas = None
    assert base_pomp != p_no_rmeas

    # rmeas: both present but different
    def other_rmeas(X_, theta_, key, covars, t):
        return {"y": X_["X"] + 1.0}

    p_diff_rmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=other_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_rmeas

    # results_history mismatch (pfilter also mutates theta, so build the
    # comparison history separately and graft it on to isolate this branch)
    p_diff_history = deepcopy(base_pomp)
    p_with_results = deepcopy(base_pomp)
    p_with_results.pfilter(J=5, reps=1, key=jax.random.key(0))
    p_diff_history.results_history = p_with_results.results_history
    assert base_pomp != p_diff_history

    # par_trans mismatch
    def to_est(theta_):
        return {k: v + 1.0 for k, v in theta_.items()}

    def from_est(theta_):
        return {k: v - 1.0 for k, v in theta_.items()}

    p_diff_par_trans = deepcopy(base_pomp)
    p_diff_par_trans.par_trans = pp.ParTrans(to_est, from_est)
    assert base_pomp != p_diff_par_trans

    # fresh_key: one is None
    p_no_key = deepcopy(base_pomp)
    p_no_key.fresh_key = None
    assert base_pomp != p_no_key


def test_merge_component_mismatches(base_pomp):
    """Cover merge() validation branches for rinit/rproc, dmeas, rmeas, par_trans."""

    def other_rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"] + 1.0}

    p_diff_rproc = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=other_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same rinit and rproc"):
        pp.Pomp.merge(base_pomp, p_diff_rproc)

    def other_dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=1.0)

    p_diff_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=other_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same dmeas"):
        pp.Pomp.merge(base_pomp, p_diff_dmeas)

    p_no_rmeas = deepcopy(base_pomp)
    p_no_rmeas.rmeas = None
    with pytest.raises(ValueError, match="same rmeas \\(both None"):
        pp.Pomp.merge(base_pomp, p_no_rmeas)

    def other_rmeas(X_, theta_, key, covars, t):
        return {"y": X_["X"] + 1.0}

    p_diff_rmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=other_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same rmeas\\."):
        pp.Pomp.merge(base_pomp, p_diff_rmeas)

    def to_est(theta_):
        return {k: v + 1.0 for k, v in theta_.items()}

    def from_est(theta_):
        return {k: v - 1.0 for k, v in theta_.items()}

    p_diff_par_trans = deepcopy(base_pomp)
    p_diff_par_trans.par_trans = pp.ParTrans(to_est, from_est)
    with pytest.raises(ValueError, match="same par_trans"):
        pp.Pomp.merge(base_pomp, p_diff_par_trans)


def test_setstate_fresh_key_reconstruction_failure(base_pomp):
    """A corrupted fresh_key payload should warn and fall back to None."""
    state = base_pomp.__getstate__()
    state["_fresh_key_data"] = jnp.zeros((3,), dtype=jnp.uint32)

    pomp_unpickled = pickle.loads(pickle.dumps(base_pomp))
    with pytest.warns(UserWarning, match="Failed to reconstruct JAX fresh_key"):
        pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.fresh_key is None


def test_setstate_legacy_by_reference_loading(base_pomp):
    """Legacy pickles referenced functions by module+name instead of bytes."""
    state = base_pomp.__getstate__()
    del state["_rinit_func_bytes"]
    state["_rinit_func_name"] = "dummy_rinit"
    state["_rinit_module"] = __name__

    pomp_unpickled = pickle.loads(pickle.dumps(base_pomp))
    del pomp_unpickled.rinit
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rinit is not None


def test_setstate_prewrapped_components(dprior_pomp):
    """If the pickled bytes already contain wrapped components, reuse them directly."""
    state = dprior_pomp.__getstate__()
    state["_rinit_func_bytes"] = cloudpickle.dumps(dprior_pomp.rinit)
    state["_rproc_func_bytes"] = cloudpickle.dumps(dprior_pomp.rproc)
    state["_dmeas_func_bytes"] = cloudpickle.dumps(dprior_pomp.dmeas)
    state["_rmeas_func_bytes"] = cloudpickle.dumps(dprior_pomp.rmeas)
    state["_dprior_func_bytes"] = cloudpickle.dumps(dprior_pomp.dprior)

    pomp_unpickled = pickle.loads(pickle.dumps(dprior_pomp))
    del pomp_unpickled.rinit
    del pomp_unpickled.rproc
    del pomp_unpickled.dmeas
    del pomp_unpickled.rmeas
    del pomp_unpickled.dprior
    pomp_unpickled.__setstate__(state)

    assert type(pomp_unpickled.rinit) is type(dprior_pomp.rinit)
    assert type(pomp_unpickled.rproc) is type(dprior_pomp.rproc)
    assert type(pomp_unpickled.dmeas) is type(dprior_pomp.dmeas)
    assert type(pomp_unpickled.rmeas) is type(dprior_pomp.rmeas)
    assert type(pomp_unpickled.dprior) is type(dprior_pomp.dprior)


def test_setstate_rproc_dt_and_nstep_both_present(base_pomp):
    """Cover the raw-function rproc reconstruction path when both dt and nstep are set."""
    state = base_pomp.__getstate__()
    state["_rproc_func_bytes"] = cloudpickle.dumps(base_pomp.rproc.original_func)
    state["_rproc_dt"] = 0.5
    state["_rproc_nstep"] = 3
    state["_rproc_accumvars"] = ["X"]

    pomp_unpickled = pickle.loads(pickle.dumps(base_pomp))
    del pomp_unpickled.rproc
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rproc is not None
    assert pomp_unpickled.rproc.nstep == 3


def test_setstate_rproc_missing_defaults_to_none(base_pomp):
    """If no rproc info is present at all in the pickled state, rproc defaults to None."""
    state = base_pomp.__getstate__()
    del state["_rproc_func_bytes"]

    pomp_unpickled = pickle.loads(pickle.dumps(base_pomp))
    del pomp_unpickled.rproc
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rproc is None


def test_setstate_dmeas_missing_defaults_to_none():
    """A Pomp with dmeas=None (rmeas-only) should pickle/unpickle with dmeas staying None."""
    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp.fresh_key = jax.random.key(1)
    assert pomp.dmeas is None

    unpickled = pickle.loads(pickle.dumps(pomp))
    assert unpickled.dmeas is None


def test_accumvars_success(base_pomp):
    """Valid accumvars are resolved to state-name indices at construction time."""
    pomp = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
        accumvars=["X"],
    )
    assert pomp._accumvars_indices == (0,)
    assert pomp.accumvars == ["X"]
