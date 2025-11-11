import jax
import pickle
import pypomp as pp
import pytest


@pytest.fixture(scope="module")
def simple_setup():
    LG = pp.LG()
    J = 2
    rw_sd = pp.RWSigma(
        sigmas={
            "A1": 0.02,
            "A2": 0.02,
            "A3": 0.02,
            "A4": 0.02,
            "C1": 0.02,
            "C2": 0.02,
            "C3": 0.02,
            "C4": 0.02,
            "Q1": 0.02,
            "Q2": 0.02,
            "Q3": 0.02,
            "Q4": 0.02,
            "R1": 0.02,
            "R2": 0.02,
            "R3": 0.02,
            "R4": 0.0,
        },
        init_names=[],
    )
    a = 0.5
    M = 2
    key = jax.random.key(111)
    theta = LG.theta
    fresh_key = LG.fresh_key
    return LG, rw_sd, J, a, M, key, theta, fresh_key


@pytest.fixture(scope="module")
def neapolitan_setup():
    LG = pp.LG()
    J = 2
    a = 0.5
    M = 2
    key = jax.random.key(111)
    rw_sd = pp.RWSigma(
        sigmas={
            "A1": 0.02,
            "A2": 0.02,
            "A3": 0.02,
            "A4": 0.02,
            "C1": 0.02,
            "C2": 0.02,
            "C3": 0.02,
            "C4": 0.02,
            "Q1": 0.02,
            "Q2": 0.02,
            "Q3": 0.02,
            "Q4": 0.02,
            "R1": 0.02,
            "R2": 0.02,
            "R3": 0.02,
            "R4": 0.0,
        },
        init_names=[],
    )

    LG.pfilter(J=J, reps=1, key=key)
    LG.mif(
        J=J,
        rw_sd=rw_sd,
        M=M,
        a=a,
        key=key,
    )
    LG.train(J=J, M=1, eta=0.2, key=key)
    results_history = LG.results_history
    theta = LG.theta
    fresh_key = LG.fresh_key
    return LG, rw_sd, J, a, M, key, theta, results_history, fresh_key


@pytest.fixture(scope="function")
def simple(simple_setup):
    # Reset results history and theta to prevent carryover from other tests.
    LG, J, sigmas, a, M, key, theta, fresh_key = simple_setup
    LG.results_history.clear()
    LG.theta = theta
    LG.fresh_key = fresh_key
    return LG, J, sigmas, a, M, key


@pytest.fixture(scope="function")
def neapolitan(neapolitan_setup):
    # Reset results history and theta to prevent carryover from other tests.
    LG, rw_sd, J, a, M, key, theta, results_history, fresh_key = neapolitan_setup
    LG.results_history = results_history
    LG.theta = theta
    LG.fresh_key = fresh_key
    return LG, rw_sd, J, a, M, key


def test_invalid_initialization(simple):
    LG, *_ = simple
    for arg in ["ys", "theta", "rinit", "rproc", "dmeas"]:
        with pytest.raises(Exception):
            kwargs = {
                "ys": LG.ys,
                "theta": LG.theta,
                "rinit": LG.rinit,
                "rproc": LG.rproc,
                "dmeas": LG.dmeas,
            }
            kwargs[arg] = None
            pp.Pomp(**kwargs)


def test_results(neapolitan):
    LG, rw_sd, J, a, M, key = neapolitan
    # Check that results() returns one row per parameter set and correct columns
    # pfilter: should be one row per parameter set (len(theta))
    n_paramsets = len(LG.theta)
    res_pfilter = LG.results(0)
    assert res_pfilter.shape[0] == n_paramsets  # one row per parameter set
    expected_cols = {"logLik", "se", *LG.theta[0].keys()}
    assert set(res_pfilter.columns) == expected_cols

    # mif: should be one row per parameter set (len(theta))
    res_mif = LG.results(1)
    n_paramsets = len(LG.theta)
    assert res_mif.shape[0] == n_paramsets  # one row per parameter set
    assert set(res_mif.columns) == expected_cols

    # train: should be one row per parameter set (len(theta))
    res_train = LG.results(2)
    n_paramsets = len(LG.theta)
    assert res_train.shape[0] == n_paramsets  # one row per parameter set
    assert set(res_train.columns) == expected_cols


def test_sample_params():
    param_bounds = {
        "R0": (0, 100),
        "sigma": (0, 100),
        "gamma": (0, 100),
    }
    n = 10
    key = jax.random.key(1)
    param_sets = pp.Pomp.sample_params(param_bounds, n, key)
    assert len(param_sets) == n
    for params in param_sets:
        param_names = list(params.keys())
        assert param_names == list(param_bounds.keys())
        for param_name, value in params.items():
            assert isinstance(value, float)


def test_theta_carryover_mif(simple):
    LG, rw_sd, J, a, M, key = simple
    # Check that theta estimate from mif is correctly carried over to attribute and traces
    theta_order = list(LG.theta[0].keys())
    LG.mif(
        J=J,
        rw_sd=rw_sd,
        M=M,
        a=a,
        key=key,
    )
    assert theta_order == list(LG.theta[0].keys())
    LG.pfilter(J=J, reps=2)
    assert list(LG.results_history[-1]["theta"][0].keys()) == theta_order
    traces_da = LG.results_history[-2]["traces"]
    param_names = traces_da.coords["variable"].values[1:]
    last_row = traces_da.sel(replicate=0, iteration=traces_da.sizes["iteration"] - 1)
    last_param_values = [
        float(last_row.sel(variable=param).values) for param in param_names
    ]
    assert list(LG.results_history[-1]["theta"][0].values()) == last_param_values
    traces = LG.traces()
    # Only compare the parameter values
    assert traces.iloc[-1, 4:].values.tolist() == traces.iloc[-2, 4:].values.tolist()


# TODO: merge mif and train tests
def test_theta_carryover_train(simple):
    LG, rw_sd, J, a, M, key = simple
    # Check that theta estimate from train is correctly carried over to attribute and traces
    theta_order = list(LG.theta[0].keys())
    LG.train(
        J=J,
        M=1,
        eta=0.2,
        key=key,
    )
    assert theta_order == list(LG.theta[0].keys())
    LG.pfilter(J=J, reps=2)
    assert list(LG.results_history[-1]["theta"][0].keys()) == theta_order
    traces_da = LG.results_history[-2]["traces"]
    param_names = traces_da.coords["variable"].values[1:]
    last_row = traces_da.sel(replicate=0, iteration=traces_da.sizes["iteration"] - 1)
    last_param_values = [
        float(last_row.sel(variable=param).values) for param in param_names
    ]
    assert list(LG.results_history[-1]["theta"][0].values()) == last_param_values
    traces = LG.traces()
    # Only compare the parameter values
    assert traces.iloc[-1, 4:].values.tolist() == traces.iloc[-2, 4:].values.tolist()


def test_pickle(simple):
    LG, rw_sd, J, a, M, key = simple
    # Generate results to pickle
    LG.pfilter(J=J, reps=1, key=key)
    # Pickle the object
    pickled_data = pickle.dumps(LG)

    # Unpickle the object
    unpickled_obj = pickle.loads(pickled_data)

    # Check that the unpickled object has the same attributes
    assert LG.ys.values.tolist() == unpickled_obj.ys.values.tolist()
    assert LG.theta == unpickled_obj.theta
    assert LG.covars == unpickled_obj.covars
    assert LG.rinit == unpickled_obj.rinit
    assert LG.rproc == unpickled_obj.rproc
    assert LG.dmeas == unpickled_obj.dmeas
    assert LG.rproc.dt == unpickled_obj.rproc.dt
    assert LG.results_history == unpickled_obj.results_history
    assert LG.traces().values.tolist() == unpickled_obj.traces().values.tolist()

    # Check that the unpickled object can be pickled again if rmeas is None
    unpickled_obj.rmeas = None
    pickled_data = pickle.dumps(unpickled_obj)

    # Check that the unpickled object can still be used for filtering
    unpickled_obj.pfilter(J=J, reps=1)


def test_prune(simple):
    LG, rw_sd, J, a, M, key = simple
    # Run pfilter with multiple replicates to generate results
    LG.pfilter(J=J, reps=5, key=key)
    # Save the original theta list length
    orig_theta = LG.theta.copy()
    orig_len = len(orig_theta)
    # Prune to top 2 thetas, refill to original length
    LG.prune(n=2, refill=True)
    assert isinstance(LG.theta, list)
    assert len(LG.theta) == orig_len
    # The unique thetas should be at most 2
    unique_thetas = [tuple(sorted(d.items())) for d in LG.theta]
    assert len(set(unique_thetas)) <= 2
    # Prune to top 1 theta, do not refill
    LG.prune(n=1, refill=False)
    assert isinstance(LG.theta, list)
    assert len(LG.theta) == 1
    # The theta should be a dict
    assert isinstance(LG.theta[0], dict)
    # Prune with n greater than available thetas (should not error, just return all)
    LG.theta = orig_theta.copy()
    LG.prune(n=10, refill=False)
    assert len(LG.theta) == min(10, len(orig_theta))
    # Test error if results are empty
    LG2 = LG.__class__(
        ys=LG.ys.copy(),
        theta=LG.theta[0].copy(),
        rinit=LG.rinit.original_func,
        rproc=LG.rproc.original_func,
        dmeas=LG.dmeas.original_func,
        t0=LG.t0,
        nstep=LG.rproc.nstep,
        ydim=LG.rmeas.ydim,
        statenames=["state_0", "state_1"],
    )
    with pytest.raises(IndexError):
        LG2.prune(n=1)


def test_time(neapolitan):
    LG, *_ = neapolitan
    time_df = LG.time()
    assert len(time_df) == 3
    assert time_df["method"].tolist() == ["pfilter", "mif", "train"]
    assert isinstance(time_df["time"].tolist(), list)


def test_print_summary(neapolitan):
    # Should not error
    LG, *_ = neapolitan
    LG.print_summary()
