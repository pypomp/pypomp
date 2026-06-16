import jax
import jax.numpy as jnp
import pypomp as pp
import pytest


@pytest.fixture(scope="function")
def simple():
    dacca = pp.models.dacca()
    J = 2
    key = jax.random.key(111)
    ys = dacca.ys
    rw_sd = pp.RWSigma(
        sigmas={
            "gamma": 0.02,
            "m": 0.02,
            "rho": 0.02,
            "epsilon": 0.02,
            "c": 0.02,
            "beta_trend": 0.02,
            "sigma": 0.02,
            "tau": 0.02,
            "alpha": 0.02,
            "delta": 0.02,
            "S_0": 0.02,
            "I_0": 0.02,
            "Y_0": 0.02,
            "R1_0": 0.02,
            "R2_0": 0.02,
            "R3_0": 0.02,
            **{f"bs{i}": 0.02 for i in range(1, 7)},
            **{f"omegas{i}": 0.02 for i in range(1, 7)},
        },
        init_names=[],
    )
    return dacca, rw_sd, J, key, ys


def test_dacca_pfilter(simple):
    dacca, rw_sd, J, key, ys = simple
    dacca.pfilter(J=1000, reps=6, key=key)
    logLik = pp.maths.logmeanexp(dacca.results_history[-1].logLiks)
    # Threshold used to be 2.0, but the logLik increased somewhat after updating the
    # model to round the starting state values, like the R pomp model. Kind of surprised
    # that made a meaningful difference.
    # -3748.6 comes from what I understand the R pomp MLE to be.
    assert abs(logLik - -3748.6) < 2.5


def test_dacca_basic(simple):
    # Check whether dacca.mif() runs without error.
    dacca, rw_sd, J, key, ys = simple
    dacca.mif(rw_sd=rw_sd.geometric_cooling(a=0.5), J=J, key=key, M=1)


def test_dacca_nstep():
    # Check that dacca.train() runs without error when nstep is specified.
    dacca_nstep = pp.models.dacca(nstep=10, dt=None)
    eta = pp.LearningRate({param: 0.2 for param in dacca_nstep.canonical_param_names})
    dacca_nstep.train(J=2, M=1, eta=eta, key=jax.random.key(111))


def test_dacca_dt():
    # Check that dacca.train() runs without error when dt is specified and nstep
    # happens to be the same for every observation interval.
    dacca_dt = pp.models.dacca(nstep=None, dt=1 / 240)
    eta = pp.LearningRate({param: 0.2 for param in dacca_dt.canonical_param_names})
    dacca_dt.train(J=2, M=1, eta=eta, key=jax.random.key(111))


def test_dhaka_alias():
    # Check that dhaka is an alias for dacca
    assert pp.models.dhaka is pp.models.dacca


def test_dacca_gamma_noise():
    """Verify that using gamma process noise works and runs basic pfilter."""
    dacca_gamma = pp.models.dacca(gamma=True)
    assert dacca_gamma.rproc.original_func.__name__ == "_rproc_gamma"
    dacca_gamma.pfilter(J=5, key=jax.random.key(1))
    logLiks = dacca_gamma.results_history[-1].logLiks  # type: ignore
    assert jnp.isfinite(logLiks.data).all()


def test_dacca_parameter_transformations(simple):
    """Verify that parameter transformations round-trip correctly."""
    dacca, _, _, _, _ = simple
    theta = dacca.theta[0]

    est_theta = dacca.par_trans.to_est(theta)
    recovered_theta = dacca.par_trans.from_est(est_theta)

    assert set(theta.keys()) == set(recovered_theta.keys())

    # IVPs are normalized on the simplex during conversion to estimation space
    IVP_list = ["S_0", "I_0", "Y_0", "R1_0", "R2_0", "R3_0"]
    ivp_sum = sum(theta[k] for k in IVP_list)

    for k in theta.keys():
        if k in IVP_list:
            expected = theta[k] / ivp_sum
        else:
            expected = theta[k]
        assert jnp.allclose(expected, recovered_theta[k], atol=1e-5, rtol=1e-5), (
            f"Mismatch for parameter {k}: expected {expected} vs recovered {recovered_theta[k]}"
        )


def test_dacca_invalid_init():
    """Verify initialization exception for specifying both dt and nstep."""
    with pytest.raises(ValueError, match="Cannot specify both dt and nstep"):
        pp.models.dacca(dt=0.1, nstep=10)


def test_dacca_dmeas_edge_cases(simple):
    """Test all branches of _dmeas likelihood logic."""
    dacca, _, _, _, _ = simple
    statenames = dacca.statenames
    param_names = dacca.canonical_param_names
    covar_names = dacca.covar_names

    Y_arr = jnp.array([10.0])

    X_dict = {
        "S": 1000.0,
        "I": 10.0,
        "Y": 5.0,
        "Mn": 10.0,
        "R1": 1.0,
        "R2": 1.0,
        "R3": 1.0,
        "count": 0.0,
    }
    X_arr = jnp.array([X_dict[name] for name in statenames])

    theta_dict = dacca.theta[0]
    theta_arr = jnp.array([theta_dict[name] for name in param_names])

    covars_dict = {name: 1.0 for name in covar_names}
    covar_arr = jnp.array([covars_dict[name] for name in covar_names])

    t = 1900.0

    # 1. Normal call (happy path)
    loglik = dacca.dmeas.struct(Y_arr, X_arr, theta_arr, covar_arr, t, False)
    assert jnp.isfinite(loglik)

    # 2. State violation: count > 0
    X_dict_invalid = X_dict.copy()
    X_dict_invalid["count"] = 1.0
    X_arr_invalid = jnp.array([X_dict_invalid[name] for name in statenames])
    loglik_invalid = dacca.dmeas.struct(
        Y_arr, X_arr_invalid, theta_arr, covar_arr, t, False
    )
    assert jnp.allclose(loglik_invalid, jnp.log(1e-18))

    # 3. Variance violation (non-finite scale)
    theta_dict_bad = dict(theta_dict)
    theta_dict_bad["tau"] = jnp.nan
    theta_arr_bad = jnp.array([theta_dict_bad[name] for name in param_names])
    loglik_bad_var = dacca.dmeas.struct(
        Y_arr, X_arr, theta_arr_bad, covar_arr, t, False
    )
    assert jnp.allclose(loglik_bad_var, jnp.log(1e-18))


def test_dacca_rmeas(simple):
    """Verify that _rmeas simulates valid observations."""
    dacca, _, _, key, _ = simple
    statenames = dacca.statenames
    param_names = dacca.canonical_param_names
    covar_names = dacca.covar_names

    X_dict = {
        "S": 1000.0,
        "I": 10.0,
        "Y": 5.0,
        "Mn": 10.0,
        "R1": 1.0,
        "R2": 1.0,
        "R3": 1.0,
        "count": 0.0,
    }
    X_arr = jnp.array([X_dict[name] for name in statenames])

    theta_dict = dacca.theta[0]
    theta_arr = jnp.array([theta_dict[name] for name in param_names])

    covars_dict = {name: 1.0 for name in covar_names}
    covar_arr = jnp.array([covars_dict[name] for name in covar_names])

    t = 1900.0

    sim_obs = dacca.rmeas.struct(X_arr, theta_arr, key, covar_arr, t, False)
    assert jnp.isfinite(sim_obs).all()
