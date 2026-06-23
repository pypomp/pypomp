import jax
import jax.numpy as jnp
import pandas as pd
import pytest
import pypomp as pp
import pypomp.functional as F


@pytest.fixture(scope="function")
def model_setup():
    model = pp.models.LG()
    struct = model.to_struct()
    key = jax.random.key(1)
    J = 5
    n_reps = 2
    param_names = model.canonical_param_names
    # Shape (n_reps, n_params)
    thetas_array = jnp.repeat(model.theta.to_jax_array(param_names), n_reps, axis=0)
    return struct, thetas_array, key, J, n_reps, param_names


def test_pfilter_functional(model_setup):
    struct, thetas_array, key, J, n_reps, _ = model_setup
    reps = 2
    rep_keys = jax.random.split(key, n_reps * reps).reshape(n_reps, reps)

    results = F.pfilter(struct, thetas_array, J, thresh=0.0, keys=rep_keys)

    assert "logLik" in results
    assert results["logLik"].shape == (n_reps, reps)
    assert jnp.all(jnp.isfinite(results["logLik"]))


def test_mop_functional(model_setup):
    struct, thetas_array, key, J, n_reps, _ = model_setup
    keys = jax.random.split(key, n_reps)

    results = F.mop(struct, thetas_array, J, alpha=0.5, keys=keys)

    assert results.shape == (n_reps,)
    assert jnp.all(jnp.isfinite(results))


def test_dpop_functional(model_setup):
    struct, thetas_array, key, J, n_reps, _ = model_setup
    keys = jax.random.split(key, n_reps)

    results = F.dpop(
        struct, thetas_array, J, alpha=0.5, process_weight_index=0, keys=keys
    )

    assert results.shape == (n_reps,)
    assert jnp.all(jnp.isfinite(results))


def test_train_functional(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    eta = jnp.ones(len(param_names)) * 0.01

    neg_logliks, theta_traces = F.train(
        struct,
        thetas_array,
        J,
        optimizer=pp.Adam(scale=False, ls=False, c=0.0, max_ls_itn=1),
        M=M,
        eta=eta,
        thresh=0.0,
        alpha=0.0,
        keys=keys,
        alpha_cooling=1.0,
        n_monitors=1,
    )

    assert neg_logliks.shape == (n_reps, M + 1)
    assert theta_traces.shape == (n_reps, M + 1, len(param_names))
    # Skip the first element which is NaN by design
    assert jnp.all(jnp.isfinite(neg_logliks[:, 1:]))


def test_mif_functional(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    sigmas = jnp.ones(len(param_names)) * 0.02

    # thetas_array for mif needs to be (n_reps, J, n_params)
    thetas_mif = jnp.repeat(thetas_array[:, jnp.newaxis, :], J, axis=1)

    a_val = 0.5
    factor = a_val ** (1 / 50)

    def cooling_fn(nt, m, ntimes):
        return factor ** (nt / ntimes + m - 1)

    neg_logliks_M, thetas_traces_Md, final_theta_Jd = F.mif(
        struct,
        thetas_mif,
        sigmas,
        sigmas,
        M=M,
        cooling_fn=cooling_fn,
        J=J,
        thresh=0.0,
        keys=keys,
        n_monitors=0,
    )

    assert neg_logliks_M.shape == (n_reps, M)
    assert thetas_traces_Md.shape == (n_reps, M + 1, len(param_names))
    assert final_theta_Jd.shape == (n_reps, J, len(param_names))


def test_simulate_functional(model_setup):
    struct, thetas_array, key, J, n_reps, _ = model_setup
    nsim = 3
    keys = jax.random.split(key, n_reps)

    X_sims, Y_sims = F.simulate(struct, thetas_array, nsim, keys)

    n_times = len(struct.times)
    n_states = X_sims.shape[-1]
    n_obs = Y_sims.shape[-1]

    # X_sims has n_times + 1 points (including t0)
    assert X_sims.shape == (n_reps, nsim, n_times + 1, n_states)
    assert Y_sims.shape == (n_reps, nsim, n_times, n_obs)


@pytest.fixture(scope="function")
def panel_setup():
    LG1 = pp.models.LG()
    LG2 = pp.models.LG()

    shared_param_names = ["A11", "A12", "A21", "A22", "C11", "C12", "C21", "C22"]
    unit_param_names = ["Q11", "Q12", "Q22", "R11", "R12", "R22"]

    # Simple ParTrans (identity)
    LG1.par_trans = pp.ParTrans()
    LG2.par_trans = pp.ParTrans()

    theta_base = LG1.theta.params()[0]

    shared_params = pd.DataFrame(
        index=pd.Index(shared_param_names),
        data={"shared": [theta_base[name] for name in shared_param_names]},
    )

    unit_specific_params = pd.DataFrame(
        index=pd.Index(unit_param_names),
        data={
            "unit1": [theta_base[name] * 0.8 for name in unit_param_names],
            "unit2": [theta_base[name] * 1.2 for name in unit_param_names],
        },
    )

    theta: list[dict[str, pd.DataFrame | None]] = [
        {"shared": shared_params, "unit_specific": unit_specific_params}
    ]

    panel = pp.PanelPomp(
        Pomp_dict={"unit1": LG1, "unit2": LG2},
        theta=pp.PanelParameters(theta),
    )

    struct = panel.to_struct()

    # For panel functional, parameters:
    # shared: shape (n_reps, n_shared)
    # unit: shape (n_reps, U, n_spec)
    n_reps = 2
    U = 2
    n_shared = len(shared_param_names)
    n_spec = len(unit_param_names)

    shared_array = jnp.repeat(
        jnp.array([theta_base[name] for name in shared_param_names])[None, :],
        n_reps,
        axis=0,
    )
    unit_array = jnp.stack(
        [
            jnp.repeat(
                jnp.array([theta_base[name] * 0.8 for name in unit_param_names])[
                    None, :
                ],
                n_reps,
                axis=0,
            ),
            jnp.repeat(
                jnp.array([theta_base[name] * 1.2 for name in unit_param_names])[
                    None, :
                ],
                n_reps,
                axis=0,
            ),
        ],
        axis=1,
    )  # shape: (n_reps, U, n_spec)

    key = jax.random.key(1)
    J = 3

    return struct, shared_array, unit_array, key, J, n_reps, U, n_shared, n_spec


def test_panel_mif_functional(panel_setup):
    struct, shared_array, unit_array, key, J, n_reps, U, n_shared, n_spec = panel_setup

    # mif takes particle swarm (n_reps, J, n_shared) and (n_reps, J, U, n_spec)
    shared_mif = jnp.repeat(shared_array[:, jnp.newaxis, :], J, axis=1)
    unit_mif = jnp.repeat(unit_array[:, jnp.newaxis, :, :], J, axis=1)

    all_param_names = struct.shared_param_names + struct.unit_param_names
    sigmas = jnp.ones(len(all_param_names)) * 0.02

    M = 2
    a_val = 0.5
    factor = a_val ** (1 / 50)

    def cooling_fn(nt, m, ntimes):
        return factor ** (nt / ntimes + m - 1)

    keys = jax.random.split(key, n_reps)

    shared_traces, unit_traces, final_shared_swarm, final_unit_swarm = F.panel_mif(
        struct,
        shared_mif,
        unit_mif,
        sigmas,
        sigmas,
        M=M,
        cooling_fn=cooling_fn,
        J=J,
        thresh=0.0,
        keys=keys,
        n_monitors=0,
    )

    assert shared_traces.shape == (n_reps, M + 1, n_shared + 1)
    assert unit_traces.shape == (n_reps, M + 1, U, n_spec + 1)
    assert final_shared_swarm.shape == (n_reps, J, n_shared)
    assert final_unit_swarm.shape == (n_reps, J, U, n_spec)


def test_panel_train_functional(panel_setup):
    struct, shared_array, unit_array, key, J, n_reps, U, n_shared, n_spec = panel_setup

    # train takes parameters without J dimension: (n_reps, n_shared) and (n_reps, U, n_spec)
    M = 2
    eta_shared = jnp.ones((M, n_shared)) * 0.01
    eta_spec = jnp.ones((M, n_spec)) * 0.01

    keys = jax.random.split(key, n_reps * M * U).reshape(n_reps, M, U)

    neg_logliks, shared_history, unit_history = F.panel_train(
        struct,
        shared_array,
        unit_array,
        J=J,
        optimizer=pp.Adam(),
        M=M,
        eta_shared=eta_shared,
        eta_spec=eta_spec,
        alpha=0.97,
        keys=keys,
        alpha_cooling=1.0,
        chunk_size=1,
    )

    assert neg_logliks.shape == (n_reps, M + 1)
    assert shared_history.shape == (n_reps, M + 1, n_shared)
    assert unit_history.shape == (n_reps, M + 1, U, n_spec)


def test_align_params():
    # 1. Test scalar float parameters
    params_scalar = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}
    names_scalar = ["gamma", "alpha", "beta"]
    aligned_scalar = F.align_params(params_scalar, names_scalar)
    expected_scalar = jnp.array([3.0, 1.0, 2.0])
    assert jnp.array_equal(aligned_scalar, expected_scalar)

    # 2. Test dynamic arrays stacked along the last axis
    params_arrays = {
        "alpha": jnp.ones((2, 5)) * 1.5,
        "beta": jnp.ones((2, 5)) * 2.5,
    }
    names_arrays = ["beta", "alpha"]
    aligned_arrays = F.align_params(params_arrays, names_arrays, axis=-1)
    assert aligned_arrays.shape == (2, 5, 2)
    assert jnp.all(aligned_arrays[..., 0] == 2.5)
    assert jnp.all(aligned_arrays[..., 1] == 1.5)

    # 3. Test dynamic arrays stacked along axis=0
    aligned_arrays_axis0 = F.align_params(params_arrays, names_arrays, axis=0)
    assert aligned_arrays_axis0.shape == (2, 2, 5)
    assert jnp.all(aligned_arrays_axis0[0, ...] == 2.5)
    assert jnp.all(aligned_arrays_axis0[1, ...] == 1.5)

    # 4. Test KeyError handling for missing parameter
    params_missing = {"alpha": 1.0}
    names_missing = ["alpha", "beta"]
    with pytest.raises(KeyError) as exc_info:
        F.align_params(params_missing, names_missing)
    assert "Parameter 'beta' is required by the model structure" in str(exc_info.value)


def test_panel_pfilter_functional(panel_setup):
    struct, shared_array, unit_array, key, J, n_reps, U, n_shared, n_spec = panel_setup

    # Construct thetas_array of shape (n_reps, U, n_params)
    thetas_panel = jnp.stack(
        [
            jnp.concatenate([shared_array, unit_array[:, u, :]], axis=-1)
            for u in range(U)
        ],
        axis=1,
    )

    keys = jax.random.split(key, n_reps * U).reshape(n_reps, U, *key.shape)

    results = F.panel_pfilter(
        struct,
        thetas_panel,
        J=J,
        thresh=0.0,
        keys=keys,
        chunk_size=1,
    )

    assert "logLik" in results
    assert results["logLik"].shape == (n_reps, U)
    assert jnp.all(jnp.isfinite(results["logLik"]))
