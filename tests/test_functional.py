import jax
import jax.numpy as jnp
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
        optimizer="Adam",
        M=M,
        eta=eta,
        c=0.0,
        max_ls_itn=1,
        thresh=0.0,
        scale=False,
        ls=False,
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

    # thetas_array for mif needs to be (J, n_reps, n_params)
    thetas_mif = jnp.repeat(thetas_array[jnp.newaxis, ...], J, axis=0)

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
