import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.functional import abc, pmcmc
from tests.helpers.models import lg_panel, sir_panel
from tests.helpers.params import uniform_rw_sd


@pytest.fixture(scope="function")
def model_setup():
    model = pp.models.lg()
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


@pytest.fixture(scope="module")
def sir_dpop_setup():
    """(struct, natural thetas, est thetas, keys, J)."""
    model = pp.models.sir(times=np.array([0.2, 0.4]), key=jax.random.key(0))
    struct = model.to_struct()
    names = model.canonical_param_names
    n_reps = 2
    thetas = jnp.repeat(model.theta.to_jax_array(names), n_reps, axis=0)
    thetas_est = struct.par_trans._transform_array(thetas, names, direction="to_est")
    keys = jax.random.split(jax.random.key(1), n_reps)
    return struct, thetas, thetas_est, keys, 5


def test_dpop_functional(sir_dpop_setup):
    struct, _, thetas_est, keys, J = sir_dpop_setup

    results = F.pop(struct, thetas_est, J, alpha=0.5, keys=keys)

    assert results.shape == (keys.shape[0],)
    assert jnp.all(jnp.isfinite(results))


def test_dpop_value_matches_mop(sir_dpop_setup):
    """The process score has value zero, so DPOP's estimate equals MOP's for
    the same keys; only the gradient differs."""
    struct, _, thetas_est, keys, J = sir_dpop_setup

    dpop_nll = F.pop(struct, thetas_est, J, alpha=0.9, keys=keys)
    mop_nll = F.mop(struct, thetas_est, J, alpha=0.9, keys=keys)

    np.testing.assert_array_equal(np.asarray(dpop_nll), np.asarray(mop_nll))


def test_pop_requires_logw_state(model_setup):
    struct, thetas_array, key, J, n_reps, _ = model_setup
    keys = jax.random.split(key, n_reps)

    with pytest.raises(ValueError, match="state named '_logw'"):
        F.pop(struct, thetas_array, J, 0.5, keys)


def test_pop_resets_logw_without_accumvars(sir_dpop_setup):
    """_logw is reset at every observation time even when it is not listed in
    accumvars; otherwise past scores would carry forward into the gradient."""
    struct, _, thetas_est, keys, J = sir_dpop_setup
    logw_index = struct.statenames.index("_logw")
    unlisted = struct._replace(
        accumvars=tuple(i for i in struct.accumvars or () if i != logw_index)
    )

    def grad(s):
        return jax.grad(lambda th: F.pop(s, th, J, 0.5, keys).sum())(thetas_est)

    np.testing.assert_array_equal(np.asarray(grad(unlisted)), np.asarray(grad(struct)))


def _zero_weight_particles_model(T: int) -> pp.Pomp:
    """About half the particles have zero measurement weight at every step, and
    the DPOP log-weight "_logw" stays identically zero."""

    def rinit(theta_, key, covars, t0):
        return {"X": jax.random.normal(key), "_logw": 0.0}

    def rproc(X_, theta_, key, covars, t, dt):
        # A fresh draw each interval, so about half the particles are dead at every step.
        return {"X": theta_["sigma"] * jax.random.normal(key), "_logw": X_["_logw"]}

    def dmeas(Y_, X_, theta_, covars, t):
        ll = jax.scipy.stats.norm.logpdf(Y_["y"], X_["X"], 1.0)
        return jnp.where(X_["X"] > 0, ll, -jnp.inf)

    def rmeas(X_, theta_, key, covars, t):
        return {"y": X_["X"] + jax.random.normal(key)}

    ys = pd.DataFrame(
        {"y": np.full(T, 0.5)}, index=pd.Index(np.arange(1.0, T + 1), name="time")
    )
    return pp.Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys,
        theta=pp.PompParameters({"sigma": 1.0}),
        statenames=["X", "_logw"],
        t0=0.0,
        nstep=1,
        accumvars=("_logw",),
        covars=None,
    )


def test_mop_dpop_finite_with_many_zero_weight_particles():
    """MOP and DPOP stay finite when many particles have zero measurement weight.

    Both rebuild their carried weights as ``(w + m - stop_gradient(m))[counts]``,
    which is NaN if resampling ever selects a particle with ``m = -inf``.
    """
    model = _zero_weight_particles_model(T=100)
    struct = model.to_struct()
    n_reps, J = 8, 10000
    thetas_array = jnp.repeat(
        model.theta.to_jax_array(model.canonical_param_names), n_reps, axis=0
    )
    keys = jax.random.split(jax.random.key(0), n_reps)

    def mop_objective(th):
        return F.mop(struct, th, J, alpha=0.97, keys=keys)

    assert jnp.all(jnp.isfinite(mop_objective(thetas_array)))
    assert jnp.all(
        jnp.isfinite(jax.grad(lambda th: mop_objective(th).sum())(thetas_array))
    )
    dpop_nll = F.pop(struct, thetas_array, J, alpha=0.97, keys=keys)
    assert jnp.all(jnp.isfinite(dpop_nll))


def test_dpop_grad_matches_mop_when_process_weight_is_zero():
    """With an identically zero process log-weight, the DPOP score term
    vanishes and its gradient equals MOP's."""
    model = _zero_weight_particles_model(T=5)
    struct = model.to_struct()
    thetas_array = model.theta.to_jax_array(model.canonical_param_names)
    keys = jax.random.split(jax.random.key(0), 1)

    grad_mop = jax.grad(lambda th: F.mop(struct, th, 50, 0.9, keys).sum())(thetas_array)
    grad_dpop = jax.grad(lambda th: F.pop(struct, th, 50, 0.9, keys).sum())(
        thetas_array
    )

    # Equal up to float32 rounding from the extra (zero) term.
    np.testing.assert_allclose(np.asarray(grad_dpop), np.asarray(grad_mop), rtol=1e-6)


def test_dpop_train_functional(sir_dpop_setup):
    struct, thetas, _, keys, J = sir_dpop_setup
    n_reps, n_params = thetas.shape
    M = 2
    eta = pp.LearningRate({name: 0.01 for name in struct.param_names})

    neg_logliks, theta_traces = F.train(
        struct,
        thetas,
        J,
        M,
        eta,
        keys,
        optimizer=pp.Adam(),
        alpha=0.8,
        dpop=True,
    )

    assert neg_logliks.shape == (n_reps, M + 1)
    assert theta_traces.shape == (n_reps, M + 1, n_params)
    assert jnp.all(jnp.isfinite(neg_logliks))
    assert jnp.all(jnp.isfinite(theta_traces))


def test_dpop_train_functional_natural_scale(sir_dpop_setup):
    """With DPOP, train still takes and returns natural-scale parameters: with
    zero learning rates the trace reproduces the input."""
    struct, thetas, _, keys, J = sir_dpop_setup
    eta = pp.LearningRate({name: 0.0 for name in struct.param_names})

    neg_logliks, theta_traces = F.train(struct, thetas, J, 2, eta, keys, dpop=True)

    assert jnp.all(jnp.isfinite(neg_logliks))
    np.testing.assert_allclose(
        np.asarray(theta_traces[:, -1, :]), np.asarray(thetas), rtol=1e-5
    )


def test_train_functional(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    eta = pp.LearningRate({name: 0.01 for name in param_names})

    neg_logliks, theta_traces = F.train(
        struct,
        thetas_array,
        J,
        M,
        eta,
        keys,
        optimizer=pp.Adam(scale=False, ls=False, c=0.0, max_ls_itn=1),
        thresh=0.0,
        alpha=0.0,
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
    rw_sd = uniform_rw_sd(param_names, cooling=0.5)

    # thetas_array for mif needs to be (n_reps, J, n_params)
    thetas_mif = jnp.repeat(thetas_array[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, final_theta_Jd = F.mif(
        struct,
        thetas_mif,
        J,
        M,
        rw_sd,
        keys,
        thresh=0.0,
        n_monitors=0,
    )

    assert logliks_M.shape == (n_reps, M)
    assert thetas_traces_Md.shape == (n_reps, M + 1, len(param_names))
    assert final_theta_Jd.shape == (n_reps, J, len(param_names))


def test_simulate_functional(model_setup):
    struct, thetas_array, key, J, n_reps, _ = model_setup
    nsim = 3
    keys = jax.random.split(key, n_reps)

    X_sims, Y_sims = F.simulate(struct, nsim, thetas_array, keys=keys)

    n_times = len(struct.times)
    n_states = X_sims.shape[-1]
    n_obs = Y_sims.shape[-1]

    # X_sims has n_times + 1 points (including t0)
    assert X_sims.shape == (n_reps, nsim, n_times + 1, n_states)
    assert Y_sims.shape == (n_reps, nsim, n_times, n_obs)


@pytest.fixture(scope="function")
def panel_setup():
    shared_param_names = ["A11", "A12", "A21", "A22", "C11", "C12", "C21", "C22"]
    unit_param_names = ["Q11", "Q21", "Q22", "R11", "R21", "R22", "X0_1", "X0_2"]

    panel = lg_panel(
        sharing="some",
        shared_names=shared_param_names,
        unit_scales=[0.8, 1.2],
        par_trans=pp.ParTrans(),
    )
    theta_base = pp.models.lg().theta.params(as_list=True)[0]

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
    M = 2
    rw_sd = uniform_rw_sd(all_param_names, cooling=0.5)

    keys = jax.random.split(key, n_reps)

    shared_traces, unit_traces, final_shared_swarm, final_unit_swarm = F.panel_mif(
        struct,
        shared_mif,
        unit_mif,
        J,
        M,
        rw_sd,
        keys,
        thresh=0.0,
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
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    eta = pp.LearningRate({name: 0.01 for name in all_param_names})

    keys = jax.random.split(key, n_reps * (M + 1) * U).reshape(n_reps, M + 1, U)

    neg_logliks, shared_history, unit_history = F.panel_train(
        struct,
        shared_array,
        unit_array,
        J,
        M,
        eta,
        keys,
        optimizer=pp.Adam(),
        alpha=0.97,
        alpha_cooling=1.0,
        chunk_size=1,
    )

    assert neg_logliks.shape == (n_reps, M + 1)
    assert shared_history.shape == (n_reps, M + 1, n_shared)
    assert unit_history.shape == (n_reps, M + 1, U, n_spec)


def test_panel_train_functional_unsupported_optimizer(panel_setup):
    """Panel train only supports SGD/Adam/FullMatrixAdam (no Hessian-based
    optimizers, since panel train doesn't compute one); other optimizers raise."""
    struct, shared_array, unit_array, key, J, n_reps, U, _, _ = panel_setup
    M = 1
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    eta = pp.LearningRate({name: 0.01 for name in all_param_names})
    keys = jax.random.split(key, n_reps * (M + 1) * U).reshape(n_reps, M + 1, U)

    with pytest.raises(ValueError, match="not supported for panel train"):
        F.panel_train(
            struct,
            shared_array,
            unit_array,
            J,
            M,
            eta,
            keys,
            optimizer=pp.Newton(),
            alpha=0.97,
            alpha_cooling=1.0,
            chunk_size=1,
        )


def test_panel_train_functional_scale_and_clip(panel_setup):
    """Exercise the gradient-clipping and direction-rescaling branches of the
    per-chunk panel train step (clip_norm and scale=True)."""
    struct, shared_array, unit_array, key, J, n_reps, U, _, _ = panel_setup
    M = 1
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    eta = pp.LearningRate({name: 0.01 for name in all_param_names})
    keys = jax.random.split(key, n_reps * (M + 1) * U).reshape(n_reps, M + 1, U)

    neg_logliks, shared_history, unit_history = F.panel_train(
        struct,
        shared_array,
        unit_array,
        J,
        M,
        eta,
        keys,
        optimizer=pp.SGD(scale=True, clip_norm=1.0),
        alpha=0.97,
        alpha_cooling=1.0,
        chunk_size=1,
    )

    assert neg_logliks.shape == (n_reps, M + 1)
    assert jnp.all(jnp.isfinite(neg_logliks))
    assert jnp.all(jnp.isfinite(shared_history))
    assert jnp.all(jnp.isfinite(unit_history))


def test_panel_train_functional_requires_final_key_slab(panel_setup):
    """Keys without the final-evaluation slab are rejected rather than clamped."""
    struct, shared_array, unit_array, key, J, n_reps, U, _, _ = panel_setup
    M = 2
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    eta = pp.LearningRate({name: 0.01 for name in all_param_names})
    keys = jax.random.split(key, n_reps * M * U).reshape(n_reps, M, U)

    with pytest.raises(ValueError, match="M \\+ 1"):
        F.panel_train(struct, shared_array, unit_array, J, M, eta, keys)


def test_panel_dpop_train_functional():
    panel = sir_panel(sharing="some", times=np.array([0.2, 0.4]))
    struct = panel.to_struct()
    theta = panel.theta
    unit_names = panel.get_unit_names()
    U = len(unit_names)
    shared = theta.to_jax_array(struct.shared_param_names, unit_names=unit_names)[
        :, 0, :
    ]
    unit = theta.to_jax_array(struct.unit_param_names, unit_names=unit_names)
    M = 2
    keys = jax.random.split(jax.random.key(0), (M + 1) * U).reshape(1, M + 1, U)
    eta = pp.LearningRate({name: 0.001 for name in struct.param_names})

    neg_logliks, shared_history, unit_history = F.panel_train(
        struct, shared, unit, 2, M, eta, keys, dpop=True
    )

    assert neg_logliks.shape == (1, M + 1)
    assert jnp.all(jnp.isfinite(neg_logliks))
    assert shared_history.shape == (1, M + 1, shared.shape[-1])
    assert unit_history.shape == (1, M + 1, U, unit.shape[-1])


def test_chunked_panel_mop_internal_direct(panel_setup):
    """``_chunked_panel_mop_internal``/``_vg_chunked_panel_mop_internal`` in
    pypomp.core.algorithms.mop compute the total panel negative log-likelihood
    (and its gradient) in one chunked pass, rather than via the per-chunk
    optimizer-step scan that pypomp.core.algorithms.train.py uses for actual
    training. Panel training uses the former to evaluate its final
    parameters; both are exercised directly here.
    """
    struct, shared_array, unit_array, key, J, n_reps, U, n_shared, n_spec = panel_setup
    from pypomp.core.algorithms.contexts import PanelTrainContext
    from pypomp.core.algorithms.mop import (
        _chunked_panel_mop_internal,
        _vg_chunked_panel_mop_internal,
    )

    M = 1
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    eta = pp.LearningRate({name: 0.01 for name in all_param_names})
    eta_shared = eta.to_array(struct.shared_param_names, M)
    eta_spec = eta.to_array(struct.unit_param_names, M)
    alpha = 0.97

    # Only the series/alpha/fns/J fields survive to_mop_context(), so the
    # placeholder shape of `keys` here doesn't need to match training usage.
    placeholder_keys = jax.random.split(key, n_reps * (M + 1) * U).reshape(
        n_reps, M + 1, U
    )
    panel_context = PanelTrainContext.from_panel_train_struct(
        struct, J, 1, M, 1.0, placeholder_keys, eta_shared, eta_spec, alpha
    )
    mop_context = panel_context.to_mop_context()

    shared_est, unit_est = struct.par_trans._transform_panel_array(
        shared_array,
        unit_array,
        struct.shared_param_names,
        struct.unit_param_names,
        direction="to_est",
    )
    rep_shared = shared_est[0]
    rep_unit = unit_est[0]
    unit_keys = jax.random.split(key, U)

    loss_chunk1 = _chunked_panel_mop_internal(
        rep_shared, rep_unit, struct.unit_param_permutations, mop_context, unit_keys, 1
    )
    loss_chunkU = _chunked_panel_mop_internal(
        rep_shared, rep_unit, struct.unit_param_permutations, mop_context, unit_keys, U
    )
    assert jnp.isfinite(loss_chunk1)
    # Chunking is purely a memory/vectorization strategy: the aggregated loss
    # must be the same regardless of how many chunks it's split across.
    assert jnp.allclose(loss_chunk1, loss_chunkU, rtol=1e-4)

    val, (grad_shared, grad_unit) = _vg_chunked_panel_mop_internal(
        rep_shared, rep_unit, struct.unit_param_permutations, mop_context, unit_keys, 1
    )
    assert jnp.isfinite(val)
    assert grad_shared.shape == rep_shared.shape
    assert grad_unit.shape == rep_unit.shape
    assert jnp.all(jnp.isfinite(grad_shared))
    assert jnp.all(jnp.isfinite(grad_unit))


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


def test_pmcmc_functional(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})

    ll_traces, lp_traces, theta_traces, accepts = pmcmc(
        struct,
        thetas_array,
        proposal=prop,
        J=J,
        M=M,
        thresh=0.0,
        keys=keys,
    )

    assert ll_traces.shape == (n_reps, M + 1)
    assert lp_traces.shape == (n_reps, M + 1)
    assert theta_traces.shape == (n_reps, M + 1, len(param_names))
    assert accepts.shape == (n_reps,)
    assert jnp.all(jnp.isfinite(theta_traces))


def test_abc_functional(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})

    probes = {
        "mean": lambda y: jnp.mean(y["Y1"]),
        "std": lambda y: jnp.std(y["Y1"]),
    }
    scale = {"mean": 10.0, "std": 10.0}

    dist_traces, lp_traces, theta_traces, accepts = abc(
        struct,
        thetas_array,
        proposal=prop,
        probes=probes,
        scale=scale,
        epsilon=1e6,
        M=M,
        keys=keys,
    )

    assert dist_traces.shape == (n_reps, M + 1)
    assert lp_traces.shape == (n_reps, M + 1)
    assert theta_traces.shape == (n_reps, M + 1, len(param_names))
    assert accepts.shape == (n_reps,)
    assert jnp.all(jnp.isfinite(theta_traces))


def test_abc_functional_scale_defaults_to_one(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})
    probes = {
        "mean": lambda y: jnp.mean(y["Y1"]),
        "std": lambda y: jnp.std(y["Y1"]),
    }

    default_traces = abc(
        struct,
        thetas_array,
        proposal=prop,
        probes=probes,
        epsilon=1e6,
        M=M,
        keys=keys,
    )

    explicit_traces = abc(
        struct,
        thetas_array,
        proposal=prop,
        probes=probes,
        scale={"mean": 1.0, "std": 1.0},
        epsilon=1e6,
        M=M,
        keys=keys,
    )

    for default_arr, explicit_arr in zip(default_traces, explicit_traces, strict=False):
        assert jnp.array_equal(default_arr, explicit_arr)


def test_pmcmc_and_abc_functional_par_trans():
    model = pp.models.lg()
    param_names = model.canonical_param_names
    first_param = param_names[0]

    def to_est(p: pp.types.ParamDict) -> pp.types.ParamDict:
        res = dict(p)
        res[first_param] = jnp.log(p[first_param])
        return res

    def from_est(p: pp.types.ParamDict) -> pp.types.ParamDict:
        res = dict(p)
        res[first_param] = jnp.exp(p[first_param])
        return res

    par_trans = pp.ParTrans(to_est=to_est, from_est=from_est)
    model.par_trans = par_trans
    struct = model.to_struct()

    n_reps = 1
    theta_val = model.theta.to_jax_array(param_names)
    thetas_array = jnp.repeat(theta_val, n_reps, axis=0)
    key = jax.random.key(123)
    keys = jax.random.split(key, n_reps)
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})

    _, _, theta_traces_pmcmc, _ = pmcmc(
        struct, thetas_array, proposal=prop, J=5, M=1, keys=keys
    )
    assert jnp.allclose(theta_traces_pmcmc[0, 0, :], theta_val[0])

    probes = {"mean": lambda y: jnp.mean(y["Y1"])}

    _, _, theta_traces_abc, _ = abc(
        struct,
        thetas_array,
        proposal=prop,
        probes=probes,
        epsilon=1e6,
        M=1,
        keys=keys,
    )
    assert jnp.allclose(theta_traces_abc[0, 0, :], theta_val[0])


def test_pmcmc_and_abc_functional_raw_dprior(model_setup):
    struct, thetas_array, key, J, n_reps, param_names = model_setup
    keys = jax.random.split(key, n_reps)
    M = 2
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})

    def raw_dprior(params: pp.types.ParamDict) -> float:
        return 0.0

    _, lp_traces, _, _ = pmcmc(
        struct,
        thetas_array,
        proposal=prop,
        J=J,
        M=M,
        keys=keys,
        dprior=raw_dprior,
    )
    assert lp_traces.shape == (n_reps, M + 1)
    assert jnp.all(jnp.isfinite(lp_traces))

    probes = {"mean": lambda y: jnp.mean(y["Y1"])}
    _, lp_traces_abc, _, _ = abc(
        struct,
        thetas_array,
        proposal=prop,
        probes=probes,
        epsilon=1e6,
        M=M,
        keys=keys,
        dprior=raw_dprior,
    )
    assert lp_traces_abc.shape == (n_reps, M + 1)
    assert jnp.all(jnp.isfinite(lp_traces_abc))
