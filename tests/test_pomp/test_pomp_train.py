import jax
import pytest
import numpy as np
import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    # Set default values for tests
    LG = pp.models.LG()
    ys = LG.ys
    covars = None
    theta = LG.theta
    J = 2
    key = jax.random.key(111)
    M = 2
    return LG, ys, covars, theta, J, key, M


@pytest.mark.parametrize(
    "opt_instance",
    [
        pp.SGD(scale=True),
        pp.Newton(scale=True),
        pp.WeightedNewton(scale=True),
        pp.BFGS(scale=True),
        pp.Adam(scale=True),
        pp.FullMatrixAdam(scale=True),
    ],
)
def test_train_basic(opt_instance, simple):
    """Test basic train functionality with per-parameter learning rates."""
    LG, ys, covars, theta, J, key, M = simple
    eta = pp.LearningRate({param: 0.2 for param in LG.canonical_param_names})

    LG.train(J=J, M=M, eta=eta, optimizer=opt_instance, key=key)

    GD_out = LG.results_history[-1]
    traces = GD_out.traces_da
    # Check shape for first replicate
    assert traces.sel(theta_idx=0).shape == (M + 1, len(LG.theta[0]) + 1)
    # Check that "logLik" and all parameter names are in variable coordinate
    assert "logLik" in list(traces.coords["variable"].values)
    for param in LG.theta.params()[0].keys():
        assert param in list(traces.coords["variable"].values)
    assert all(isinstance(v, float) for v in LG.theta[0].values())
    from pypomp.core.optimizer import Optimizer

    assert isinstance(GD_out.optimizer, Optimizer)
    assert GD_out.optimizer.__class__.__name__ == opt_instance.__class__.__name__


def test_train_with_line_search(simple):
    """Test train with line search enabled."""
    LG, ys, covars, theta, J, key, M = simple
    eta = pp.LearningRate({param: 0.2 for param in LG.canonical_param_names})

    LG.train(
        J=J,
        M=M,
        eta=eta,
        optimizer=pp.SGD(scale=True, ls=True),
        n_monitors=1,
        key=key,
    )

    GD_out = LG.results_history[-1]
    traces = GD_out.traces_da
    assert traces.sel(theta_idx=0).shape == (M + 1, len(LG.theta[0]) + 1)
    assert "logLik" in list(traces.coords["variable"].values)
    assert GD_out.optimizer.ls is True


def test_train_validation(simple):
    """Test train input validation."""
    LG, ys, covars, theta, J, key, M = simple
    eta = pp.LearningRate({param: 0.2 for param in LG.canonical_param_names})

    # Invalid J should raise ValueError
    with pytest.raises(ValueError):
        LG.train(J=0, M=M, eta=eta, optimizer=pp.SGD(scale=True), key=key)

    # Wrong eta keys should raise ValueError
    wrong_eta = pp.LearningRate({"wrong_param": 0.1, "another_wrong": 0.2})
    with pytest.raises(ValueError, match="Parameter '.*' not found"):
        LG.train(J=J, M=M, eta=wrong_eta, key=key)

    # Missing eta keys should raise ValueError
    partial_eta = pp.LearningRate({LG.canonical_param_names[0]: 0.1})
    with pytest.raises(ValueError, match="Parameter '.*' not found"):
        LG.train(J=J, M=M, eta=partial_eta, key=key)

    # Wrong theta keys should raise an error
    bad_theta = {"not_a_param": 1.0}
    with pytest.raises(
        ValueError,
        match="theta parameter names must match canonical_param_names up to reordering",
    ):
        LG.train(J=J, M=M, eta=eta, key=key, theta=pp.PompParameters(bad_theta))


def test_train_param_order_invariance(simple):
    """Test that parameter order doesn't affect results."""
    LG, ys, covars, theta, J, key, M = simple
    eta = pp.LearningRate({param: 0.2 for param in LG.canonical_param_names})

    LG.train(J=J, M=M, eta=eta, optimizer=pp.Newton(scale=True), key=key, theta=theta)
    out1 = LG.results_history[-1].traces_da.values

    # Permute theta parameter order
    param_keys = list(theta.params()[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta.params()]

    LG.train(
        J=J,
        M=M,
        eta=eta,
        optimizer=pp.Newton(scale=True),
        key=key,
        theta=pp.PompParameters(permuted_theta),
    )
    out2 = LG.results_history[-1].traces_da.values
    np.testing.assert_allclose(out1, out2, atol=1e-7)


def test_different_learning_rates(simple):
    """Test that different per-parameter learning rates produce different results."""
    LG, ys, covars, theta, J, key, M = simple
    params = LG.canonical_param_names

    # Run with uniform learning rates
    eta_uniform = pp.LearningRate({param: 0.1 for param in params})
    LG.train(J=J, M=M, eta=eta_uniform, optimizer=pp.SGD(), key=key)
    out_uniform = LG.results_history[-1].traces_da.values

    # Run with varied learning rates
    eta_varied_dict = {params[0]: 0.05, params[1]: 0.2}
    for p in params[2:]:
        eta_varied_dict[p] = 0.1
    eta_varied = pp.LearningRate(eta_varied_dict)

    LG.train(J=J, M=M, eta=eta_varied, optimizer=pp.SGD(), key=key)
    out_varied = LG.results_history[-1].traces_da.values

    # Results should differ
    assert not np.allclose(out_uniform, out_varied, atol=1e-10)


def test_train_clipping(simple):
    """Test that gradient clipping correctly limits parameter updates."""
    LG, _, _, theta, J, key, _ = simple
    M = 1
    eta = pp.LearningRate(
        {param: 10.0 for param in LG.canonical_param_names}
    )  # Large learning rate

    LG.train(J=J, M=M, eta=eta, optimizer=pp.SGD(clip_norm=None), key=key, theta=theta)
    p0 = (
        LG.results_history[-1]
        .traces_da.sel(theta_idx=0, iteration=0, variable=LG.canonical_param_names)
        .values
    )
    p1_no_clip = (
        LG.results_history[-1]
        .traces_da.sel(theta_idx=0, iteration=1, variable=LG.canonical_param_names)
        .values
    )
    diff_no_clip = np.linalg.norm(p1_no_clip - p0)

    LG.train(J=J, M=M, eta=eta, optimizer=pp.SGD(clip_norm=1e-5), key=key, theta=theta)
    p1_clip = (
        LG.results_history[-1]
        .traces_da.sel(theta_idx=0, iteration=1, variable=LG.canonical_param_names)
        .values
    )
    diff_clip = np.linalg.norm(p1_clip - p0)

    assert diff_clip < diff_no_clip
