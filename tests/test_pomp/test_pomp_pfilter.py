import sys
from copy import deepcopy

import jax
import jax.numpy as jnp
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.core.algorithms.contexts import PfilterContext
from pypomp.core.algorithms.pfilter import _even_batch

# pypomp.functional.pfilter is the function, which hides the module of that name.
PFILTER_MODULE = sys.modules["pypomp.functional.pfilter"]


@pytest.fixture(scope="module")
def simple_setup_module():
    LG = pp.models.lg()
    key = jax.random.key(111)
    J = 3
    return LG, key, J


@pytest.fixture(scope="function")
def simple(simple_setup_module):
    LG_orig, key, J = simple_setup_module
    LG = deepcopy(LG_orig)
    LG.results_history.clear()
    return LG, key, J


def test_class_basic_default(simple):
    LG, key, J = simple
    LG.pfilter(J=J, key=key)
    val1 = LG.results_history[-1].logLiks
    assert val1.shape == (1, 1)
    assert jnp.isfinite(val1.item())
    assert val1.dtype == jnp.float32
    assert LG.results_history[-1].CLL_da is None


def test_reps_default(simple):
    LG, key, J = simple
    theta = LG.theta.params(as_list=True)
    theta_list = [theta[0], {k: v * 2 for k, v in theta[0].items()}]
    LG.pfilter(J=J, key=key, theta=pp.PompParameters(theta_list), reps=2)
    val1 = LG.results_history[-1].logLiks
    assert val1.shape == (2, 2)
    assert LG.results_history[-1].CLL_da is None


def test_reps_batch_size_matches_parallel(simple):
    LG, key, J = simple
    LG.pfilter(J=J, key=key, reps=4)
    parallel = LG.results_history[-1].logLiks.values
    for batch in (1, 3):
        LG.pfilter(J=J, key=key, reps=4, reps_batch_size=batch)
        assert jnp.allclose(LG.results_history[-1].logLiks.values, parallel, rtol=1e-5)


def test_all_replicates_at_once_match_one_at_a_time(simple):
    LG, key, J = simple
    struct = LG.to_struct()
    th = LG.theta.to_jax_array(LG.canonical_param_names)
    th = jnp.concatenate([th, th * 1.1])
    keys = jax.random.split(key, 6).reshape(2, 3)
    one, flat = (
        F.pfilter(struct, th, J, keys, CLL=True, filter_mean=True, reps_batch_size=b)
        for b in (1, 3)
    )
    for k in one:
        assert flat[k].shape == one[k].shape
        assert jnp.allclose(flat[k], one[k], rtol=1e-5)


def test_gpu_path_runs_replicates_in_one_batch(simple, monkeypatch):
    LG, key, J = simple
    struct = LG.to_struct()
    th = LG.theta.to_jax_array(LG.canonical_param_names)
    keys = jax.random.split(key, 3)[None]
    one = F.pfilter(struct, th, J, keys, reps_batch_size=1)["logLik"]
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    eager = F.pfilter(struct, th, J, keys)["logLik"]
    traced = jax.jit(lambda t: F.pfilter(struct, t, J, keys)["logLik"])(th)
    assert jnp.allclose(eager, one, rtol=1e-5)
    assert jnp.allclose(traced, one, rtol=1e-5)


def test_batch_for_memory_budget():
    from pypomp.functional.pfilter import _batch_for

    assert _batch_for(100, 600, 1000, 10) is None
    # Budget 800: 600 fixed leaves room for one filter's 140 of workspace.
    assert _batch_for(600, 1400, 1000, 10) == 1
    assert _batch_for(0, 1000, 1000, 10) == 8
    assert _batch_for(10**12, 10**12, 1000, 10) == 1


@pytest.mark.parametrize(
    "n, reps, batch", [(2, 3, 2), (2, 3, 4), (5, 1, 2), (3, 2, 100)]
)
def test_gpu_path_batches_match_one_at_a_time(simple, monkeypatch, n, reps, batch):
    """Batches cut across parameter sets and pad unevenly without changing results."""
    LG, key, J = simple
    struct = LG.to_struct()
    th = LG.theta.to_jax_array(LG.canonical_param_names)
    th = jnp.concatenate([th * (1 + 0.1 * i) for i in range(n)])
    keys = jax.random.split(key, n * reps).reshape(n, reps)
    one = F.pfilter(struct, th, J, keys, CLL=True, reps_batch_size=1)

    monkeypatch.setattr(PFILTER_MODULE, "_PARTICLES_PER_BATCH", batch * J)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    batched = F.pfilter(struct, th, J, keys, CLL=True)
    for k in one:
        assert batched[k].shape == one[k].shape
        assert jnp.allclose(batched[k], one[k], rtol=1e-5)


def test_gpu_path_splits_filters_across_devices(simple):
    """The per-device batching keeps results when the filters span two devices."""
    LG, key, J = simple
    struct = LG.to_struct()
    th = jnp.tile(LG.theta.to_jax_array(LG.canonical_param_names), (2, 1))
    keys = jax.random.split(key, 6).reshape(2, 3)
    one = F.pfilter(struct, th, J, keys, reps_batch_size=1)["logLik"]
    out = PFILTER_MODULE._filter_batches(
        th, keys, PfilterContext.from_struct(struct, J=J), batch=2, n_devices=2
    )
    assert jnp.allclose(-out["neg_loglik"], one, rtol=1e-5)


def test_even_batch_avoids_padding():
    even_batch = _even_batch
    assert even_batch(16, 4) == 4
    # 3 would pad 16 filters to 18, and 2 divides them evenly.
    assert even_batch(16, 3) == 2
    assert even_batch(64, 5) == 4
    assert even_batch(7, 1) == 1
    # 3 needs the same padding as 5, so the larger wins.
    assert even_batch(14, 5) == 5
    # Filters that no batch divides still pad as little as possible.
    assert even_batch(13, 4) == 2


def test_gpu_path_shrinks_batches_to_the_memory_budget(simple, monkeypatch):
    LG, key, J = simple
    struct = LG.to_struct()
    th = LG.theta.to_jax_array(LG.canonical_param_names)
    keys = jax.random.split(key, 4)[None]
    one = F.pfilter(struct, th, J, keys, reps_batch_size=1)["logLik"]

    batches = []
    real = PFILTER_MODULE._filter_batches

    def spy(*args, **kwargs):
        batches.append(kwargs["batch"])
        return real(*args, **kwargs)

    spy.lower = real.lower  # type: ignore[attr-defined]
    monkeypatch.setattr(PFILTER_MODULE, "_filter_batches", spy)
    monkeypatch.setattr(PFILTER_MODULE, "_memory_bytes", lambda c: (0, 4 * 10**9))
    monkeypatch.setattr(PFILTER_MODULE, "_free_device_bytes", lambda a: 10**9)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    out = F.pfilter(struct, th, J, keys)["logLik"]
    assert batches == [1]
    assert jnp.allclose(out, one, rtol=1e-5)


def test_reps_batch_size_validation(simple):
    LG, key, J = simple
    with pytest.raises(ValueError, match="reps_batch_size"):
        LG.pfilter(J=J, key=key, reps=2, reps_batch_size=0)
    struct = LG.to_struct()
    th = LG.theta.to_jax_array(LG.canonical_param_names)
    keys = jax.random.split(key, 2)[None]
    with pytest.raises(ValueError, match="reps_batch_size"):
        F.pfilter(struct, th, J, keys, reps_batch_size=-1)


@pytest.mark.parametrize(
    "theta_spec, reps, expected_shape",
    [
        ("single", 1, (1, 1)),
        ("double", 2, (2, 2)),
    ],
)
@pytest.mark.parametrize(
    "CLL, ESS, filter_mean, prediction_mean",
    [
        (False, False, False, False),
        (True, True, False, False),
        (False, False, True, True),
    ],
)
def test_diagnostics(
    simple, theta_spec, reps, expected_shape, CLL, ESS, filter_mean, prediction_mean
):
    LG, key, J = simple
    theta_params = LG.theta.params(as_list=True)
    if theta_spec == "single":
        theta = pp.PompParameters(theta_params[0])
    else:
        theta = pp.PompParameters(
            [theta_params[0], {k: v * 2 for k, v in theta_params[0].items()}]
        )

    ys = LG.ys
    LG.results_history.clear()
    LG.pfilter(
        J=J,
        key=key,
        theta=theta,
        reps=reps,
        CLL=CLL,
        ESS=ESS,
        filter_mean=filter_mean,
        prediction_mean=prediction_mean,
    )
    method = LG.results_history[-1].method
    assert method == "pfilter"
    negLogLiks = LG.results_history[-1].logLiks
    negLogLiks_arr = negLogLiks.data
    assert negLogLiks_arr.shape == expected_shape
    assert jnp.all(jnp.isfinite(negLogLiks_arr))
    assert jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)

    # CLL:
    if CLL:
        condLogLiks = LG.results_history[-1].CLL_da
        assert condLogLiks is not None
        condLogLiks_arr = condLogLiks.data
        assert condLogLiks_arr.shape == expected_shape + (len(ys),)
        assert jnp.all(jnp.isfinite(condLogLiks_arr))
        assert jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)
    else:
        assert LG.results_history[-1].CLL_da is None

    # ESS:
    if ESS:
        ess = LG.results_history[-1].ESS_da
        assert ess is not None
        ess_arr = ess.data
        assert ess_arr.shape == expected_shape + (len(ys),)
        assert jnp.all(jnp.isfinite(ess_arr))
        assert jnp.issubdtype(ess_arr.dtype, jnp.floating)
        # all elements should be  between 0 and J inclusive
        assert jnp.all((ess_arr >= 0) & (ess_arr <= J))
    else:
        assert LG.results_history[-1].ESS_da is None

    # filter_mean:
    if filter_mean:
        filt_mean = LG.results_history[-1].filter_mean
        assert filt_mean is not None
        filter_mean_arr = filt_mean.data
        assert filter_mean_arr.shape == expected_shape + (len(ys), 2)
        assert jnp.all(jnp.isfinite(filter_mean_arr))
        assert jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)
    else:
        assert LG.results_history[-1].filter_mean is None

    # prediction_mean:
    if prediction_mean:
        pred_mean = LG.results_history[-1].prediction_mean
        assert pred_mean is not None
        prediction_mean_arr = pred_mean.data
        assert prediction_mean_arr.shape == expected_shape + (len(ys), 2)
        assert jnp.all(jnp.isfinite(prediction_mean_arr))
        assert jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)
    else:
        assert LG.results_history[-1].prediction_mean is None


def test_pfilter_invalid_theta_keys(simple):
    """theta with non-canonical keys should raise an error."""
    LG, key, J = simple
    # Use a theta dict with the wrong keys
    bad_theta = {"not_a_param": 1.0}
    with pytest.raises(
        ValueError,
        match="theta parameter names must match canonical_param_names up to reordering",
    ):
        LG.pfilter(J=J, key=key, theta=pp.PompParameters(bad_theta))


def test_pfilter_validation(simple):
    """Test input validation for pfilter."""
    LG, key, J = simple
    # Invalid J should raise ValueError
    with pytest.raises(ValueError, match="J should be greater than 0"):
        LG.pfilter(J=0, key=key)


def test_thresh_is_a_weight_ratio(simple):
    """thresh compares the max/min weight ratio, so values below 1 always resample."""
    LG, key, _ = simple
    logliks = {}
    for thresh in [0.0, 0.5, 0.99, 1e6]:
        LG.pfilter(J=200, key=key, thresh=thresh)
        logliks[thresh] = LG.results_history[-1].logLiks.item()
    assert logliks[0.5] == logliks[0.0]
    assert logliks[0.99] == logliks[0.0]
    assert logliks[1e6] != logliks[0.0]
