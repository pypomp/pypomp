import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp


# ---------------------------------------------------------------------------
# MVNDiagRW
# ---------------------------------------------------------------------------


def test_mvndiagrw_init_drops_nonpositive():
    prop = pp.MVNDiagRW({"a": 0.1, "b": 0.0, "c": -1.0, "d": 0.2})
    assert prop.param_names == ("a", "d")
    assert np.allclose(prop.sd_arr, [0.1, 0.2])


def test_mvndiagrw_init_empty_raises():
    with pytest.raises(ValueError, match="at least one positive entry"):
        pp.MVNDiagRW({"a": 0.0, "b": -1.0})


def test_mvndiagrw_eq():
    p1 = pp.MVNDiagRW({"a": 0.1, "b": 0.2})
    p2 = pp.MVNDiagRW({"a": 0.1, "b": 0.2})
    p3 = pp.MVNDiagRW({"a": 0.1, "b": 0.3})
    p4 = pp.MVNDiagRW({"a": 0.1, "c": 0.2})

    assert p1 == p2
    assert p1 != p3
    assert p1 != p4
    # Comparison against an unrelated type falls back to NotImplemented,
    # which Python turns into a False result via `!=`/`==`.
    assert p1 != "not a proposal"
    assert (p1 == "not a proposal") is False


def test_mvndiagrw_step():
    prop = pp.MVNDiagRW({"a": 0.1, "b": 0.2})
    theta = jnp.array([1.0, 2.0])
    key = jax.random.key(0)
    state = prop.init_state(theta)
    assert state == ()
    proposed, new_state = prop.step(state, theta, key, 0, 0)
    assert proposed.shape == theta.shape
    assert new_state == ()
    # Deterministic given the key: matches a manual normal draw.
    z = jax.random.normal(key, shape=theta.shape)
    expected = theta + z * prop.sd_arr
    assert jnp.allclose(proposed, expected)


def test_mvndiagrw_canonicalize():
    prop = pp.MVNDiagRW({"a": 0.1, "b": 0.2})
    canon = prop.canonicalize(["b", "a", "c"])
    assert canon.param_names == ("b", "a", "c")
    assert np.allclose(canon.sd_arr, [0.2, 0.1, 0.0])


def test_mvndiagrw_canonicalize_unknown_param_raises():
    prop = pp.MVNDiagRW({"a": 0.1, "z": 0.2})
    with pytest.raises(ValueError, match="Proposal parameter 'z' not in model"):
        prop.canonicalize(["a", "b"])


def test_mvndiagrw_pytree_roundtrip():
    prop = pp.MVNDiagRW({"a": 0.1, "b": 0.2})
    leaves, treedef = jax.tree_util.tree_flatten(prop)
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == prop


# ---------------------------------------------------------------------------
# MVNRWFull
# ---------------------------------------------------------------------------


def test_mvnrwfull_init_valid():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    prop = pp.MVNRWFull(cov, ["a", "b"])
    assert prop.param_names == ("a", "b")
    assert np.allclose(prop.chol, np.linalg.cholesky(cov))


def test_mvnrwfull_init_invalid_shapes():
    with pytest.raises(ValueError, match="rw_var must be a square matrix"):
        pp.MVNRWFull(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ["a", "b"])

    with pytest.raises(ValueError, match="rw_var dimensions must match"):
        pp.MVNRWFull(np.eye(2), ["a", "b", "c"])


def test_mvnrwfull_eq():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    p1 = pp.MVNRWFull(cov, ["a", "b"])
    p2 = pp.MVNRWFull(cov, ["a", "b"])
    p3 = pp.MVNRWFull(cov * 2, ["a", "b"])
    p4 = pp.MVNRWFull(cov, ["a", "c"])

    assert p1 == p2
    assert p1 != p3
    assert p1 != p4
    assert p1 != "not a proposal"
    assert (p1 == "not a proposal") is False


def test_mvnrwfull_step():
    cov = np.eye(2) * 0.01
    prop = pp.MVNRWFull(cov, ["a", "b"])
    theta = jnp.array([1.0, 2.0])
    key = jax.random.key(0)
    state = prop.init_state(theta)
    assert state == ()
    proposed, new_state = prop.step(state, theta, key, 0, 0)
    assert proposed.shape == theta.shape
    assert new_state == ()
    z = jax.random.normal(key, shape=theta.shape)
    expected = theta + prop.chol @ z
    assert jnp.allclose(proposed, expected)


def test_mvnrwfull_canonicalize():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    prop = pp.MVNRWFull(cov, ["a", "b"])
    canon = prop.canonicalize(["b", "a", "c"])
    assert canon.param_names == ("b", "a", "c")
    # Reordered Cholesky factor should reconstruct the reordered covariance.
    full_chol = np.asarray(canon.chol)
    reordered_cov = full_chol @ full_chol.T
    expected_cov = np.array(
        [
            [0.09, 0.01, 0.0],
            [0.01, 0.04, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(reordered_cov, expected_cov)


def test_mvnrwfull_canonicalize_unknown_row_param_raises():
    # An unknown name in the *first* row position is caught by the outer
    # (i_local) check before any column is examined.
    cov = np.eye(2) * 0.01
    prop = pp.MVNRWFull(cov, ["z", "a"])
    with pytest.raises(ValueError, match="Proposal parameter 'z' not in model"):
        prop.canonicalize(["a", "b"])


def test_mvnrwfull_canonicalize_unknown_col_param_raises():
    # With a valid first row, the unknown name is instead discovered while
    # scanning that row's columns -- exercising the inner (j_local) check.
    cov = np.eye(2) * 0.01
    prop = pp.MVNRWFull(cov, ["a", "z"])
    with pytest.raises(ValueError, match="Proposal parameter 'z' not in model"):
        prop.canonicalize(["a", "b"])


def test_mvnrwfull_pytree_roundtrip():
    cov = np.eye(2) * 0.01
    prop = pp.MVNRWFull(cov, ["a", "b"])
    leaves, treedef = jax.tree_util.tree_flatten(prop)
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == prop


# ---------------------------------------------------------------------------
# MVNRWAdaptive
# ---------------------------------------------------------------------------


def test_mvnrwadaptive_init_from_rw_sd():
    prop = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2})
    assert prop.param_names == ("a", "b")
    assert np.allclose(prop.init_rw_var, np.diag([0.01, 0.04]))
    assert prop.scale_start == 200
    assert prop.scale_cooling == 0.999
    assert prop.shape_start == 200
    assert prop.target == 0.234
    assert prop.max_scaling == 50.0


def test_mvnrwadaptive_init_from_rw_var():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    prop = pp.MVNRWAdaptive(rw_var=cov, param_names=["a", "b"])
    assert prop.param_names == ("a", "b")
    assert np.allclose(prop.init_rw_var, cov)


def test_mvnrwadaptive_init_requires_exactly_one_of_rw_sd_rw_var():
    with pytest.raises(
        ValueError, match="Exactly one of rw_sd and rw_var must be given"
    ):
        pp.MVNRWAdaptive()

    with pytest.raises(
        ValueError, match="Exactly one of rw_sd and rw_var must be given"
    ):
        pp.MVNRWAdaptive(rw_sd={"a": 0.1}, rw_var=np.eye(1))


def test_mvnrwadaptive_init_rw_var_requires_param_names():
    with pytest.raises(ValueError, match="param_names required when rw_var is given"):
        pp.MVNRWAdaptive(rw_var=np.eye(2))


def test_mvnrwadaptive_init_rw_var_shape_mismatch():
    with pytest.raises(ValueError, match="rw_var shape must match param_names"):
        pp.MVNRWAdaptive(rw_var=np.eye(2), param_names=["a", "b", "c"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"scale_start": 0}, "scale_start must be a positive integer"),
        ({"scale_start": -5}, "scale_start must be a positive integer"),
        ({"scale_cooling": 0.0}, r"scale_cooling must be in \(0, 1\]"),
        ({"scale_cooling": 1.5}, r"scale_cooling must be in \(0, 1\]"),
        ({"shape_start": 0}, "shape_start must be a positive integer"),
        ({"shape_start": -1}, "shape_start must be a positive integer"),
        ({"target": 0.0}, r"target must be in \(0, 1\)"),
        ({"target": 1.0}, r"target must be in \(0, 1\)"),
    ],
)
def test_mvnrwadaptive_init_invalid_scalars(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pp.MVNRWAdaptive(rw_sd={"a": 0.1}, **kwargs)


def test_mvnrwadaptive_eq():
    p1 = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2})
    p2 = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2})
    assert p1 == p2
    assert p1 != "not a proposal"
    assert (p1 == "not a proposal") is False

    # Each scalar field differing should break equality.
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.1, "c": 0.2})
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.5, "b": 0.2})
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, scale_start=100)
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, scale_cooling=0.9)
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, shape_start=100)
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, target=0.4)
    assert p1 != pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, max_scaling=10.0)


def test_mvnrwadaptive_init_state_and_step():
    prop = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2})
    theta = jnp.array([1.0, 2.0])
    state = prop.init_state(theta)
    assert np.allclose(state.scaling, 1.0)
    assert np.allclose(state.theta_mean, [0.0, 0.0])
    assert np.allclose(state.covmat_emp, np.zeros((2, 2)))
    assert np.allclose(state.initialized, 0.0)

    key = jax.random.key(0)
    proposed, new_state = prop.step(state, theta, key, n=1, accepts=0)
    assert proposed.shape == theta.shape
    # theta_mean is seeded with theta on the very first call.
    assert np.allclose(new_state.theta_mean, theta)
    assert np.allclose(new_state.initialized, 1.0)

    # A second call continues from the returned state (exercises the
    # already-initialized branch of the lazy seeding, and phase-1 scaling
    # once n exceeds scale_start).
    key2 = jax.random.key(1)
    proposed2, new_state2 = prop.step(new_state, proposed, key2, n=250, accepts=5)
    assert proposed2.shape == theta.shape
    assert new_state2.initialized == 1.0


def test_mvnrwadaptive_step_phase2():
    """Once accepts >= shape_start, the empirical covariance branch is used."""
    prop = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, shape_start=1, scale_start=1)
    theta = jnp.array([1.0, 2.0])
    state = prop.init_state(theta)
    key = jax.random.key(0)
    state = prop.step(state, theta, key, n=1, accepts=0)[1]
    # Second step has accepts >= shape_start=1, triggering phase 2 covariance.
    proposed, new_state = prop.step(state, theta, jax.random.key(1), n=2, accepts=1)
    assert proposed.shape == theta.shape
    assert new_state.initialized == 1.0


def test_mvnrwadaptive_canonicalize():
    prop = pp.MVNRWAdaptive(
        rw_var=np.array([[0.04, 0.01], [0.01, 0.09]]), param_names=["a", "b"]
    )
    canon = prop.canonicalize(["b", "a", "c"])
    assert canon.param_names == ("b", "a", "c")
    assert canon.scale_start == prop.scale_start
    assert canon.scale_cooling == prop.scale_cooling
    assert canon.shape_start == prop.shape_start
    assert canon.target == prop.target
    assert canon.max_scaling == prop.max_scaling
    expected = np.array(
        [
            [0.09, 0.01, 0.0],
            [0.01, 0.04, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(canon.init_rw_var, expected)


def test_mvnrwadaptive_canonicalize_unknown_row_param_raises():
    # Unknown name in the first row position -> outer (i_local) check.
    prop = pp.MVNRWAdaptive(rw_sd={"z": 0.1, "a": 0.2})
    with pytest.raises(ValueError, match="Proposal parameter 'z' not in model"):
        prop.canonicalize(["a", "b"])


def test_mvnrwadaptive_canonicalize_unknown_col_param_raises():
    # Valid first row, unknown name found while scanning its columns ->
    # inner (j_local) check.
    prop = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "z": 0.2})
    with pytest.raises(ValueError, match="Proposal parameter 'z' not in model"):
        prop.canonicalize(["a", "b"])


def test_mvnrwadaptive_pytree_roundtrip():
    prop = pp.MVNRWAdaptive(rw_sd={"a": 0.1, "b": 0.2}, scale_start=50)
    leaves, treedef = jax.tree_util.tree_flatten(prop)
    assert len(leaves) == 1
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == prop
    assert rebuilt.scale_start == 50
