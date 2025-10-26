# tests/test_parameter_trans_unit.py
# Unit tests for the parameter transform primitives only.
# These tests do not involve the model algorithms (MIF/MOP/Train).

import jax
import jax.numpy as jnp
import pytest

from pypomp.parameter_trans import (
    ParTrans,
    parameter_trans,
    _pt_forward,
    _pt_inverse,
    IDENTITY_PARTRANS,
)

def _rand_positive(key, shape):
    # Strictly positive values for log-transform coverage.
    return jnp.exp(jax.random.normal(key, shape))

def _rand_unit_interval(key, shape):
    # Values in (0, 1) for logit-transform coverage.
    return jax.random.uniform(key, shape, minval=1e-6, maxval=1 - 1e-6)

def test_forward_inverse_roundtrip_indices():
    # Round-trip on mixed transforms using integer indices.
    key = jax.random.key(0)
    d = 5
    J, B = 3, 2
    k1, k2, k3 = jax.random.split(key, 3)

    theta = jnp.stack(
        [
            _rand_positive(k1, (B, J)),          # idx 0 -> log
            jax.random.normal(k2, (B, J)),       # idx 1 -> custom
            _rand_positive(k1, (B, J)),          # idx 2 -> log
            jax.random.normal(k2, (B, J)),       # idx 3 -> custom
            _rand_unit_interval(k3, (B, J)),     # idx 4 -> logit
        ],
        axis=-1,
    )  # shape: (B, J, d)

    to_est = lambda x: 2.0 * x + 1.0
    from_est = lambda z: 0.5 * (z - 1.0)

    pt = parameter_trans(
        log=[0, 2],
        logit=[4],
        custom=[1, 3],
        to_est=to_est,
        from_est=from_est,
    )

    z = _pt_forward(theta, pt)
    back = _pt_inverse(z, pt)
    assert jnp.allclose(theta, back, atol=1e-8)

def test_forward_inverse_roundtrip_names_equiv_to_indices():
    # Name-based API must be equivalent to index-based API.
    names = ["a", "b", "c", "d", "e"]
    key = jax.random.key(1)
    theta = jax.random.normal(key, (4, len(names)))  # (J, d)

    to_est = lambda x: x ** 3
    from_est = lambda z: jnp.cbrt(z)

    pt_names = parameter_trans(
        log=["a"], logit=["e"], custom=["b", "d"],
        to_est=to_est, from_est=from_est, paramnames=names,
    )
    pt_idx = parameter_trans(
        log=[0], logit=[4], custom=[1, 3],
        to_est=to_est, from_est=from_est,
    )

    z1 = _pt_forward(theta, pt_names)
    z2 = _pt_forward(theta, pt_idx)
    assert jnp.allclose(z1, z2, atol=1e-8)
    back1 = _pt_inverse(z1, pt_names)
    back2 = _pt_inverse(z2, pt_idx)
    assert jnp.allclose(back1, back2, atol=1e-8)

def test_disjoint_sets_error():
    # Overlapping indices must raise.
    with pytest.raises(ValueError, match="disjoint"):
        _ = parameter_trans(log=[0, 1], logit=[1])

def test_names_without_paramnames_error():
    # Using names without supplying `paramnames` must raise.
    with pytest.raises(TypeError):
        _ = parameter_trans(log=["alpha"])

def test_custom_pair_required_error():
    # If any custom indices are given, both to_est/from_est must be provided.
    with pytest.raises(ValueError):
        _ = parameter_trans(custom=[0], to_est=lambda x: x)  # missing from_est

def test_custom_shape_mismatch_errors():
    # Custom transforms must return the same shape subvector.
    pt_bad_to = parameter_trans(
        custom=[0],
        to_est=lambda x: jnp.concatenate([x, x], axis=-1),
        from_est=lambda z: z,
    )
    with pytest.raises(ValueError, match="same shape"):
        _ = _pt_forward(jnp.zeros((2, 1)), pt_bad_to)

    pt_bad_from = parameter_trans(
        custom=[0],
        to_est=lambda x: x,
        from_est=lambda z: jnp.concatenate([z, z], axis=-1),
    )
    with pytest.raises(ValueError, match="same shape"):
        _ = _pt_inverse(jnp.zeros((2, 1)), pt_bad_from)

def test_clip_behavior_on_boundaries():
    """
    Numerical-stability test for boundary clipping:

    - For log-transform, 0 must be clipped to a small positive epsilon.
    - For logit-transform, 0 and 1 must be clipped into (0,1).

    Compare in the same dtype and allow a tiny rtol: float32 cannot represent
    1 - 1e-12 distinctly from 1.0, and round-trip is not bit-exact.
    """
    eps = 1e-12
    # Use float32 explicitly to match common JAX default when x64 is disabled.
    theta = jnp.array([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)  # (2,2): [log, logit]
    pt = parameter_trans(log=[0], logit=[1])

    z = _pt_forward(theta, pt)
    back = _pt_inverse(z, pt)

    eps_f = jnp.asarray(eps, dtype=back.dtype)
    one_minus_eps_f = jnp.asarray(1.0 - eps, dtype=back.dtype)  # may round to 1.0 in float32

    assert jnp.allclose(back[0, 0], eps_f, rtol=5e-6, atol=0.0)        # 0 -> ~eps
    assert jnp.allclose(back[0, 1], eps_f, rtol=5e-6, atol=0.0)        # 0 -> ~eps after logit^-1
    assert jnp.allclose(back[1, 1], one_minus_eps_f, rtol=5e-6, atol=0.0)  # 1 -> ~(1-eps)

def test_batching_lastdim_contracts():
    # Transform acts on the last dimension only and preserves batch axes.
    key = jax.random.key(2)
    x = jax.random.uniform(key, (2, 3, 7))  # arbitrary batch dims
    pt = parameter_trans(log=[1, 3], logit=[5], custom=[0], to_est=lambda u: u + 1, from_est=lambda v: v - 1)
    assert _pt_inverse(_pt_forward(x, pt), pt).shape == x.shape

def test_identity_constant():
    # Identity transform must leave arrays unchanged.
    key = jax.random.key(3)
    x = jax.random.normal(key, (5, 4))
    assert jnp.allclose(_pt_forward(x, IDENTITY_PARTRANS), x)
    assert jnp.allclose(_pt_inverse(x, IDENTITY_PARTRANS), x)

def test_is_custom_property():
    # Convenience property compatibility check.
    pt1 = ParTrans()
    pt2 = parameter_trans(custom=[0], to_est=lambda x: x, from_est=lambda z: z)
    assert not pt1.is_custom
    assert pt2.is_custom
