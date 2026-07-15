import jax.numpy as jnp
import pypomp.random as ppr
from tests.test_random.helpers import jax_x64_enabled


def test_binominv_basics() -> None:
    u = jnp.array([0.1, 0.5, 0.9])
    n_binom = jnp.array([10.0, 10.0, 10.0])
    p_binom = jnp.array([0.5, 0.5, 0.5])

    res_binom_default = ppr.binominv(u, n_binom, p_binom)
    assert res_binom_default.shape == (3,)
    assert res_binom_default.dtype == jnp.float32


def test_binominv_dtypes() -> None:
    """Verify dtype overrides and x64 precision behavior for binominv."""
    u = jnp.array([0.1, 0.5, 0.9])
    n_binom = jnp.array([10.0, 10.0, 10.0])
    p_binom = jnp.array([0.5, 0.5, 0.5])

    res_binom_int = ppr.binominv(u, n_binom, p_binom, dtype=jnp.int32)
    assert res_binom_int.dtype == jnp.int32
    assert jnp.all(res_binom_int >= 0)

    with jax_x64_enabled():
        res_binom_float64 = ppr.binominv(u, n_binom, p_binom, dtype=jnp.float64)
        assert res_binom_float64.dtype == jnp.float64

        u_64 = jnp.array([0.1, 0.5, 0.9])
        n_64 = jnp.array([10.0, 10.0, 10.0])
        p_64 = jnp.array([0.5, 0.5, 0.5])
        res_binom_default_64 = ppr.binominv(u_64, n_64, p_64)
        assert res_binom_default_64.dtype == jnp.float64


def test_binominv_invalid_inputs() -> None:
    n_binom = jnp.array([10.0, 10.0, 10.0])
    p_binom = jnp.array([0.5, 0.5, 0.5])

    invalid_u = jnp.array([-0.5, 1.5, 0.5])
    res_binom_invalid = ppr.binominv(invalid_u, n_binom, p_binom, dtype=jnp.int32)
    assert res_binom_invalid[0] == -1
    assert res_binom_invalid[1] == -1
    assert res_binom_invalid[2] >= 0


def test_poissoninv_basics_and_dtypes() -> None:
    u = jnp.array([0.1, 0.5, 0.9])
    lam = jnp.array([1.0, 4.0, 10.0])

    res_poisson_default = ppr.poissoninv(u, lam)
    assert res_poisson_default.shape == (3,)
    assert res_poisson_default.dtype == jnp.float32

    res_poisson_int = ppr.poissoninv(u, lam, dtype=jnp.int32)
    assert res_poisson_int.dtype == jnp.int32
    assert jnp.all(res_poisson_int >= 0)

    with jax_x64_enabled():
        res_poisson_float64 = ppr.poissoninv(u, lam, dtype=jnp.float64)
        assert res_poisson_float64.dtype == jnp.float64

        u_64 = jnp.array([0.1, 0.5, 0.9])
        lam_64 = jnp.array([1.0, 4.0, 10.0])
        res_poisson_default_64 = ppr.poissoninv(u_64, lam_64)
        assert res_poisson_default_64.dtype == jnp.float64


def test_poissoninv_invalid_and_edge_inputs() -> None:
    u = jnp.array([0.1, 0.5, 0.9])

    invalid_lam = jnp.array([-1.0, 1.0, 1.0])
    res_poisson_invalid = ppr.poissoninv(u, invalid_lam, dtype=jnp.int32)
    assert res_poisson_invalid[0] == -1
    assert res_poisson_invalid[1] >= 0

    res_poisson_u1 = ppr.poissoninv(jnp.array([1.0]), jnp.array([2.0]), dtype=jnp.int32)
    assert res_poisson_u1[0] == -1


def test_gammainv_basics_and_dtypes() -> None:
    u = jnp.array([0.1, 0.5, 0.9])
    alpha = jnp.array([1.0, 2.0, 5.0])

    res_gamma_default = ppr.gammainv(u, alpha)
    assert res_gamma_default.shape == (3,)
    assert res_gamma_default.dtype == jnp.float32

    res_gamma_int = ppr.gammainv(u, alpha, dtype=jnp.int32)
    assert res_gamma_int.dtype == jnp.int32
    assert jnp.all(res_gamma_int >= -1)  # invalid or valid

    with jax_x64_enabled():
        res_gamma_float64 = ppr.gammainv(u, alpha, dtype=jnp.float64)
        assert res_gamma_float64.dtype == jnp.float64

        u_64 = jnp.array([0.1, 0.5, 0.9])
        alpha_64 = jnp.array([1.0, 2.0, 5.0])
        res_gamma_default_64 = ppr.gammainv(u_64, alpha_64)
        assert res_gamma_default_64.dtype == jnp.float64


def test_gammainv_edge_inputs() -> None:
    res_gamma_u1 = ppr.gammainv(jnp.array([1.0]), jnp.array([2.0]), dtype=jnp.int32)
    assert res_gamma_u1[0] == -1  # inf becomes -1
