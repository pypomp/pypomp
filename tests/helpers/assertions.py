"""Assertion and round-trip utilities shared across the test suite."""

import pickle
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any, TypeVar

import jax
import numpy as np

T = TypeVar("T")


def pickle_roundtrip(obj: T) -> T:
    """Return ``obj`` after a pickle dump/load cycle."""
    return pickle.loads(pickle.dumps(obj))


def reversed_theta(theta: Mapping[str, Any]) -> dict[str, Any]:
    """``theta`` with its keys in reverse order.

    Results must not depend on the order the user supplied parameters in, since
    they are aligned to canonical_param_names internally.
    """
    return {k: theta[k] for k in reversed(list(theta))}


@contextmanager
def jax_x64_enabled() -> Generator[None, None, None]:
    """Context manager to temporarily enable x64 mode in JAX."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


def calculate_empirical_moments(samples: np.ndarray) -> tuple[float, float, float]:
    """Calculate mean, variance, and skewness of the samples."""
    mean_emp = float(samples.mean())
    var_emp = float(samples.var())
    centered = samples - mean_emp
    m3 = float(np.mean(centered**3))
    std_emp = float(np.std(samples))
    skew_emp = m3 / (std_emp**3) if std_emp > 0 else 0.0
    return mean_emp, var_emp, skew_emp


def check_moments(
    dist_name: str,
    params_str: str,
    samples: np.ndarray,
    mean_th: float,
    var_th: float,
    skew_th: float = 0.0,
    mean_tol: tuple[float, float] = (0.02, 0.02),
    var_tol: tuple[float, float] = (0.03, 0.03),
    skew_tol: tuple[float, float] = (0.10, 0.04),
    check_skew: bool = False,
) -> None:
    """Compare empirical against theoretical moments, failing if they diverge."""
    mean_emp, var_emp, skew_emp = calculate_empirical_moments(samples)

    assert np.allclose(mean_emp, mean_th, rtol=mean_tol[0], atol=mean_tol[1]), (
        f"{dist_name} mean fail for {params_str}. Empirical: {mean_emp}, Theoretical: {mean_th}"
    )
    assert np.allclose(var_emp, var_th, rtol=var_tol[0], atol=var_tol[1]), (
        f"{dist_name} var fail for {params_str}. Empirical: {var_emp}, Theoretical: {var_th}"
    )
    if check_skew:
        assert np.allclose(skew_emp, skew_th, rtol=skew_tol[0], atol=skew_tol[1]), (
            f"{dist_name} skew fail for {params_str}. Empirical: {skew_emp}, Theoretical: {skew_th}"
        )
