import jax
import numpy as np
import warnings
from contextlib import contextmanager
from typing import Generator, Tuple


@contextmanager
def jax_x64_enabled() -> Generator[None, None, None]:
    """Context manager to temporarily enable x64 mode in JAX."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


def calculate_empirical_moments(samples: np.ndarray) -> Tuple[float, float, float]:
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
    mean_tol: Tuple[float, float] = (0.02, 0.02),
    var_tol: Tuple[float, float] = (0.03, 0.03),
    skew_tol: Tuple[float, float] = (0.10, 0.04),
    check_skew: bool = False,
) -> None:
    """Utility to compare empirical vs theoretical moments and issue warnings if they diverge."""
    mean_emp, var_emp, skew_emp = calculate_empirical_moments(samples)

    if not np.allclose(mean_emp, mean_th, rtol=mean_tol[0], atol=mean_tol[1]):
        warnings.warn(
            f"{dist_name} mean fail for {params_str}. Empirical: {mean_emp}, Theoretical: {mean_th}"
        )
    if not np.allclose(var_emp, var_th, rtol=var_tol[0], atol=var_tol[1]):
        warnings.warn(
            f"{dist_name} var fail for {params_str}. Empirical: {var_emp}, Theoretical: {var_th}"
        )
    if check_skew:
        if not np.allclose(skew_emp, skew_th, rtol=skew_tol[0], atol=skew_tol[1]):
            warnings.warn(
                f"{dist_name} skew fail for {params_str}. Empirical: {skew_emp}, Theoretical: {skew_th}"
            )
