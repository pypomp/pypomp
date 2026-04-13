"""
He10 model with continuous process model.
"""

import jax.numpy as jnp
import jax
import jax.scipy.special as jspecial
from pypomp.random.gammainvf import fast_approx_rgamma
from pypomp.types import (
    ObservationDict,
    StateDict,
    ParamDict,
    CovarDict,
    TimeFloat,
    RNGKey,
    InitialTimeFloat,
    StepSizeFloat,
)
from jax.scipy.special import log_ndtr


def softclamp(x, floor: float | jax.Array = 0.0, sharpness: float = 100.0):
    """Smooth, differentiable approximation to jnp.maximum(x, floor).

    Uses softplus: floor + softplus(sharpness * (x - floor)) / sharpness
    As sharpness -> inf this converges to max(x, floor).
    """
    return floor + jax.nn.softplus(sharpness * (x - floor)) / sharpness


param_names = (
    "R0",
    "sigma",
    "gamma",
    "iota",
    "rho",
    "sigmaSE",
    "psi",
    "cohort",
    "amplitude",
    "S_0",
    "E_0",
    "I_0",
    "R_0",
)

statenames = ["S", "E", "I", "R", "W", "C"]
accumvars = ["W", "C"]


def rinit(
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t0: InitialTimeFloat | None = None,
):
    S_0 = theta_["S_0"]
    E_0 = theta_["E_0"]
    I_0 = theta_["I_0"]
    R_0 = theta_["R_0"]

    m = covars["pop"] / (S_0 + E_0 + I_0 + R_0)

    S = m * S_0
    E = m * E_0
    I = m * I_0
    R = m * R_0
    W = 0.0
    C = 0.0

    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


def rproc(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
    dt: StepSizeFloat,
):
    S, E, I = X_["S"], X_["E"], X_["I"]
    W, C = X_["W"], X_["C"]

    R0 = theta_["R0"]
    sigma = theta_["sigma"]
    gamma = theta_["gamma"]
    iota = theta_["iota"]
    sigmaSE = theta_["sigmaSE"]
    cohort = theta_["cohort"]
    amplitude = theta_["amplitude"]
    pop = covars["pop"]
    birthrate = covars["birthrate"]
    mu = 0.02

    # Seasonality
    t_mod = t - jnp.floor(t)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    br = jnp.where(
        is_cohort_time,
        cohort * birthrate / dt + (1 - cohort) * birthrate,
        (1 - cohort) * birthrate,
    )

    t_days = t_mod * 365.25
    in_term_time = (
        ((t_days >= 7) & (t_days <= 100))
        | ((t_days >= 115) & (t_days <= 199))
        | ((t_days >= 252) & (t_days <= 300))
        | ((t_days >= 308) & (t_days <= 356))
    )
    seas = jnp.where(in_term_time, 1.0 + amplitude * 0.2411 / 0.7589, 1 - amplitude)

    # Transmission Rate
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # Expected Force of Infection
    foi = beta * (I + iota) / pop

    normal_keys, gamma_key = jax.random.split(key, 2)
    all_noise = jax.random.normal(normal_keys, shape=(7,))

    dw = fast_approx_rgamma(gamma_key, dt / sigmaSE**2) * sigmaSE**2

    birth_mean = br * dt
    birth_noise = all_noise[0]
    # Use a 1e-8 floor to avoid gradient numerical instability
    safe_birth_mean = softclamp(birth_mean)
    births = softclamp(birth_mean + jnp.sqrt(safe_birth_mean) * birth_noise)

    # Rates
    rate_inf = foi * (dw / dt)  # effective infection rate

    flux_noises = all_noise[1:7]

    rates = jnp.array([rate_inf, mu, sigma, mu, gamma, mu])
    states = jnp.array([S, S, E, E, I, I])

    mu_fluxes = rates * states * dt

    # Use a 1e-8 floor to avoid gradient numerical instability
    safe_mu_fluxes = softclamp(mu_fluxes)
    fluxes = mu_fluxes + jnp.sqrt(safe_mu_fluxes) * flux_noises

    flux_SE, flux_SD, flux_EI, flux_ED, flux_IR, flux_ID = fluxes

    S_new = softclamp(S + births - flux_SE - flux_SD)
    E_new = softclamp(E + flux_SE - flux_EI - flux_ED)
    I_new = softclamp(I + flux_EI - flux_IR - flux_ID)
    R_new = softclamp(pop - S_new - E_new - I_new)

    W_new = W + (dw - dt) / sigmaSE
    C_new = softclamp(C + flux_EI)

    return {"S": S_new, "E": E_new, "I": I_new, "R": R_new, "W": W_new, "C": C_new}


def dmeas_continuous(Y_, X_, theta_, covars=None, t=None):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]

    y = jnp.asarray(Y_["cases"])
    safe_y = jnp.nan_to_num(y, nan=0.0)
    safe_C = softclamp(C)

    m = rho * safe_C
    v = m * (1.0 - rho + psi**2 * m)
    scale = jnp.sqrt(softclamp(v))
    loglik = jax.scipy.stats.norm.logpdf(safe_y, loc=m, scale=scale)
    loglik = jnp.where(jnp.isnan(y), 0.0, loglik)
    loglik = jnp.where(C < 0, -jnp.inf, loglik)

    return loglik


LOG_SQRT_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


def _log_phi(z):
    return -0.5 * z * z - LOG_SQRT_2PI


def log_cdf_single(z):
    return log_cdf_diff(z, -jnp.inf)


def _log_sub_exp_stable(a, b):
    return a + jnp.log1p(-jnp.exp(b - a))


@jax.custom_jvp
def log_cdf_diff(zh, zl):
    a = log_ndtr(zh)
    b = log_ndtr(zl)
    hi = jnp.maximum(a, b)
    lo = jnp.minimum(a, b)
    return _log_sub_exp_stable(hi, lo)


@log_cdf_diff.defjvp
def _log_cdf_diff_jvp(primals, tangents):
    zh, zl = primals
    tzh, tzl = tangents
    y = log_cdf_diff(zh, zl)
    lphi_h = _log_phi(zh)
    lphi_l = _log_phi(zl)

    LOG_MAX = jnp.log(jnp.finfo(zh.dtype).max)

    diff_h = lphi_h - y
    diff_l = lphi_l - y

    safe_diff_h = jnp.where(
        jnp.isfinite(diff_h), jnp.clip(diff_h, -LOG_MAX, LOG_MAX), -LOG_MAX
    )
    safe_diff_l = jnp.where(
        jnp.isfinite(diff_l), jnp.clip(diff_l, -LOG_MAX, LOG_MAX), -LOG_MAX
    )

    r_hi = jnp.exp(safe_diff_h)
    r_lo = jnp.exp(safe_diff_l)
    dy = r_hi * tzh - r_lo * tzl
    return y, dy


def dmeas(
    Y_: ObservationDict,
    X_: StateDict,
    theta_: ParamDict,
    covars: CovarDict | None = None,
    t: TimeFloat | None = None,
):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    y_raw = jnp.asarray(Y_["cases"])

    tol = 1e-12

    y_is_nan = jnp.isnan(y_raw)
    y = jnp.where(y_is_nan, 0.0, y_raw)

    Cpos = jnp.maximum(C, 0.0)
    m = rho * Cpos
    v = m * (1.0 - rho + psi**2 * m)
    v = jnp.maximum(v, tol**2)
    s = jnp.sqrt(v)

    z_hi = (y + 0.5 - m) / s
    z_lo = (y - 0.5 - m) / s

    ll_box = jnp.where(y > tol, log_cdf_diff(z_hi, z_lo), log_cdf_single(z_hi))

    loglik = jnp.maximum(ll_box, jnp.log(tol))

    loglik = loglik * (1.0 - y_is_nan.astype(loglik.dtype))

    return loglik


def rmeas(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict | None = None,
    t: TimeFloat | None = None,
):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = jnp.asarray(X_["C"])
    safe_C = softclamp(C)

    m = rho * safe_C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18

    cases = jax.random.normal(key) * (jnp.sqrt(softclamp(v)) + tol) + m

    return softclamp(cases)


def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    SEIR_0 = jnp.array([theta["S_0"], theta["E_0"], theta["I_0"], theta["R_0"]])
    S_0, E_0, I_0, R_0 = jnp.log(SEIR_0 / jnp.sum(SEIR_0))
    return {
        "R0": jnp.log(theta["R0"]),
        "sigma": jnp.log(theta["sigma"]),
        "gamma": jnp.log(theta["gamma"]),
        "iota": jnp.log(theta["iota"]),
        "sigmaSE": jnp.log(theta["sigmaSE"]),
        "psi": jnp.log(theta["psi"]),
        "cohort": jspecial.logit(theta["cohort"]),
        "amplitude": jspecial.logit(theta["amplitude"]),
        "rho": jspecial.logit(theta["rho"]),
        "S_0": S_0,
        "E_0": E_0,
        "I_0": I_0,
        "R_0": R_0,
    }


def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    SEIR_0 = jnp.exp(
        jnp.array([theta["S_0"], theta["E_0"], theta["I_0"], theta["R_0"]])
    )
    S_0, E_0, I_0, R_0 = SEIR_0 / jnp.sum(SEIR_0)
    return {
        "R0": jnp.exp(theta["R0"]),
        "sigma": jnp.exp(theta["sigma"]),
        "gamma": jnp.exp(theta["gamma"]),
        "iota": jnp.exp(theta["iota"]),
        "sigmaSE": jnp.exp(theta["sigmaSE"]),
        "psi": jnp.exp(theta["psi"]),
        "cohort": jspecial.expit(theta["cohort"]),
        "amplitude": jspecial.expit(theta["amplitude"]),
        "rho": jspecial.expit(theta["rho"]),
        "S_0": S_0,
        "E_0": E_0,
        "I_0": I_0,
        "R_0": R_0,
    }
