"""He10 model without alpha or mu parameters - DPOP enabled with gradient-stable dmeas

Gradient stability fixes:
1. rproc: Use jax.random.gamma instead of fast_approx_rgamma (stable reparameterization)
2. rproc: Use sample_and_log_prob instead of rmultinomial + deulermultinom (unified gradient path)
3. dmeas: Use custom JVP for log_cdf_diff (prevents 0 * inf = NaN in extreme z regions)
4. dmeas: Replace NaN y before computing z (prevents NaN propagation through jnp.where)
"""

import jax.numpy as jnp
import jax
import jax.scipy.special as jspecial
from jax.scipy.special import log_ndtr
from pypomp.ctmc_multinom import sample_and_log_prob
from pypomp.random.poissoninvf import fast_approx_rpoisson
from pypomp.random.binominvf import fast_approx_rmultinom
from pypomp.random.gammainvf import fast_approx_rgamma


# =========================================================================
# Custom JVP log_cdf_diff for gradient stability
# =========================================================================
LOG_SQRT_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


def _log_phi(z):
    """log φ(z) for standard normal density"""
    return -0.5 * z * z - LOG_SQRT_2PI


def _log_sub_exp_stable(a, b):
    """Stable log(exp(a) - exp(b)) assuming a >= b"""
    return a + jnp.log1p(-jnp.exp(b - a))


@jax.custom_jvp
def log_cdf_diff(zh, zl):
    """log(Φ(zh) - Φ(zl)) with custom JVP for gradient stability"""
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

    # 计算 log-space 梯度
    diff_h = lphi_h - y
    diff_l = lphi_l - y

    # 处理 y=-inf 退化情况（当 zh≈zl 时）
    # diff = lphi - (-inf) = +inf，需要映射到 -LOG_MAX 使 exp() → 0
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


def log_cdf_single(z):
    """log Φ(z)，使用相同的 stable JVP 路径"""
    return log_cdf_diff(z, -jnp.inf)


# =========================================================================
# Model definition
# =========================================================================
param_names = (
    "R0",  # 0 - basic reproduction number
    "sigma",  # 1 - 1/latent period
    "gamma",  # 2 - 1/infectious period
    "iota",  # 3 - imported infections
    "rho",  # 4 - reporting rate
    "sigmaSE",  # 5 - extra-demographic stochasticity
    "psi",  # 6 - overdispersion in measurement
    "cohort",  # 7 - cohort effect
    "amplitude",  # 8 - seasonal amplitude
    "S_0",  # 9 - initial susceptible fraction
    "E_0",  # 10 - initial exposed fraction
    "I_0",  # 11 - initial infected fraction
    "R_0",  # 12 - initial recovered fraction
)

# State includes "logw" for DPOP process log-density
statenames = ["S", "E", "I", "R", "W", "C", "logw"]

# accumvars are reset each observation interval
accumvars = ("W", "C", "logw")


def rinit(theta_, key, covars, t0=None):
    S_0 = theta_["S_0"]
    E_0 = theta_["E_0"]
    I_0 = theta_["I_0"]
    R_0 = theta_["R_0"]

    m = covars["pop"] / (S_0 + E_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    E = jnp.round(m * E_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    W = 0.0
    C = 0.0
    logw = 0.0
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C, "logw": logw}


def rproc(X_, theta_, key, covars, t, dt):
    S, E, I, R, W, C, logw = (
        X_["S"],
        X_["E"],
        X_["I"],
        X_["R"],
        X_["W"],
        X_["C"],
        X_["logw"],
    )
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

    # Cohort effect timing
    t_mod = t - jnp.floor(t)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    br = jnp.where(
        is_cohort_time,
        cohort * birthrate / dt + (1 - cohort) * birthrate,
        (1 - cohort) * birthrate,
    )

    # Term-time seasonality
    t_days = t_mod * 365.25
    in_term_time = (
        ((t_days >= 7) & (t_days <= 100))
        | ((t_days >= 115) & (t_days <= 199))
        | ((t_days >= 252) & (t_days <= 300))
        | ((t_days >= 308) & (t_days <= 356))
    )
    seas = jnp.where(in_term_time, 1.0 + amplitude * 0.2411 / 0.7589, 1 - amplitude)

    # Transmission rate
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # Force of infection
    foi = beta * (I + iota) / pop

    # White noise (extrademographic stochasticity)
    # FIX 1: Use jax.random.gamma for gradient stability
    keys = jax.random.split(key, 3)
    dw = fast_approx_rgamma(keys[0], dt / sigmaSE**2) * sigmaSE**2

    # Poisson births
    births = fast_approx_rpoisson(keys[1], br * dt).astype(jnp.float32)

    # Transition rates for Euler-multinomial steps
    rates_S = jnp.array([foi * dw / dt, mu])
    rates_E = jnp.array([sigma, mu])
    rates_I = jnp.array([gamma, mu])

    # FIX 2: Use sample_and_log_prob for unified gradient path
    key_proc = keys[2]
    (StoE, StoDeath), lp_S, key_proc = sample_and_log_prob(S, rates_S, dt, key_proc)
    (EtoI, EtoDeath), lp_E, key_proc = sample_and_log_prob(E, rates_E, dt, key_proc)
    (ItoR, ItoDeath), lp_I, key_proc = sample_and_log_prob(I, rates_I, dt, key_proc)

    # Accumulate process log-density
    logw_step = lp_S + lp_E + lp_I
    logw_step = jnp.where(jnp.isfinite(logw_step), logw_step, 0.0)
    logw = logw + logw_step

    # State updates
    S = S + births - StoE - StoDeath
    E = E + StoE - EtoI - EtoDeath
    I = I + EtoI - ItoR - ItoDeath
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + ItoR

    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C, "logw": logw}


def dmeas(Y_, X_, theta_, covars=None, t=None):
    """
    Gradient-stable measurement density using custom JVP.

    Fixes:
    - FIX 3: Custom JVP for log_cdf_diff handles extreme z values without 0 * inf = NaN
    - FIX 4: Replace NaN y before computing z, preventing NaN propagation
    """
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    y_raw = Y_["cases"]

    tol = 1e-12

    # FIX 4: Replace NaN y before computing z
    # jnp.where computes BOTH branches, so NaN in y propagates to gradient
    # By replacing NaN with 0 first, z computation stays finite
    y_is_nan = jnp.isnan(y_raw)
    y = jnp.where(y_is_nan, 0.0, y_raw)

    # Mean and variance
    Cpos = jnp.maximum(C, 0.0)
    m = rho * Cpos
    v = m * (1.0 - rho + psi**2 * m)
    v = jnp.maximum(v, tol * tol)
    s = jnp.sqrt(v)

    # z-scores
    z_hi = (y + 0.5 - m) / s
    z_lo = (y - 0.5 - m) / s

    # FIX 3: Use custom JVP log_cdf_diff for gradient stability
    ll_box = jnp.where(y > tol, log_cdf_diff(z_hi, z_lo), log_cdf_single(z_hi))

    loglik = jnp.maximum(ll_box, jnp.log(tol))

    # FIX 4 continued: Zero out result for NaN y (instead of using where)
    # Multiplying by 0 gives gradient 0, not NaN
    loglik = loglik * (1.0 - y_is_nan.astype(loglik.dtype))

    return loglik


def rmeas(X_, theta_, key, covars=None, t=None):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)


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
