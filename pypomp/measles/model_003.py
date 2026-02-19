"""
He10 model with continuous process model.
"""

import jax.numpy as jnp
import jax
import jax.scipy.special as jspecial
from pypomp.random.gammainvf import fast_approx_rgamma

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


def rinit(theta_, key, covars, t0=None):
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


def rproc(X_, theta_, key, covars, t, dt):
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
    safe_birth_mean = jnp.maximum(birth_mean, 1e-8)
    births = jnp.maximum(birth_mean + jnp.sqrt(safe_birth_mean) * birth_noise, 0.0)

    # Rates
    rate_inf = foi * (dw / dt)  # effective infection rate

    flux_noises = all_noise[1:7]

    rates = jnp.array([rate_inf, mu, sigma, mu, gamma, mu])
    states = jnp.array([S, S, E, E, I, I])

    mu_fluxes = rates * states * dt

    # Use a 1e-8 floor to avoid gradient numerical instability
    safe_mu_fluxes = jnp.maximum(mu_fluxes, 1e-8)
    fluxes = mu_fluxes + jnp.sqrt(safe_mu_fluxes) * flux_noises

    flux_SE, flux_SD, flux_EI, flux_ED, flux_IR, flux_ID = fluxes

    S_new = jnp.maximum(S + births - flux_SE - flux_SD, 0.0)
    E_new = jnp.maximum(E + flux_SE - flux_EI - flux_ED, 0.0)
    I_new = jnp.maximum(I + flux_EI - flux_IR - flux_ID, 0.0)
    R_new = jnp.maximum(pop - S_new - E_new - I_new, 0.0)

    W_new = W + (dw - dt) / sigmaSE
    C_new = jnp.maximum(C + flux_EI, 0.0)

    return {"S": S_new, "E": E_new, "I": I_new, "R": R_new, "W": W_new, "C": C_new}


def dmeas(Y_, X_, theta_, covars=None, t=None):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]

    y = Y_["cases"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18
    scale = jnp.sqrt(jnp.maximum(v, tol))
    loglik = jax.scipy.stats.norm.logpdf(y, loc=m, scale=scale)
    loglik = jnp.where(jnp.isnan(y), 0.0, loglik)
    loglik = jnp.where(C < 0, -jnp.inf, loglik)

    return loglik


def rmeas(X_, theta_, key, covars=None, t=None):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18

    cases = jax.random.normal(key) * (jnp.sqrt(jnp.maximum(v, 0.0)) + tol) + m

    return jnp.where(cases > 0.0, cases, 0.0)


def to_est(theta):
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


def from_est(theta):
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
