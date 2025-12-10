"""
He10 model without alpha or mu parameters. iota varies linearly with population size according to iota = exp(iota1 + iota2 * log(pop)). iota1 and iota2 are intended to be shared parameters in a panel model.

Parameters:
- R0: Basic reproduction number
- sigma: Rate of transition from susceptible to exposed
- gamma: Rate of transition from exposed to infectious
- iota1: Baseline imported cases
- iota2: Rate at which imported cases increase with population size
- sigmaSE: Rate of stochastic extrademographic variation
- cohort: Cohort effect
- amplitude: Seasonality amplitude
- rho: Reporting probability
- psi: Reporting error over-dispersion
- S_0: Initial susceptible population proportion
- E_0: Initial exposed population proportion
- I_0: Initial infectious population proportion
- R_0: Initial recovered population proportion
"""

import jax.numpy as jnp
import jax
import jax.scipy.special as jspecial
from pypomp.random.poissoninvf import fast_approx_rpoisson
from pypomp.random.binominvf import fast_approx_rmultinom
from pypomp.random.gammainvf import rgamma


param_names = (
    "R0",  # 0
    "sigma",  # 1
    "gamma",  # 2
    "iota1",  # 3
    "iota2",  # 4
    "rho",  # 5
    "sigmaSE",  # 6
    "psi",  # 7
    "cohort",  # 8
    "amplitude",  # 9
    "S_0",  # 10
    "E_0",  # 11
    "I_0",  # 12
    "R_0",  # 13
)

statenames = ["S", "E", "I", "R", "W", "C"]


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
    W = 0
    C = 0
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


def rproc(X_, theta_, key, covars, t, dt):
    S, E, I, R, W, C = X_["S"], X_["E"], X_["I"], X_["R"], X_["W"], X_["C"]
    R0 = theta_["R0"]
    sigma = theta_["sigma"]
    gamma = theta_["gamma"]
    iota1 = theta_["iota1"]
    iota2 = theta_["iota2"]
    sigmaSE = theta_["sigmaSE"]
    cohort = theta_["cohort"]
    amplitude = theta_["amplitude"]
    pop = covars["pop"]
    birthrate = covars["birthrate"]
    mu = 0.02

    iota = jnp.exp(
        iota1 + iota2 * jnp.log(pop)
    )  # TODO: change pop to 1950 pop, maybe also standardize pop

    t_mod = t - jnp.floor(t)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    br = jnp.where(
        is_cohort_time,
        cohort * birthrate / dt + (1 - cohort) * birthrate,
        (1 - cohort) * birthrate,
    )

    # term-time seasonality
    t_days = t_mod * 365.25
    in_term_time = (
        ((t_days >= 7) & (t_days <= 100))
        | ((t_days >= 115) & (t_days <= 199))
        | ((t_days >= 252) & (t_days <= 300))
        | ((t_days >= 308) & (t_days <= 356))
    )
    seas = jnp.where(in_term_time, 1.0 + amplitude * 0.2411 / 0.7589, 1 - amplitude)

    # transmission rate
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # expected force of infection
    foi = beta * (I + iota) / pop

    # white noise (extrademographic stochasticity)
    keys = jax.random.split(key, 3)
    dw = rgamma(keys[0], dt / sigmaSE**2) * sigmaSE**2

    rate = jnp.array([foi * dw / dt, mu, sigma, mu, gamma, mu])

    # Poisson births
    births = fast_approx_rpoisson(keys[1], br * dt)

    # transitions between classes
    rt_final = jnp.zeros((3, 3))

    rate_pairs = jnp.array([[rate[0], rate[1]], [rate[2], rate[3]], [rate[4], rate[5]]])
    populations = jnp.array([S, E, I])

    rate_sums = jnp.sum(rate_pairs, axis=1)
    p0_values = jnp.exp(-rate_sums * dt)

    rt_final = (
        rt_final.at[:, 0:2]
        .set(jnp.einsum("ij,i,i->ij", rate_pairs, 1 / rate_sums, 1 - p0_values))
        .at[:, 2]
        .set(p0_values)
    )

    transitions = fast_approx_rmultinom(keys[2], populations, rt_final)

    trans_S = transitions[0]
    trans_E = transitions[1]
    trans_I = transitions[2]

    S = S + births - trans_S[0] - trans_S[1]
    E = E + trans_S[0] - trans_E[0] - trans_E[1]
    I = I + trans_E[0] - trans_I[0] - trans_I[1]
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + trans_I[0]
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


def dmeas(Y_, X_, theta_, covars=None, t=None):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    tol = 1.0e-18

    y = Y_["cases"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    sqrt_v_tol = jnp.sqrt(v) + tol

    upper_cdf = jax.scipy.stats.norm.cdf(y + 0.5, m, sqrt_v_tol)
    lower_cdf = jax.scipy.stats.norm.cdf(y - 0.5, m, sqrt_v_tol)

    lik = (
        jnp.where(
            y > tol,
            upper_cdf - lower_cdf,
            upper_cdf,
        )
        + tol
    )

    lik = jnp.where(C < 0, 0.0, lik)
    lik = jnp.where(jnp.isnan(y), 1.0, lik)
    return jnp.log(lik)


def rmeas(X_, theta_, key, covars=None, t=None):
    rho = theta_["rho"]
    psi = theta_["psi"]
    C = X_["C"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18  # 1.0e-18 in He10 model; 0.0 is 'correct'
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)


def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    SEIR_0 = jnp.array([theta["S_0"], theta["E_0"], theta["I_0"], theta["R_0"]])
    S_0, E_0, I_0, R_0 = jnp.log(SEIR_0 / jnp.sum(SEIR_0))
    return {
        "R0": jnp.log(theta["R0"]),
        "sigma": jnp.log(theta["sigma"]),
        "gamma": jnp.log(theta["gamma"]),
        "iota1": theta["iota1"],
        "iota2": theta["iota2"],
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
        "iota1": theta["iota1"],
        "iota2": theta["iota2"],
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
