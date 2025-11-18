"""He10 model without alpha or deaths"""

import jax.numpy as jnp
import jax
import jax.scipy.special as jspecial
from pypomp.random.poissoninvf import rpoisson
from pypomp.random.binominvf import rbinom
from pypomp.random.gammainvf import rgamma


param_names = (
    "R0",  # 0
    "sigma",  # 1
    "gamma",  # 2
    "iota",  # 3
    "rho",  # 4
    "sigmaSE",  # 5
    "psi",  # 6
    "cohort",  # 7
    "amplitude",  # 8
    "S_0",  # 9
    "E_0",  # 10
    "I_0",  # 11
    "R_0",  # 12
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
    iota = theta_["iota"]
    sigmaSE = theta_["sigmaSE"]
    cohort = theta_["cohort"]
    amplitude = theta_["amplitude"]
    pop = covars["pop"]
    birthrate = covars["birthrate"]

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
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma) * dt)) / dt

    # expected force of infection
    foi = beta * (I + iota) / pop

    # white noise (extrademographic stochasticity)
    keys = jax.random.split(key, 3)
    # dw = jax.random.gamma(keys[0], dt / sigmaSE**2) * sigmaSE**2
    dw = rgamma(keys[0], dt / sigmaSE**2) * sigmaSE**2

    rate = jnp.array([foi * dw / dt, sigma, gamma])

    # Poisson births
    # births = jax.random.poisson(keys[1], br * dt)
    births = rpoisson(keys[1], br * dt)

    # transitions between classes
    # rt_final = jnp.zeros((3, 2))

    populations = jnp.array([S, E, I])

    p0_values = jnp.exp(-rate * dt)

    # rt_final = rt_final.at[:, 0].set(1 - p0_values).at[:, 1].set(p0_values)

    # transitions = jax.random.multinomial(keys[2], populations, rt_final)
    transitions = rbinom(keys[2], populations, p0_values)

    trans_S = transitions[0]
    trans_E = transitions[1]
    trans_I = transitions[2]

    S = S + births - trans_S
    E = E + trans_S - trans_E
    I = I + trans_E - trans_I
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + trans_I
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
