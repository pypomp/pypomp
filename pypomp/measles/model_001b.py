"""He10 model without alpha or mu parameters"""

import jax.numpy as jnp
import jax
from pypomp.util import expit
from pypomp.fast_random import (
    fast_approx_multinomial,
    fast_approx_poisson,
    fast_approx_gamma,
)


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
    exp_theta_9_13 = jnp.exp(
        jnp.array([theta_["S_0"], theta_["E_0"], theta_["I_0"], theta_["R_0"]])
    )
    S_0, E_0, I_0, R_0 = exp_theta_9_13 / jnp.sum(exp_theta_9_13)
    m = covars[0] / (S_0 + E_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    E = jnp.round(m * E_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    W = 0
    C = 0
    return {"S": S, "E": E, "I": I, "R": R, "W": W, "C": C}


def rproc(X_, theta_, key, covars, t, dt):
    S, E, I, R, W, C = X_["S"], X_["E"], X_["I"], X_["R"], X_["W"], X_["C"]
    exp_theta = jnp.exp(
        jnp.array(
            [
                theta_["R0"],
                theta_["sigma"],
                theta_["gamma"],
                theta_["iota"],
                theta_["sigmaSE"],
            ]
        )
    )
    R0 = exp_theta[0]
    sigma = exp_theta[1]
    gamma = exp_theta[2]
    iota = exp_theta[3]
    sigmaSE = exp_theta[4]
    cohort = expit(theta_["cohort"])
    amplitude = expit(theta_["amplitude"])
    pop = covars[0]
    birthrate = covars[1]
    mu = 0.02

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
    # dw = jax.random.gamma(keys[0], dt / sigmaSE**2) * sigmaSE**2
    dw = fast_approx_gamma(keys[0], dt / sigmaSE**2, max_rejections=1) * sigmaSE**2

    rate = jnp.array([foi * dw / dt, mu, sigma, mu, gamma, mu])

    # Poisson births
    # births = jax.random.poisson(keys[1], br * dt)
    births = fast_approx_poisson(keys[1], br * dt, max_rejections=1)

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

    # transitions = jax.random.multinomial(keys[2], populations, rt_final)
    transitions = fast_approx_multinomial(
        keys[2], populations, rt_final, max_rejections=1
    )

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
    rho = expit(theta_["rho"])
    psi = jnp.exp(theta_["psi"])
    C = X_["C"]
    tol = 1.0e-18

    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    sqrt_v_tol = jnp.sqrt(v) + tol

    upper_cdf = jax.scipy.stats.norm.cdf(Y_ + 0.5, m, sqrt_v_tol)
    lower_cdf = jax.scipy.stats.norm.cdf(Y_ - 0.5, m, sqrt_v_tol)

    lik = (
        jnp.where(
            Y_ > tol,
            upper_cdf - lower_cdf,
            upper_cdf,
        )
        + tol
    )

    lik = jnp.where(C < 0, 0.0, lik)
    lik = jnp.where(jnp.isnan(Y_), 1.0, lik)
    return jnp.log(lik)


def rmeas(X_, theta_, key, covars=None, t=None):
    rho = expit(theta_["rho"])
    psi = jnp.exp(theta_["psi"])
    C = X_["C"]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18  # 1.0e-18 in He10 model; 0.0 is 'correct'
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)
