"""He10 model without alpha or deaths"""

import jax.numpy as jnp
import jax
from pypomp.util import expit
from pypomp.fast_random import (
    fast_approx_multinomial,
    fast_approx_poisson,
    fast_approx_gamma,
    fast_approx_binomial,
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


def rinit(theta_, key, covars, t0=None):
    exp_theta_9_13 = jnp.exp(theta_[9:])
    S_0, E_0, I_0, R_0 = exp_theta_9_13 / jnp.sum(exp_theta_9_13)
    m = covars[0] / (S_0 + E_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    E = jnp.round(m * E_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    W = 0
    C = 0
    return jnp.array([S, E, I, R, W, C])


def rproc(X_, theta_, key, covars, t, dt):
    S, E, I, R, W, C = X_
    exp_theta = jnp.exp(theta_[jnp.array([0, 1, 2, 3, 5])])
    R0 = exp_theta[0]
    sigma = exp_theta[1]
    gamma = exp_theta[2]
    iota = exp_theta[3]
    sigmaSE = exp_theta[4]
    cohort = expit(theta_[7])
    amplitude = expit(theta_[8])
    pop = covars[0]
    birthrate = covars[1]

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
    dw = fast_approx_gamma(keys[0], dt / sigmaSE**2, max_rejections=2) * sigmaSE**2

    rate = jnp.array([foi * dw / dt, sigma, gamma])

    # Poisson births
    # births = jax.random.poisson(keys[1], br * dt)
    births = fast_approx_poisson(keys[1], br * dt, max_rejections=2)

    # transitions between classes
    # rt_final = jnp.zeros((3, 2))

    populations = jnp.array([S, E, I])

    p0_values = jnp.exp(-rate * dt)

    # rt_final = rt_final.at[:, 0].set(1 - p0_values).at[:, 1].set(p0_values)

    # transitions = jax.random.multinomial(keys[2], populations, rt_final)
    transitions = fast_approx_binomial(
        keys[2], populations, 1 - p0_values, max_rejections=2
    )

    trans_S = transitions[0]
    trans_E = transitions[1]
    trans_I = transitions[2]

    S = S + births - trans_S
    E = E + trans_S - trans_E
    I = I + trans_E - trans_I
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + trans_I
    return jnp.array([S, E, I, R, W, C])


def dmeas(Y_, X_, theta_, covars=None, t=None):
    rho = expit(theta_[4])
    psi = jnp.exp(theta_[6])
    C = X_[5]
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
    rho = expit(theta_[4])
    psi = jnp.exp(theta_[6])
    C = X_[5]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18  # 1.0e-18 in He10 model; 0.0 is 'correct'
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)
