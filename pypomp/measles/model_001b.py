"""He10 model without alpha or mu"""

import jax.numpy as jnp
import jax.random


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


def rinit(theta_, key, covars=None):
    S_0, E_0, I_0, R_0 = jnp.exp(theta_[9:]) / jnp.sum(jnp.exp(theta_[9:]))
    m = covars[0, 0] / (S_0 + E_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    E = jnp.round(m * E_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    W = 0
    C = 0
    return jnp.array([S, E, I, R, W, C])


def rproc(X_, theta_, key, covars, t):
    S, E, I, R, W, C, t = X_
    R0 = theta_[0]
    sigma = theta_[1]
    gamma = theta_[2]
    iota = theta_[3]
    sigmaSE = theta_[5]
    cohort = theta_[7]
    amplitude = theta_[8]
    pop = covars[t, 0]
    birthrate = covars[t, 1]
    dt = 1 / 365.25
    mu = 0.02
    if jnp.abs(t - jnp.floor(t) - 251.0 / 365.0) < 0.5 * dt:
        br = cohort * birthrate / dt + (1 - cohort) * birthrate
    else:
        br = (1.0 - cohort) * birthrate

    # term-time seasonality
    t = (t - jnp.floor(t)) * 365.25
    if (
        (t >= 7) & (t <= 100)
        | (t >= 115) & (t <= 199)
        | (t >= 252) & (t <= 300)
        | (t >= 308) & (t <= 356)
    ):
        seas = 1.0 + amplitude * 0.2411 / 0.7589
    else:
        seas = 1.0 - amplitude

    # transmission rate
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # expected force of infection
    foi = beta * (I + iota) / pop

    # white noise (extrademographic stochasticity)
    key, subkey = jax.random.key(key)
    dw = jnp.random.gamma(sigmaSE, dt, key=subkey)

    rate = jnp.array([foi * dw / dt, mu, sigma, mu, gamma, mu])

    # Poisson births
    key_last, subkey = jax.random.key(key)
    births = jax.random.poisson(subkey, br * dt)

    # transitions between classes
    trans = jax.random.poisson(key_last, rate * dt)  # TODO: fix this

    S = S + births - trans[0] - trans[1]
    E = E + trans[0] - trans[2] - trans[3]
    I = I + trans[2] - trans[4] - trans[5]
    R = pop - S - E - I
    W = W + (dw - dt) / theta_[6]
    C = C + trans[4]
    return jnp.array([S, E, I, R, W, C])


def dmeas(Y_, X_, theta_, covars=None, t=None):
    C = X_[5]
    m = theta_[4] * C
    v = m * (1.0 - theta_[4] + theta_[6] ** 2 * m)
    tol = 1.0e-18
    if jnp.isnan(Y_):
        lik = 1.0
    else:
        if C < 0:
            lik = 0.0
        else:
            if Y_ > tol:
                lik = (
                    jax.scipy.stats.norm.cdf(Y_ + 0.5, m, jnp.sqrt(v) + tol)
                    - jax.scipy.stats.norm.cdf(Y_ - 0.5, m, jnp.sqrt(v) + tol)
                    + tol
                )
            else:
                lik = jax.scipy.stats.norm.cdf(Y_ + 0.5, m, jnp.sqrt(v) + tol) + tol
    return jnp.log(lik)


def rmeas(X_, theta_, key, covars=None, t=None):
    m = theta_[4] * X_[5]
    v = m * (1.0 - theta_[4] + theta_[6] ** 2 * m)
    tol = 1.0e-18  # 1.0e-18 in He10 model; 0.0 is 'correct'
    cases = jax.random.normal(key, (m, jnp.sqrt(v) + tol))
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)
