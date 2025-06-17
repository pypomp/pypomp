"""He10 model without alpha or mu parameters"""

import jax.numpy as jnp
import jax


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
    S_0, E_0, I_0, R_0 = jnp.exp(theta_[9:]) / jnp.sum(jnp.exp(theta_[9:]))
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
    R0 = jnp.exp(theta_[0])
    sigma = jnp.exp(theta_[1])
    gamma = jnp.exp(theta_[2])
    iota = jnp.exp(theta_[3])
    sigmaSE = jnp.exp(theta_[5])
    cohort = theta_[7]
    amplitude = theta_[8]
    pop = covars[0]
    birthrate = covars[1]
    mu = 0.02
    br = jax.lax.cond(
        jnp.squeeze(jnp.abs(t - jnp.floor(t) - 251.0 / 365.0)) < 0.5 * dt,
        lambda cohort, birthrate, dt: (
            cohort * birthrate / dt + (1 - cohort) * birthrate
        ),
        lambda cohort, birthrate, dt: (1 - cohort) * birthrate,
        *(cohort, birthrate, dt),
    )

    # term-time seasonality
    t = (t - jnp.floor(t)) * 365.25
    seas = jax.lax.cond(
        jnp.squeeze(
            ((t >= 7) & (t <= 100))
            | ((t >= 115) & (t <= 199))
            | ((t >= 252) & (t <= 300))
            | ((t >= 308) & (t <= 356))
        ),
        lambda amplitude: 1.0 + amplitude * 0.2411 / 0.7589,
        lambda amplitude: 1 - amplitude,
        amplitude,
    )

    # transmission rate
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt

    # expected force of infection
    foi = beta * (I + iota) / pop

    # white noise (extrademographic stochasticity)
    key, subkey = jax.random.split(key)
    dw = jax.random.gamma(subkey, dt / sigmaSE**2) * sigmaSE**2
    # dw = jnp.exp(
    #     jax.random.loggamma(subkey, jnp.exp(jnp.log(dt) - 2 * jnp.log(sigmaSE)))
    #     + 2 * jnp.log(sigmaSE)
    # )

    # ir = jnp.exp(jnp.log(foi) + jnp.log(dw) - jnp.log(dt))
    # rate = jnp.array([ir, mu, sigma, mu, gamma, mu])
    rate = jnp.array([foi * dw / dt, mu, sigma, mu, gamma, mu])

    # Poisson births
    key, subkey = jax.random.split(key)
    births = jax.random.poisson(subkey, br * dt)

    # transitions between classes
    # TODO use a loop for this
    key, subkey = jax.random.split(key)
    p0 = jnp.exp(-(rate[0] + rate[1]) * dt)
    rt = (rate[0:2]) / (rate[0] + rate[1]) * (1 - p0)
    rt = jnp.concatenate([rt.reshape((2,)), p0.reshape((1,))])
    trans_S = jax.random.multinomial(subkey, S, rt)

    lastkey, subkey = jax.random.split(key)
    p0 = jnp.exp(-(rate[2] + rate[3]) * dt)
    rt = (rate[2:4]) / (rate[2] + rate[3]) * (1 - p0)
    rt = jnp.concatenate([rt.reshape((2,)), p0.reshape((1,))])
    trans_E = jax.random.multinomial(subkey, E, rt)

    p0 = jnp.exp(-(rate[4] + rate[5]) * dt)
    rt = (rate[4:6]) / (rate[4] + rate[5]) * (1 - p0)
    rt = jnp.concatenate([rt.reshape((2,)), p0.reshape((1,))])
    trans_I = jax.random.multinomial(lastkey, I, rt)

    S = S + births - trans_S[0] - trans_S[1]
    E = E + trans_S[0] - trans_E[0] - trans_E[1]
    I = I + trans_E[0] - trans_I[0] - trans_I[1]
    R = pop - S - E - I
    W = W + (dw - dt) / sigmaSE
    C = C + trans_I[0]
    # jax.debug.print("dt: {x}", x=dt)
    # jax.debug.print("dw: {x}", x=dw)
    # jax.debug.print("foi: {x}", x=foi)
    # jax.debug.print("dw/dt: {x}", x=dw / dt)
    # jax.debug.print("rate[0]: {x}", x=rate[0])
    return jnp.array([S, E, I, R, W, C])


def dmeas(Y_, X_, theta_, covars=None, t=None):
    rho = theta_[4]
    psi = jnp.exp(theta_[6])
    C = X_[5]
    tol = 1.0e-18

    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    sqrt_v_tol = jnp.sqrt(v) + tol

    upper_cdf = jax.scipy.stats.norm.cdf(Y_ + 0.5, m, sqrt_v_tol)

    lik = (
        jnp.where(
            Y_ > tol,
            upper_cdf - jax.scipy.stats.norm.cdf(Y_ - 0.5, m, sqrt_v_tol),
            upper_cdf,
        )
        + tol
    )

    lik = jnp.where(C < 0, 0.0, lik)
    lik = jnp.where(jnp.isnan(Y_), 1.0, lik)
    return jnp.log(lik)


def rmeas(X_, theta_, key, covars=None, t=None):
    rho = theta_[4]
    psi = jnp.exp(theta_[6])
    C = X_[5]
    m = rho * C
    v = m * (1.0 - rho + psi**2 * m)
    tol = 1.0e-18  # 1.0e-18 in He10 model; 0.0 is 'correct'
    cases = jax.random.normal(key) * (jnp.sqrt(v) + tol) + m
    return jnp.where(cases > 0.0, jnp.round(cases), 0.0)
