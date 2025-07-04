from functools import partial
import os
import csv
import jax
import jax.numpy as jnp
import pandas as pd
from pypomp.pomp_class import Pomp
from pypomp.model_struct import RInit, RProc, DMeas, RMeas
import jax.scipy.special as jspecial
import numpy as np


def get_thetas(theta):
    gamma = jnp.exp(theta[0])
    m = jnp.exp(theta[1])
    rho = jnp.exp(theta[2])
    epsilon = jnp.exp(theta[3])
    omega = jnp.exp(theta[4])
    c = jspecial.expit(theta[5])
    beta_trend = theta[6] / 100
    sigma = jnp.exp(theta[7])
    tau = jnp.exp(theta[8])
    bs = theta[9:15]
    omegas = theta[15:21]
    k = 3
    delta = 0.02
    return (
        gamma,
        m,
        rho,
        epsilon,
        omega,
        c,
        beta_trend,
        sigma,
        tau,
        bs,
        omegas,
        k,
        delta,
    )


def transform_thetas(
    gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas
):
    return jnp.concatenate(
        [
            jnp.array(
                [
                    jnp.log(gamma),
                    jnp.log(m),
                    jnp.log(rho),
                    jnp.log(epsilon),
                    jnp.log(omega),
                    jspecial.logit(c),
                    beta_trend * 100,
                    jnp.log(sigma),
                    jnp.log(tau),
                ]
            ),
            bs,
            omegas,
        ]
    )


gamma = 20.8  # recovery rate
epsilon = 19.1  # rate of waning of immunity for severe infections
rho = 0  # rate of waning of immunity for inapparent infections
delta = 0.02  # baseline mortality rate
m = 0.06  # cholera mortality rate
c = jnp.array(1)  # fraction of infections that lead to severe infection
beta_trend = -0.00498  # slope of secular trend in transmission
bs = jnp.array([0.747, 6.38, -3.44, 4.23, 3.33, 4.55])  # seasonal transmission rates
sigma = 3.13  # 3.13 # 0.77 # environmental noise intensity
tau = 0.23  # measurement error s.d.
omega = jnp.exp(-4.5)
omegas = jnp.log(
    jnp.array([0.184, 0.0786, 0.0584, 0.00917, 0.000208, 0.0124])
)  # seasonal environmental reservoir parameters

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data/dacca")

dacca_path = os.path.join(data_dir, "dacca.csv")
covars_path = os.path.join(data_dir, "covars.csv")
covart_path = os.path.join(data_dir, "covart.csv")

with open(dacca_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    data = [(float(row[1]), float(row[2])) for row in reader]
    times, values = zip(*data)
    ys = pd.DataFrame(values, index=pd.Index(times), columns=pd.Index(["deaths"]))

with open(covart_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    covart_index = [float(row[1]) for row in reader]
    covart_index = jnp.array(covart_index)

with open(covars_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    covars_data = [[float(value) for value in row[1:]] for row in reader]
    covars = pd.DataFrame(covars_data, index=np.array(covart_index))

key = jax.random.key(111)
theta_names = (
    [
        "gamma",
        "m",
        "rho",
        "epsilon",
        "omega",
        "c",
        "beta_trend",
        "sigma",
        "tau",
    ]
    + [f"b{i}" for i in range(1, 7)]
    + [f"omega{i}" for i in range(1, 7)]
)
theta = dict(
    zip(
        theta_names,
        transform_thetas(
            gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas
        ).tolist(),
    )
)


@partial(RInit, t0=1891.0)
def rinit(theta_, key, covars, t0=None):
    S_0 = 0.621
    I_0 = 0.378
    Y_0 = 0
    R1_0 = 0.000843
    R2_0 = 0.000972
    R3_0 = 1.16e-07
    pop = covars[2]
    S = pop * S_0
    I = pop * I_0
    Y = pop * Y_0
    R1 = pop * R1_0
    R2 = pop * R2_0
    R3 = pop * R3_0
    Mn = 0
    count = 0
    return jnp.array([S, I, Y, Mn, R1, R2, R3, count])


def rproc(X_, theta_, key, covars, t, dt):
    S = X_[0]
    I = X_[1]
    Y = X_[2]
    deaths = X_[3]
    pts = X_[4:-1]
    count = X_[-1]
    trend = covars[0]
    dpopdt = covars[1]
    pop = covars[2]
    seas = covars[3:]
    (
        gamma,
        deltaI,
        rho,
        eps,
        omega,
        clin,
        beta_trend,
        sd_beta,
        tau,
        bs,
        omegas,
        nrstage,
        delta,
    ) = get_thetas(theta_)
    nrstage = 3
    clin = 1  # HARDCODED SEIR
    rho = 0  # HARDCODED INAPPARENT INFECTIONS
    std = jnp.sqrt(dt)

    neps = eps * nrstage  # rate
    passages = jnp.zeros(nrstage + 1)

    # Get current time step values
    beta = jnp.exp(beta_trend * trend + jnp.dot(bs, seas))
    omega = jnp.exp(jnp.dot(omegas, seas))

    subkey, key = jax.random.split(key)
    dw = jax.random.normal(subkey) * std

    effI = I / pop
    births = dpopdt + delta * pop
    passages = passages.at[0].set(gamma * I)
    ideaths = delta * I
    disease = deltaI * I
    ydeaths = delta * Y
    wanings = rho * Y

    rdeaths = pts * delta
    passages = passages.at[1:].set(pts * neps)

    infections = (omega + (beta + sd_beta * dw / dt) * effI) * S
    sdeaths = delta * S

    S += (births - infections - sdeaths + passages[nrstage] + wanings) * dt
    I += (clin * infections - disease - ideaths - passages[0]) * dt
    Y += ((1 - clin) * infections - ydeaths - wanings) * dt

    pts = pts + (passages[:-1] - passages[1:] - rdeaths) * dt

    deaths = deaths + disease * dt

    count = count + jnp.any(jnp.hstack([jnp.array([S, I, Y, deaths]), pts]) < 0)

    S = jnp.clip(S, 0)
    I = jnp.clip(I, 0)
    Y = jnp.clip(Y, 0)
    pts = jnp.clip(pts, 0)
    deaths = jnp.clip(deaths, 0)

    return jnp.hstack([jnp.array([S, I, Y, deaths]), pts, jnp.array([count])])


def dmeas_helper(y, deaths, v, tol, ltol):
    return jnp.logaddexp(
        jax.scipy.stats.norm.logpdf(y, loc=deaths, scale=v + tol), ltol
    )


def dmeas_helper_tol(y, deaths, v, tol, ltol):
    return jnp.array([ltol])


@DMeas
def dmeas(Y_, X_, theta_, covars=None, t=None):
    deaths = X_[3]
    count = X_[-1]
    tol = 1.0e-18
    ltol = jnp.log(tol)
    (gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas, k, delta) = (
        get_thetas(theta_)
    )
    v = tau * deaths
    # return jax.scipy.stats.norm.logpdf(y, loc=deaths, scale=v)
    return jax.lax.cond(
        jnp.logical_or(
            (1 - jnp.isfinite(v)).astype(bool), count > 0
        ),  # if Y < 0 then count violation
        dmeas_helper_tol,
        dmeas_helper,
        *(Y_, deaths, v, tol, ltol),
    )


@partial(RMeas, ydim=1)
def rmeas(X_, theta_, key, covars=None, t=None):
    deaths = X_[3]
    (gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas, k, delta) = (
        get_thetas(theta_)
    )
    v = tau * deaths
    return jax.random.normal(key) * v + deaths


def dacca(dt: float | None = 1 / 240, nstep: int | None = None) -> Pomp:
    """
    Creates a POMP model for the Dacca measles data.

    This function constructs a Partially Observed Markov Process (POMP) model
    for the Dacca measles dataset. The model includes a stochastic process for
    the underlying disease dynamics and a measurement model for observed deaths.

    Parameters
    ----------
    dt : float, optional
        Time step size for the process model. Determines the number of sub-steps per observation interval for the process model.
    nstep : int, optional
        Number of sub-steps per observation interval for the process model.
        If None, uses Euler discretization with the specified step size. nstep and dt cannot both be not None.

    Returns
    -------
    Pomp
        A POMP model object representing the Dacca cholera model.
    """
    from pypomp.dacca import rproc

    if nstep is not None and dt is not None:
        raise ValueError("Cannot specify both dt and nstep")

    if nstep is not None:
        rproc = RProc(rproc, step_type="fixedstep", nstep=nstep)
    else:
        rproc = RProc(rproc, step_type="euler", dt=dt, accumvars=(3,))

    dacca_obj = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys,
        theta=theta,
        covars=covars,
    )
    return dacca_obj
