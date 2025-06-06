from functools import partial
import os
import csv
import jax
import jax.numpy as jnp
from pypomp.pomp_class import Pomp
from pypomp.model_struct import RInit
from pypomp.model_struct import RProc
from pypomp.model_struct import DMeas

import jax.scipy.special as jspecial


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
    omegas = theta[15:]
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
    ys = [float(row[2]) for row in reader]
    ys = jnp.array(ys)

with open(covars_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    covars_data = [[float(value) for value in row[1:]] for row in reader]
    covars_data = jnp.array(covars_data)

with open(covart_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    covart_index = [float(row[1]) for row in reader]
    covart_index = jnp.array(covart_index)

target_index = jnp.array([1891 + i * (1 / 240) for i in range(12037)])
interpolated_covars = []
for col in range(covars_data.shape[1]):
    interpolated_column = jnp.interp(target_index, covart_index, covars_data[:, col])
    interpolated_covars.append(interpolated_column)
    covars = jnp.array(interpolated_covars).T

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
ys = ys
covars = covars


@partial(RInit, t0=0)
def rinit(theta_, key, covars, t0=None):
    S_0 = 0.621
    I_0 = 0.378
    Y_0 = 0
    R1_0 = 0.000843
    R2_0 = 0.000972
    R3_0 = 1.16e-07
    pop = covars[0, 2]
    S = pop * S_0
    I = pop * I_0
    Y = pop * Y_0
    R1 = pop * R1_0
    R2 = pop * R2_0
    R3 = pop * R3_0
    Mn = 0
    t = 0
    count = 0
    return jnp.array([S, I, Y, Mn, R1, R2, R3, t, count])


def rproc_step(
    i,
    loop_vars,
    trends,
    dpopdts,
    pops,
    seass,
    std,
    dt,
    passages,
    nrstage,
    neps,
    rho,
    sd_beta,
    delta,
    deltaI,
    clin,
):
    S, I, Y, deaths, pts, t, count, key = loop_vars
    trend = trends[t]
    dpopdt = dpopdts[t]
    pop = pops[t]
    seas = seass[t]
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

    rdeaths = jnp.zeros(nrstage)

    rdeaths = pts * delta
    passages = passages.at[1:].set(pts * neps)

    infections = (omega + (beta + sd_beta * dw / dt) * effI) * S
    sdeaths = delta * S

    S += (births - infections - sdeaths + passages[nrstage] + wanings) * dt
    I += (clin * infections - disease - ideaths - passages[0]) * dt
    Y += ((1 - clin) * infections - ydeaths - wanings) * dt

    pts = pts + (passages[:-1] - passages[1:] - rdeaths) * dt

    deaths += disease * dt

    count += jnp.any(jnp.hstack([jnp.array([S, I, Y, deaths]), pts]) < 0)

    S = jnp.clip(S, 0)
    I = jnp.clip(I, 0)
    Y = jnp.clip(Y, 0)
    pts = jnp.clip(pts, 0)
    deaths = jnp.clip(deaths, 0)

    t += 1
    return S, I, Y, deaths, pts, t, count, key


@RProc
def rproc(X_, theta_, key, covars, t=None, dt=None):
    S = X_[0]
    I = X_[1]  # noqa
    Y = X_[2]
    deaths = X_[3]
    pts = X_[4:-2]
    t = X_[-2]
    count = X_[-1]
    t = t.astype(int)
    trends = covars[:, 0]
    dpopdts = covars[:, 1]
    pops = covars[:, 2]
    seass = covars[:, 3:]
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
    dt = 1 / 240
    deaths = 0
    nrstage = 3
    clin = 1  # HARDCODED SEIR
    rho = 0  # HARDCODED INAPPARENT INFECTIONS
    std = jnp.sqrt(dt)  # onp.sqrt(onp.sqrt(dt))

    neps = eps * nrstage  # rate
    # rdeaths = jnp.zeros(nrstage)  # the number of death in R1, R2, R3
    passages = jnp.zeros(nrstage + 1)

    rproc_step2 = partial(
        rproc_step,
        trends=trends,
        dpopdts=dpopdts,
        pops=pops,
        seass=seass,
        std=std,
        dt=dt,
        passages=passages,
        nrstage=nrstage,
        clin=clin,
        rho=rho,
        sd_beta=sd_beta,
        delta=delta,
        deltaI=deltaI,
        neps=neps,
    )

    initial_loop_vars = (S, I, Y, deaths, pts, t, count, key)
    S, I, Y, deaths, pts, t, count, key = jax.lax.fori_loop(
        lower=0, upper=20, body_fun=rproc_step2, init_val=initial_loop_vars
    )

    return jnp.hstack(
        [jnp.array([S, I, Y, deaths]), pts, jnp.array([t]), jnp.array([count])]
    )


def dmeas_helper(y, deaths, v, tol, ltol):
    return jnp.logaddexp(
        jax.scipy.stats.norm.logpdf(y, loc=deaths, scale=v + tol), ltol
    )


def dmeas_helper_tol(y, deaths, v, tol, ltol):
    return ltol


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
        Y_,
        deaths,
        v,
        tol,
        ltol,
    )


def dacca():
    dacca_obj = Pomp(
        rinit=rinit, rproc=rproc, dmeas=dmeas, ys=ys, theta=theta, covars=covars
    )
    return dacca_obj
