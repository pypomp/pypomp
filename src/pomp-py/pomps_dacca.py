

import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels



def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))



'''
def get_thetas(theta):
    gamma = np.exp(theta[0]) #rate at which I recovers
    m = np.exp(theta[1]) #probability of death from cholera
    rho = np.exp(theta[2]) #1/rho is mean duration of short-term immunity
    epsilon = np.exp(theta[3]) # 1/eps is mean duration of immunity
    omega = np.exp(theta[4]) #mean foi
    c = sigmoid(theta[5] / 5) #probability exposure infects
    beta_trend = theta[6] / 1000 #trend in foi
    sigma = theta[7]**2 / 2 #stdev of foi perturbations
    tau = theta[8]**2 / 5 #stdev of gaussian measurements
    bs = theta[9:15] #seasonality coefficients
    omegas = theta[15:]
    k = 3# 1/(np.exp(theta[3])**2) #1/sqrt(k) is coefficient of variation of immune period
    delta = 0.02 #death rate
    return gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas, k, delta

def transform_thetas(gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas):
    return np.concatenate([np.array([np.log(gamma), np.log(m), np.log(rho), np.log(epsilon), np.log(omega),
                    logit(c)*5, beta_trend * 1000, np.sqrt(sigma*2), np.sqrt(tau*5)]), bs, omegas])
'''
def get_thetas(theta):
    gamma = np.exp(theta[0]) #rate at which I recovers
    m = np.exp(theta[1]) #probability of death from cholera
    rho = np.exp(theta[2]) #1/rho is mean duration of short-term immunity
    epsilon = np.exp(theta[3]) # 1/eps is mean duration of immunity
    omega = np.exp(theta[4]) #mean foi
    c = sigmoid(theta[5] ) #probability exposure infects
    beta_trend = theta[6] / 100 #trend in foi
    sigma = np.exp(theta[7]) #stdev of foi perturbations
    tau = np.exp(theta[8]) #stdev of gaussian measurements
    bs = theta[9:15] #seasonality coefficients
    omegas = theta[15:]
    k = 3# 1/(np.exp(theta[3])**2) #1/sqrt(k) is coefficient of variation of immune period
    delta = 0.02 #death rate
    return gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas, k, delta

def transform_thetas(gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas):
    return np.concatenate([np.array([np.log(gamma), np.log(m), np.log(rho), np.log(epsilon), np.log(omega),
                    logit(c), beta_trend * 100, np.log(sigma), np.log(tau)]), bs, omegas])




def rinit(theta, J, covars):
    S_0, I_0, Y_0, R1_0, R2_0, R3_0 = 0.621, 0.378, 0, 0.000843, 0.000972, 1.16e-07
    pop = covars[0,2]
    S = pop*S_0
    I = pop*I_0
    Y = pop*Y_0
    R1 = pop*R1_0
    R2 = pop*R2_0
    R3 = pop*R3_0
    Mn = 0
    t = 0
    count = 0
    return np.tile(np.array([S,I,Y,Mn,R1,R2,R3,t, count]), (J,1))

def rinits(thetas, J, covars):
    return rinit(thetas[0], len(thetas), covars)

def dmeas_helper(y, deaths, v, tol, ltol):
    return np.logaddexp(
        jax.scipy.stats.norm.logpdf(y, loc=deaths, scale=v+tol), 
                     ltol)
def dmeas_helper_tol(y, deaths, v, tol, ltol):
    return ltol

def dmeas(y, preds, theta, keys=None):
    deaths = preds[3]; count = preds[-1]; tol = 1.0e-18
    ltol = np.log(tol)
    gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas, k, delta = get_thetas(theta)
    v = tau*deaths
    #return jax.scipy.stats.norm.logpdf(y, loc=deaths, scale=v)
    return jax.lax.cond(np.logical_or((1-np.isfinite(v)).astype(bool), count>0), #if Y < 0 then count violation
                         dmeas_helper_tol, 
                         dmeas_helper,
                       y, deaths, v, tol, ltol)


    '''
    return jax.lax.cond(np.logical_or(np.isfinite(v), count>0), #if Y < 0 then count violation
                 np.log(tol), 
                 np.logaddexp(
                     jax.scipy.stats.norm.logpdf(y, loc=deaths, scale=v+tol), 
                     np.log(tol)
                 ))
    '''


dmeasure = jax.vmap(dmeas, (None,0,None))
dmeasures = jax.vmap(dmeas, (None,0,0))




'''
def rproc(state, theta, key, covar):
    S, I, Y, Mn, Rs = state[0], state[1], state[2], state[3], state[4:]
    trend, dH, H, seas = covar[0], covar[1], covar[2], covar[3:]
    gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, k, delta = get_thetas(theta)
    dt = 1/30
    Mn = 0
    for i in range(30):
        subkey, key = jax.random.split(key)
        dw = jax.random.normal(subkey)*onp.sqrt(dt)
        key = subkey
        foi = omega + (np.exp(beta_trend*trend + np.dot(bs, seas)) + sigma*dw/dt)*I/H
        dS = (k*epsilon*Rs[-1] + rho*Y + dH + delta*H - (foi+delta)*S)*dt
        dI = (c*foi*S - (m+gamma+delta)*I)*dt
        dY = ((1-c)*foi*S - (rho+delta)*Y)*dt
        dRs = [(gamma*I - (k*epsilon+delta)*Rs[0])*dt]
        for i in range(1, len(Rs)):
            dRs.append((k*epsilon*Rs[i-1] - (k*epsilon+delta)*Rs[i])*dt)
        S += dS
        I += dI
        Y += dY
        Mn += m*I*dt
        for i in range(len(Rs)):
            Rs = Rs.at[i].add(dRs[i])
    return np.hstack([np.array([S, I, Y, Mn]), Rs])
'''

def rproc(state, theta, key, covar):
    S, I, Y, deaths, pts, t, count = state[0], state[1], state[2], state[3], state[4:-2], state[-2], state[-1]
    t = t.astype(int)
    trends, dpopdts, pops, seass = covar[:,0], covar[:,1], covar[:,2], covar[:,3:]
    gamma, deltaI, rho, eps, omega, clin, beta_trend, sd_beta, tau, bs, omegas, nrstage, delta = get_thetas(theta)
    dt = 1/240
    deaths = 0
    nrstage = 3
    clin = 1 # HARDCODED SEIR
    rho = 0 # HARDCODED INAPPARENT INFECTIONS
    std = onp.sqrt(dt) #onp.sqrt(onp.sqrt(dt))
    
    neps = eps*nrstage
    rdeaths = np.zeros(nrstage)
    passages = np.zeros(nrstage+1)
    

    for i in range(20):
        trend = trends[t]; dpopdt = dpopdts[t]; pop = pops[t]; seas = seass[t]
        beta = np.exp(beta_trend*trend + np.dot(bs, seas))
        omega = np.exp(np.dot(omegas, seas))
        
        subkey, key = jax.random.split(key)
        dw = jax.random.normal(subkey)*std #rnorm uses variance sqrt(dt), not stdev
        
        effI = I/pop
        births = dpopdt + delta*pop # births
        passages = passages.at[0].set(gamma*I) #recovery
        ideaths = delta*I #natural i deaths
        disease = deltaI*I #disease death
        ydeaths = delta*Y #natural rs deaths
        wanings = rho*Y #loss of immunity
        
        for j in range(nrstage):
            rdeaths = rdeaths.at[j].set(pts[j]*delta) #natural R deaths
            passages = passages.at[j+1].set(pts[j]*neps) # passage to the next immunity class
            
        infections = (omega+(beta+sd_beta*dw/dt)*effI)*S # infection
        sdeaths = delta*S # natural S deaths
        
        S += (births - infections - sdeaths + passages[nrstage] + wanings)*dt
        I += (clin*infections - disease - ideaths - passages[0])*dt
        Y += ((1-clin)*infections - ydeaths - wanings)*dt
        for j in range(nrstage):
            pts = pts.at[j].add((passages[j] - passages[j+1] - rdeaths[j])*dt)
        deaths += disease*dt # cumulative deaths due to disease
                        
        count += np.any(np.hstack([np.array([S, I, Y, deaths]), pts]) < 0)
        
        S = np.clip(S, a_min=0); I = np.clip(I, a_min=0); Y = np.clip(Y, a_min=0)
        pts = np.clip(pts, a_min=0); deaths = np.clip(deaths, a_min=0)
        
        t += 1

    return np.hstack([np.array([S, I, Y, deaths]), pts, np.array([t]), np.array([count])])

rprocess = jax.vmap(rproc, (0, None, 0, None))
rprocesses = jax.vmap(rproc, (0, 0, 0, None))