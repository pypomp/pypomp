[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![codecov](https://codecov.io/gh/pypomp/pypomp/graph/badge.svg?token=8TA2X3DRML)](https://codecov.io/gh/pypomp/pypomp)
[![Documentation Status](https://app.readthedocs.org/projects/pypomp/badge/?version=latest)](https://pypomp.readthedocs.io/en/latest/?badge=latest)

# Pypomp

Pypomp is a Python/JAX library for modeling and inference using partially observed Markov process (POMP) models.
Newcomers are invited to read the [introductory tutorial](https://pypomp.github.io/tutorials) and a short course teaching [practical modeling and data analysis](https://pypomp.github.io/tutorials/sbied)  using Pypomp.
Documentation is on [readthedocs](https://pypomp.readthedocs.io/).
Additional [quantitative tests](https://pypomp.github.io/quant) provide performance evaluation and technical examples. 


### Expected users

1. Scientists wanting to perform data analysis on a dynamic system via a POMP model, also called a state-space model (SSM) or hidden Markov model (HMM).

2. Researchers wishing to develop novel inference methodology. Pypomp provides an abstract representation of POMP models that enables researchers to develop, test, and deploy novel algorithms applicable to arbitrary nonlinear non-Gaussian POMP models.

3. Researchers familiar with the [pomp R package](https://kingaa.github.io/pomp/). Pypomp extends R-pomp by supporting GPU computing, automatic differentation, and just-in-time compilation. Conceptually, Pypomp is similar to R-pomp, and so case studies listed in the [R-pomp package bibliography](https://kingaa.github.io/pomp/biblio.html) are pertinent.


### Key features 

1. Parameter estimation, model evaluation and latent state estimation for nonlinear, non-Gaussian POMP models via the particle filter.

2. Gradient descent using a new [particle filter gradient estimate](https://arxiv.org/abs/2407.03085). This provides state-of-the-art simulation-based maximum likelihood and Bayesian inference.

3. Pypomp uses JAX to provide GPU support, automatic differentiation and just-in-time compilation.


### Governance and contributions

The Pypomp library is run by the [Pypomp organization](https://github.com/pypomp).
All contributions are welcome. Please raise issues or make pull requests on the [Pypomp GitHub site](https://github.com/pypomp/pypomp) or contact the [core development team](https://github.com/pypomp/.github/blob/main/profile/Governance.md)

