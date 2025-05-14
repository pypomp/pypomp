[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![codecov](https://codecov.io/gh/pypomp/pypomp/graph/badge.svg?token=8TA2X3DRML)](https://codecov.io/gh/pypomp/pypomp)

# pypomp

Python code for modeling and inference using partially observed Markov process (POMP) models.
See the [tutorials](https://pypomp.github.io/tutorials) for user-friendly guides, and the [quantitative tests](https://pypomp.github.io/quant) for additional technical examples.

### Expected package users

* Scientists wanting to perform data analysis on a dynamic system via partially observed Markov processes (POMPs), also called state-space models (SSM) or hidden Markov models (HMM) in other contexts.

  * This package design and intended use is similar to the popular [**pomp** R package](https://kingaa.github.io/pomp/). As such, many of the expected use cases and motivating examples of this package can be found on the [pomp package bibliography page](https://kingaa.github.io/pomp/biblio.html).
  
* Researchers wishing to develop novel inference methodology for POMP models.

  * Like the **pomp** R package, this package provides a framework for implementing computer representations of arbitrary POMP models. This ability provides an environment for researchers to develop, test, and deploy novel algorithms that are applicable to POMP models.
 
### Key features 

* Estimation, filtering, and inference for highly nonlinear, non-Gaussian state space models via the particle filter.

* New algorithms for model-fitting. Gradient descent using a new [gradient estimate](https://arxiv.org/abs/2407.03085) initialized with a [warm-start](https://www.pnas.org/doi/full/10.1073/pnas.1410597112) allows for improved maximum-likelihood inference in even highly challenging epidemiological models, while the gradient estimate can readily be plugged into a sampler from [Tensorflow Probability](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler) to facilitate more efficient Bayesian inference. 

* This package leverages JAX for GPU support and just-in-time compilation, enabling a speedup of up to 16x when compared to the **pomp** R package.

### Package Development 

* The **pypomp** package is currently at early stages of development. All version numbers below 0.1 are pre-release.

* All contributions are welcome! Contributions should keep in mind the intended uses of this package, and its intended users.

* The **pypomp** package is run by the [pypomp organization](https://github.com/pypomp).

