[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![codecov](https://codecov.io/gh/pypomp/pypomp/graph/badge.svg?token=8TA2X3DRML)](https://codecov.io/gh/pypomp/pypomp)
[![Documentation Status](https://app.readthedocs.org/projects/pypomp/badge/?version=latest)](https://pypomp.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pypomp.svg)](https://badge.fury.io/py/pypomp)

# Pypomp

Pypomp is a Python/JAX library for modeling and inference using partially observed Markov process (POMP) models, also called state-space models or hidden Markov models.

---

### 🚀 Quick Links
| 📖 **[Read the Documentation](https://pypomp.readthedocs.io/)** | 🎓 **[Introductory Tutorial](https://pypomp.github.io/tutorials)** |
|:---:|:---:|
| 🏫 **[SBIED Course (Practical Modeling)](https://pypomp.github.io/tutorials/sbied)** | 📊 **[Quantitative Benchmarks](https://pypomp.github.io/quant)** |

---

### Installation

Install Pypomp using pip:

```bash
pip install pypomp
```

> 📝 **Note:** Pypomp is powered by **JAX** for GPU acceleration and just-in-time (JIT) compilation. To configure JAX for GPU computing, please follow the [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html).


### Quick Start

Get started quickly by running a particle filter on a built-in Linear Gaussian model:

```python
import jax
import pypomp as pp

# 1. Initialize a built-in Linear Gaussian model with 50 time steps
model = pp.models.LG(T=50)

# 2. Run the particle filter with J=1000 particles
key = jax.random.key(1)
model.pfilter(J=1000, key=key)

# 3. Retrieve and print the estimated log-likelihood using .results()
results_df = model.results()
log_lik = results_df["logLik"].iloc[0]
print(f"Log-Likelihood: {log_lik:.4f}")
```


### Expected Users

1. Scientists wanting to perform data analysis on a dynamic system via a POMP model.

2. Researchers wishing to develop novel inference methodology. Pypomp provides an abstract representation of POMP models that enables researchers to develop, test, and deploy novel algorithms applicable to arbitrary nonlinear non-Gaussian POMP models.

3. Researchers familiar with the [pomp R package](https://kingaa.github.io/pomp/). Pypomp extends R-pomp by supporting GPU computing, automatic differentiation, and just-in-time compilation. Conceptually, Pypomp is similar to R-pomp, and so case studies listed in the [R-pomp package bibliography](https://kingaa.github.io/pomp/biblio.html) are pertinent.


### Key Features

1. **State & Parameter Inference**: Parameter estimation, model evaluation, and latent state estimation for nonlinear, non-Gaussian POMP models via the particle filter.

2. **Differentiable Particle Filtering**: Gradient descent using a new [particle filter gradient estimate](https://arxiv.org/abs/2407.03085). This provides state-of-the-art simulation-based maximum likelihood and Bayesian inference.

3. **JAX-Backed Performance**: Leverages JAX to provide GPU support, automatic differentiation, and just-in-time compilation.


### Governance and Contributions

The Pypomp library is run by the [Pypomp organization](https://github.com/pypomp).

All contributions are welcome. Please review our [Contributing Guide](CONTRIBUTING.md) before submitting pull requests or raising issues.

For governance details, you can contact the core development team or refer to the [pypomp governance profile](https://github.com/pypomp/.github/blob/main/profile/Governance.md).


### 📚 Citation

If you use Pypomp in your research, please cite it as:

```bibtex
@software{pypomp2024github,
  author  = {Aaron Abkemeier and Jun Chen and Edward Ionides and Jesse Wheeler and Kevin Tan},
  title   = {Pypomp},
  url     = {https://github.com/pypomp/pypomp},
  version = {0.4.7.0},
  year    = {2026}
}
```
