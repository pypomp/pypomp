"""
MCMC proposal distributions for use with pmcmc.

Provides three proposal constructors following pomp (R):
- ``mvn_diag_rw``: diagonal multivariate normal random walk
- ``mvn_rw``: full-covariance multivariate normal random walk
- ``mvn_rw_adaptive``: adaptive MVN random walk (Roberts & Rosenthal 2009)

Each constructor returns a callable with signature::

    proposal(theta, key, n=0, accepts=0) -> theta_proposed

where *theta* is a ``dict[str, float]`` of parameter values in the
natural (untransformed) scale.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Diagonal random-walk proposal
# ---------------------------------------------------------------------------

def mvn_diag_rw(rw_sd: dict[str, float]):
    """
    Construct a diagonal multivariate normal random-walk proposal.

    Args:
        rw_sd: Dictionary mapping parameter names to random-walk standard
            deviations.  Parameters with ``sd <= 0`` are silently dropped.

    Returns:
        A callable proposal function.
    """
    rw_sd = {k: float(v) for k, v in rw_sd.items() if v > 0}
    if not rw_sd:
        raise ValueError("rw_sd must contain at least one positive entry.")
    parnm = list(rw_sd.keys())
    sd_arr = jnp.array([rw_sd[p] for p in parnm])

    def _proposal(
        theta: dict[str, float],
        key: jax.Array,
        n: int = 0,
        accepts: int = 0,
    ) -> dict[str, float]:
        z = jax.random.normal(key, shape=(len(parnm),))
        theta_new = dict(theta)
        for i, p in enumerate(parnm):
            theta_new[p] = float(theta[p] + z[i] * sd_arr[i])
        return theta_new

    _proposal._parnm = parnm  # type: ignore[attr-defined]
    return _proposal


# ---------------------------------------------------------------------------
# Full-covariance random-walk proposal
# ---------------------------------------------------------------------------

def mvn_rw(rw_var: np.ndarray, param_names: list[str]):
    """
    Construct a full-covariance multivariate normal random-walk proposal.

    Args:
        rw_var: Symmetric positive-definite covariance matrix of shape
            ``(d, d)`` where ``d = len(param_names)``.
        param_names: Parameter names corresponding to the rows/columns of
            *rw_var*.

    Returns:
        A callable proposal function.
    """
    rw_var = np.asarray(rw_var, dtype=float)
    if rw_var.ndim != 2 or rw_var.shape[0] != rw_var.shape[1]:
        raise ValueError("rw_var must be a square matrix.")
    if rw_var.shape[0] != len(param_names):
        raise ValueError("rw_var dimensions must match len(param_names).")
    chol = jnp.array(np.linalg.cholesky(rw_var))  # lower-triangular
    parnm = list(param_names)

    def _proposal(
        theta: dict[str, float],
        key: jax.Array,
        n: int = 0,
        accepts: int = 0,
    ) -> dict[str, float]:
        z = jax.random.normal(key, shape=(len(parnm),))
        delta = chol @ z
        theta_new = dict(theta)
        for i, p in enumerate(parnm):
            theta_new[p] = float(theta[p] + delta[i])
        return theta_new

    _proposal._parnm = parnm  # type: ignore[attr-defined]
    return _proposal


# ---------------------------------------------------------------------------
# Adaptive random-walk proposal  (Roberts & Rosenthal, 2009)
# ---------------------------------------------------------------------------

class MVNRWAdaptive:
    """
    Adaptive multivariate normal random-walk proposal.

    Implements the two-phase adaptation scheme of Roberts & Rosenthal (2009):

    * **Phase 1** (scale adaptation, ``n >= scale_start`` and
      ``accepts < shape_start``): a global scaling factor is adjusted to
      drive the acceptance ratio toward *target*.
    * **Phase 2** (shape adaptation, ``accepts >= shape_start``): the
      proposal covariance is replaced by a scaled empirical covariance
      ``(2.38² / d) × Σ_emp``.

    Args:
        rw_sd: Named dict of per-parameter random-walk SDs (diagonal
            initialisation).  Mutually exclusive with *rw_var*.
        rw_var: Full initial covariance matrix.  Mutually exclusive with
            *rw_sd*.
        param_names: Required when *rw_var* is supplied.
        scale_start: Iteration at which to begin scale adaptation.
        scale_cooling: Cooling factor for the scale update.
        shape_start: Number of *accepted* proposals before switching to the
            empirical covariance.
        target: Target Metropolis acceptance ratio (default 0.234).
        max_scaling: Upper bound for the scaling factor.
    """

    def __init__(
        self,
        rw_sd: dict[str, float] | None = None,
        rw_var: np.ndarray | None = None,
        param_names: list[str] | None = None,
        scale_start: int = 200,
        scale_cooling: float = 0.999,
        shape_start: int = 200,
        target: float = 0.234,
        max_scaling: float = 50.0,
    ):
        if (rw_sd is None) == (rw_var is None):
            raise ValueError("Exactly one of rw_sd and rw_var must be given.")

        if rw_sd is not None:
            rw_sd = {k: float(v) for k, v in rw_sd.items() if v > 0}
            self.parnm = list(rw_sd.keys())
            d = len(self.parnm)
            self._rw_var = np.diag([rw_sd[p] ** 2 for p in self.parnm])
        else:
            rw_var = np.asarray(rw_var, dtype=float)
            if param_names is None:
                raise ValueError("param_names required when rw_var is given.")
            if rw_var.shape != (len(param_names), len(param_names)):
                raise ValueError("rw_var shape must match param_names.")
            self.parnm = list(param_names)
            d = len(self.parnm)
            self._rw_var = rw_var.copy()

        if scale_start < 1:
            raise ValueError("scale_start must be a positive integer.")
        if not (0 < scale_cooling <= 1):
            raise ValueError("scale_cooling must be in (0, 1].")
        if shape_start < 1:
            raise ValueError("shape_start must be a positive integer.")
        if not (0 < target < 1):
            raise ValueError("target must be in (0, 1).")

        self.scale_start = int(scale_start)
        self.scale_cooling = float(scale_cooling)
        self.shape_start = int(shape_start)
        self.target = float(target)
        self.max_scaling = float(max_scaling)

        # Mutable state
        self._scaling = 1.0
        self._theta_mean: np.ndarray | None = None
        self._covmat_emp = np.zeros((d, d))
        self._d = d

    def reset(self):
        """Reset the adaptive state (e.g. when starting a new chain)."""
        self._scaling = 1.0
        self._theta_mean = None
        self._covmat_emp = np.zeros((self._d, self._d))

    def __call__(
        self,
        theta: dict[str, float],
        key: jax.Array,
        n: int = 0,
        accepts: int = 0,
    ) -> dict[str, float]:
        """Draw a proposal given current *theta* and chain state."""
        d = self._d
        parnm = self.parnm

        # Extract current values as array
        theta_arr = np.array([theta[p] for p in parnm])

        # Initialise running mean on first real call
        if self._theta_mean is None:
            self._theta_mean = theta_arr.copy()

        # ---- Determine covariance matrix for this iteration ----
        if n >= self.scale_start and accepts < self.shape_start:
            # Phase 1: scale adaptation
            accept_rate = accepts / max(n, 1)
            self._scaling = min(
                self._scaling
                * np.exp(
                    self.scale_cooling ** (n - self.scale_start)
                    * (accept_rate - self.target)
                ),
                self.max_scaling,
            )
            covmat = self._scaling**2 * self._rw_var
        elif accepts >= self.shape_start:
            # Phase 2: shape adaptation with empirical covariance
            scaling = 2.38**2 / d
            covmat = scaling * self._covmat_emp
        else:
            # Before any adaptation
            covmat = self._rw_var

        # Update running mean and empirical covariance (Welford's algorithm)
        old_mean = self._theta_mean.copy()
        self._theta_mean = old_mean + (theta_arr - old_mean) / max(n, 1)
        diff_old = theta_arr - old_mean
        diff_new = theta_arr - self._theta_mean
        self._covmat_emp = (
            (n - 1) * self._covmat_emp + np.outer(diff_old, diff_new)
        ) / max(n, 1)

        # Draw from proposal
        try:
            chol = np.linalg.cholesky(covmat)
        except np.linalg.LinAlgError:
            # Fall back to initial covariance if empirical is singular
            chol = np.linalg.cholesky(self._rw_var + 1e-10 * np.eye(d))

        z = jax.random.normal(key, shape=(d,))
        delta = jnp.array(chol) @ z

        theta_new = dict(theta)
        for i, p in enumerate(parnm):
            theta_new[p] = float(theta[p] + delta[i])
        return theta_new

    @property
    def _parnm(self):
        return self.parnm


def mvn_rw_adaptive(
    rw_sd: dict[str, float] | None = None,
    rw_var: np.ndarray | None = None,
    param_names: list[str] | None = None,
    scale_start: int = 200,
    scale_cooling: float = 0.999,
    shape_start: int = 200,
    target: float = 0.234,
    max_scaling: float = 50.0,
) -> MVNRWAdaptive:
    """
    Construct an adaptive MVN random-walk proposal (Roberts & Rosenthal 2009).

    See :class:`MVNRWAdaptive` for argument documentation.
    """
    return MVNRWAdaptive(
        rw_sd=rw_sd,
        rw_var=rw_var,
        param_names=param_names,
        scale_start=scale_start,
        scale_cooling=scale_cooling,
        shape_start=shape_start,
        target=target,
        max_scaling=max_scaling,
    )
