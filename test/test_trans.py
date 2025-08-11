# test/test_trans_mif.py
# Integration tests for parameter transformations that actually call Pomp.mif.
# We keep them tiny and deterministic so the suite stays fast and reliable.

import unittest
import jax
import jax.numpy as jnp

import pypomp as pp
from pypomp.mif import parameter_trans


class TestParTransMIF(unittest.TestCase):
    """
    This suite verifies two things at the algorithm (integration) level:
      (1) Built-in log/logit transformations keep parameters in their proper domains
          across MIF iterations.
      (2) Custom to_est/from_est transformations behave equivalently in terms of
          domain constraints and do not break MIF.
    We use the shipped linear Gaussian (LG) model for speed & determinism.
    """

    def _common_small_mif_args(self):
        # Small, fast settings so tests finish quickly.
        return dict(J=30, M=3, a=0.9, sigmas=0.005, sigmas_init=0.002)

    def test_mif_with_builtin_log_logit(self):
        # Fresh model to avoid state pollution across tests
        lg = pp.LG()
        names = list(lg.theta[0].keys())

        # Pick one positive parameter and one (0,1) parameter.
        # In LG defaults: R1 ~ 0.1 (>0), A1 ~ cos(0.2) (~0.98, in (0,1)).
        pos_name = "R1"
        unit_name = "A1"
        assert pos_name in names and unit_name in names

        # Built-in spec: log on R1, logit on A1 (standard 0..1 logit as in R pomp).
        par = parameter_trans(log=[pos_name], logit=[unit_name])

        # Run a tiny MIF
        key = jax.random.key(20250811)
        lg.mif(key=key, partrans=par, paramnames=names, **self._common_small_mif_args())

        # Grab the trace (only one replication here)
        trace = lg.results_history[-1]["traces"][0]  # pandas DataFrame

        # Domain checks: all iterations should respect the constraints after inverse transform.
        self.assertTrue((trace[pos_name] > 0.0).all(), "R1 must stay positive with log transform")
        self.assertTrue((trace[unit_name] > 0.0).all() and (trace[unit_name] < 1.0).all(),
                        "A1 must stay in (0,1) with logit transform")

        # Basic shape/sanity
        self.assertIn("logLik", trace.columns)
        self.assertEqual(trace.shape[0], self._common_small_mif_args()["M"] + 1)

    def test_mif_with_custom_to_from(self):
        # Fresh model again
        lg = pp.LG()
        names = list(lg.theta[0].keys())

        pos_name = "R1"
        unit_name = "A1"
        i_pos = names.index(pos_name)
        i_unit = names.index(unit_name)

        # Define custom to_est/from_est that perform log on R1 and logit on A1.
        eps = 1e-12

        def to_est(theta):
            # theta: (..., d)
            z = theta
            # log on R1
            z = z.at[..., i_pos].set(jnp.log(theta[..., i_pos]))
            # logit on A1
            u = jnp.clip(theta[..., i_unit], eps, 1 - eps)
            z = z.at[..., i_unit].set(jnp.log(u) - jnp.log1p(-u))
            return z

        def from_est(z):
            # z: (..., d)
            x = z
            x = x.at[..., i_pos].set(jnp.exp(z[..., i_pos]))
            x = x.at[..., i_unit].set(jax.nn.sigmoid(z[..., i_unit]))
            return x

        par = parameter_trans(to_est=to_est, from_est=from_est)

        key = jax.random.key(20250812)
        lg.mif(key=key, partrans=par, paramnames=names, **self._common_small_mif_args())

        trace = lg.results_history[-1]["traces"][0]

        # Domain checks (same as built-in path)
        self.assertTrue((trace[pos_name] > 0.0).all(), "R1 must stay positive under custom log")
        self.assertTrue((trace[unit_name] > 0.0).all() and (trace[unit_name] < 1.0).all(),
                        "A1 must stay in (0,1) under custom logit")

        # Shape/sanity
        self.assertIn("logLik", trace.columns)
        self.assertEqual(trace.shape[0], self._common_small_mif_args()["M"] + 1)


if __name__ == "__main__":
    unittest.main()