import unittest
import jax.numpy as jnp
import pypomp as pp

# The test here is based on a
# direct test against R-pomp::logmeanexp
#
# library(pomp)
# x = c(100,101,102,103,104)
# logmeanexp(x,se=TRUE)
#         est          se
# 102.8424765   0.7510094
#
# import jax.numpy as jnp
# import pypomp as pp
# x = jnp.array([100,101,102,103,104])
# pp.logmeanexp(x)
# pp.logmeanexp_se(x)
# >>> pp.logmeanexp(x)
# Array(102.842476, dtype=float32)
# >>> pp.logmeanexp_se(x)
# Array(0.7510107, dtype=float32)


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.logmeanexp = pp.logmeanexp
        self.logmeanexp_se = pp.logmeanexp_se
        self.x = jnp.array([100, 101, 102, 103, 104])

    def test_val(self):
        lme = self.logmeanexp(self.x)
        lme_se = self.logmeanexp_se(self.x)
        self.assertEqual(jnp.round(lme, 2), 102.84)
        self.assertEqual(jnp.round(lme_se, 2), 0.75)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
