import jax
import unittest
import pypomp as pp


class Test_Dacca(unittest.TestCase):
    def setUp(self):
        self.dacca = pp.dacca()
        self.J = 3
        self.key = jax.random.key(111)
        self.ys = self.dacca.ys

    def test_dacca_basic(self):
        # Check whether dacca.mif() finishes running. I think this should be sufficient
        # to check whether dacca's attributes are set up correctly, as mif uses all of
        # them.
        self.dacca.mif(
            sigmas=0.02,
            sigmas_init=0.1,
            J=self.J,
            thresh=-1,
            key=self.key,
            M=1,
            a=0.9,
        )

    def test_dacca_nstep(self):
        # Check that dacca.train() runs without error when nstep is specified.
        dacca_nstep = pp.dacca(nstep=10, dt=None)
        dacca_nstep.train(J=self.J, itns=1, key=self.key)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
