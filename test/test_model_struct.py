import unittest
import pypomp as pp
import jax.numpy as jnp


class TestModelStruct(unittest.TestCase):
    def test_RInit_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.RInit(lambda foo, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RInit(lambda theta_, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RInit(lambda theta_, key, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.RInit(lambda theta_, key, covars: jnp.array([0]))

    def test_RProc_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.RProc(lambda foo, theta_, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RProc(lambda X_, foo, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RProc(lambda X_, theta_, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RProc(lambda X_, theta_, key, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.RProc(lambda X_, theta_, key, covars: jnp.array([0]))

    def test_DMeas_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.DMeas(lambda foo, X_, theta_, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.DMeas(lambda Y_, foo, theta_, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.DMeas(lambda Y_, X_, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.DMeas(lambda Y_, X_, theta_, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.DMeas(lambda Y_, X_, theta_, covars: jnp.array([0]))

    def test_RMeas_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.RMeas(lambda foo, theta_, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RMeas(lambda X_, foo, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RMeas(lambda X_, theta_, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RMeas(lambda X_, theta_, key, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.RMeas(lambda X_, theta_, key, covars: jnp.array([0]))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
