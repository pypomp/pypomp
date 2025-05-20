import unittest
import pypomp as pp
import jax.numpy as jnp


class TestModelStruct(unittest.TestCase):
    def test_RInit_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.RInit(lambda foo, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RInit(lambda params, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RInit(lambda params, key, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.RInit(lambda params, key, covars: jnp.array([0]))

    def test_RProc_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.RProc(lambda foo, params, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RProc(lambda state, foo, key, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RProc(lambda state, params, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.RProc(lambda state, params, key, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.RProc(lambda state, params, key, covars: jnp.array([0]))

    def test_DMeas_value_error(self):
        # Test that an error is thrown with incorrect arguments
        with self.assertRaises(ValueError):
            pp.DMeas(lambda foo, state, params, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.DMeas(lambda y, foo, params, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.DMeas(lambda y, state, foo, covars: jnp.array([0]))
        with self.assertRaises(ValueError):
            pp.DMeas(lambda y, state, params, foo: jnp.array([0]))
        # Test that correct arguments run without error
        pp.DMeas(lambda y, state, params, covars: jnp.array([0]))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
