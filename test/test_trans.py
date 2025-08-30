import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pypomp.parameter_trans import (
    ParTrans,
    parameter_trans,
    materialize_partrans,
    _pt_forward,
    _pt_inverse,
)


class TestParameterTrans(unittest.TestCase):
    def test_materialize_none_identity(self):
        pt = materialize_partrans(None, None)
        self.assertEqual(pt, ParTrans(False, (), (), (), None, None))

    def test_materialize_passthrough_partrans(self):
        base = ParTrans(True, (0,), (1,), (2,), lambda x: x + 1.0, lambda x: x - 1.0)
        pt = materialize_partrans(base, ["a", "b", "c"])
        self.assertIs(pt, base)

    def test_materialize_by_names(self):
        names = ["a", "b", "c", "d"]
        spec = parameter_trans(
            log="a",
            logit=["c"],
            custom="d",
            to_est=lambda x: x * 2.0 + 1.0,
            from_est=lambda x: (x - 1.0) / 2.0,
        )
        pt = materialize_partrans(spec, names)
        self.assertEqual(pt.log_idx, (0,))
        self.assertEqual(pt.logit_idx, (2,))
        self.assertEqual(pt.custom_idx, (3,))
        self.assertTrue(pt.is_custom)

    def test_materialize_by_indices(self):
        spec = parameter_trans(
            log=(0, 2),
            logit=1,
            custom=3,
            to_est=lambda x: x + 5.0,
            from_est=lambda x: x - 5.0,
        )
        pt = materialize_partrans(spec, None)
        self.assertEqual(pt.log_idx, (0, 2))
        self.assertEqual(pt.logit_idx, (1,))
        self.assertEqual(pt.custom_idx, (3,))

    def test_overlaps_raise(self):
        names = ["a", "b", "c"]
        for bad in [
            parameter_trans(log="a", logit="a"),
            parameter_trans(log="a", custom="a", to_est=lambda x: x, from_est=lambda x: x),
            parameter_trans(logit="b", custom="b", to_est=lambda x: x, from_est=lambda x: x),
        ]:
            with self.assertRaises(ValueError):
                materialize_partrans(bad, names)

    def test_custom_requires_both_functions(self):
        names = ["a"]
        with self.assertRaises(ValueError):
            materialize_partrans(parameter_trans(custom="a", to_est=lambda x: x + 1.0), names)
        with self.assertRaises(ValueError):
            materialize_partrans(parameter_trans(custom="a", from_est=lambda x: x - 1.0), names)

    def test_custom_functions_given_without_indices_raises(self):
        with self.assertRaises(ValueError):
            materialize_partrans(
                parameter_trans(to_est=lambda x: x, from_est=lambda x: x), ["a", "b"]
            )

    def test_unknown_name_raises(self):
        names = ["a", "b"]
        spec = parameter_trans(log="zzz")
        with self.assertRaises(ValueError):
            materialize_partrans(spec, names)

    def test_names_without_paramnames_raises(self):
        spec = parameter_trans(log="a")
        with self.assertRaises(TypeError):
            materialize_partrans(spec, None)

    def test_forward_inverse_roundtrip_vector(self):
        # indices: log=0, logit=1, custom=2, last one untouched
        spec = parameter_trans(
            log=0,
            logit=1,
            custom=2,
            to_est=lambda x: x * 3.0 - 2.0,
            from_est=lambda x: (x + 2.0) / 3.0,
        )
        pt = materialize_partrans(spec, None)
        theta_nat = jnp.array([2.0, 0.2, -4.0, 7.0])
        z = _pt_forward(theta_nat, pt)
        x = _pt_inverse(z, pt)
        self.assertTrue(jnp.allclose(x, theta_nat, rtol=1e-6, atol=1e-6))

        # Check individual transformed components numerically
        self.assertTrue(np.isclose(z[0], jnp.log(2.0)))
        self.assertTrue(np.isclose(z[1], jnp.log(0.2) - jnp.log1p(-0.2)))
        self.assertTrue(np.isclose(z[2], -4.0 * 3.0 - 2.0))
        self.assertTrue(np.isclose(z[3], 7.0))

    def test_forward_inverse_roundtrip_batch(self):
        spec = parameter_trans(
            log=(0,),
            logit=(1,),
            custom=(2,),
            to_est=lambda x: x * 2.0 + 1.0,
            from_est=lambda x: (x - 1.0) / 2.0,
        )
        pt = materialize_partrans(spec, None)
        theta_nat = jnp.array(
            [
                [1.5, 0.75, 3.0, -1.0],
                [3.2, 0.1, -2.5, 4.0],
            ]
        )
        z = _pt_forward(theta_nat, pt)
        x = _pt_inverse(z, pt)
        self.assertEqual(z.shape, theta_nat.shape)
        self.assertTrue(jnp.allclose(x, theta_nat, rtol=1e-6, atol=1e-6))

    def test_custom_shape_mismatch_forward_raises(self):
        names = ["a", "b", "c"]
        # wrong to_est returns wrong shape
        spec = parameter_trans(
            custom="c",
            to_est=lambda x: jnp.stack([x, x], axis=-1),
            from_est=lambda x: x[..., 0],
        )
        pt = materialize_partrans(spec, names)
        theta_nat = jnp.array([1.0, 0.5, 2.0])
        with self.assertRaises(ValueError):
            _pt_forward(theta_nat, pt)

    def test_custom_shape_mismatch_inverse_raises(self):
        names = ["a", "b", "c"]
        # wrong from_est returns wrong shape
        spec = parameter_trans(
            custom="c",
            to_est=lambda x: x + 1.0,
            from_est=lambda x: jnp.stack([x, x], axis=-1),
        )
        pt = materialize_partrans(spec, names)
        theta_nat = jnp.array([2.0, 0.3, 1.0])
        z = _pt_forward(theta_nat, pt)
        with self.assertRaises(ValueError):
            _pt_inverse(z, pt)


if __name__ == "__main__":
    unittest.main()