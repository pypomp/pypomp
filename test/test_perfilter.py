import os
import jax
import sys
import unittest
import jax.numpy as jnp

from jax import vmap
from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.perfilter import perfilter
from pypomp.internal_functions import _perfilter_internal

#current_dir = os.getcwd()
#sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
#from LG import LG

sys.path.insert(0, 'pypomp')
from LG import LG

def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return jnp.array([A, C, Q, R])

get_perthetas = vmap(get_thetas, in_axes = 0)

LG_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = LG()

class TestPerfilter_LG(unittest.TestCase):
    def setUp(self):
        self.J = 5
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.sigmas = 0.02
        self.key = jax.random.PRNGKey(111)
        
        self.rinit = rinit
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures

    def test_internal_basic(self):
        val1, theta1 = perfilter(J=self.J, rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                 theta=self.theta, ys=self.ys, sigmas=self.sigmas, covars=self.covars, thresh=-1, 
                                 key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (self.J, 16))
        theta1_new = get_perthetas(theta1) #?
        self.assertEqual(theta1_new.shape, (self.J, 4, 2, 2))

        val2, theta2 = perfilter(rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                 theta=self.theta, ys=self.ys, sigmas=self.sigmas)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)
        self.assertEqual(theta2.shape, (50, 16))
        theta2_new = get_perthetas(theta2)
        self.assertEqual(theta2_new.shape, (50, 4, 2, 2))


    def test_class_basic(self):
        
        val1, theta1 = perfilter(LG_obj, J=self.J, sigmas=self.sigmas, thresh=100)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (self.J, 16))
        theta1_new = get_perthetas(theta1)
        self.assertEqual(theta1_new.shape, (self.J, 4, 2, 2))


        val2, theta2 = perfilter(LG_obj, sigmas=self.sigmas)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)
        self.assertEqual(theta2.shape, (50, 16))
        theta2_new = get_perthetas(theta2)
        self.assertEqual(theta2_new.shape, (50, 4, 2, 2))


        val3, theta3 = perfilter(LG_obj, J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                                 dmeasures=self.dmeasures, theta=[], ys=[])
        self.assertEqual(val3.shape, ())
        self.assertTrue(jnp.isfinite(val3.item()))
        self.assertEqual(val3.dtype, jnp.float32)
        self.assertEqual(theta3.shape, (self.J, 16))
        theta3_new = get_perthetas(theta3)
        self.assertEqual(theta3_new.shape, (self.J, 4, 2, 2))


    def test_invalid_input(self):
        with self.assertRaises(ValueError) as text:
            perfilter()

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, thresh=-1, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                      dmeasures=self.dmeasures)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                      dmeasures=self.dmeasures, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                      dmeasures=self.dmeasures, theta=self.theta)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(TypeError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                      theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "perfilter() got an unexpected keyword argument 'rprocess'")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
