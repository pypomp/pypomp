import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG

class TestPfilter_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG()
        self.key = jax.random.key(111)
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas
    
    # default diagnostics = False
    def test_class_basic_default(self):
        self.LG.pfilter(J=self.J, key=self.key)
        val1 = self.LG.results_history[-1]["logLiks"][0]
        self.assertEqual(val1.shape, (1,))
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_reps_default(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]
        self.LG.pfilter(J=self.J, key=self.key, reps=2, theta=theta_list)
        val1 = self.LG.results_history[-1]["logLiks"][0]
        self.assertEqual(val1.shape, (2,))

    def test_class_basic_false(self):
        self.LG.pfilter(J=self.J, key=self.key, diagnostics=False)
        val1 = self.LG.results_history[-1]["logLiks"][0]
        self.assertEqual(val1.shape, (1,))
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_reps_false(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]
        self.LG.pfilter(J=self.J, key=self.key, reps=2, theta=theta_list, diagnostics=False)
        val1 = self.LG.results_history[-1]["logLiks"][0]
        self.assertEqual(val1.shape, (2,))

    def test_class_basic_diagnostic(self):
        self.LG.pfilter(J=self.J, key=self.key, diagnostics=True)
        method = self.LG.results_history[-1]["method"]
        self.assertEqual(method, "pfilter")
        
        negLogLiks = self.LG.results_history[-1]["logLiks"]
        negLogLiks_arr = negLogLiks.data
        self.assertEqual(negLogLiks_arr.shape, (1, 1))
        self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(negLogLiks_arr.dtype, jnp.float32)

        condLogLiks = self.LG.results_history[-1]["CLL"]
        condLogLiks_arr = condLogLiks.data
        self.assertEqual(condLogLiks_arr.shape, (1, 1, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(condLogLiks.dtype, jnp.floating)) 
        self.assertEqual(condLogLiks.dtype, jnp.float32)

        filter_mean = self.LG.results_history[-1]["filter_mean"]
        filter_mean_arr = filter_mean.data
        self.assertEqual(filter_mean_arr.shape, (1, 1, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
        self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)) 
        self.assertEqual(filter_mean_arr.dtype, jnp.float32)

        ess = self.LG.results_history[-1]["ESS"]
        ess_arr = ess.data
        self.assertEqual(ess_arr.shape, (1, 1, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
        self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
        self.assertEqual(ess_arr.dtype, jnp.float32)
        # all elements should be smaller than self.J and leq than 0
        self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr < self.J)))

        prediction_mean = self.LG.results_history[-1]["prediction_mean"]
        prediction_mean_arr = prediction_mean.data
        self.assertEqual(prediction_mean_arr.shape, (1, 1, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(prediction_mean_arr)))  
        self.assertTrue(jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)) 
        self.assertEqual(prediction_mean_arr.dtype, jnp.float32)

        
    def test_reps_diagnostic(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]
        self.LG.pfilter(J=self.J, key=self.key, reps=2, theta=theta_list, diagnostics=True)
        method = self.LG.results_history[-1]["method"]
        self.assertEqual(method, "pfilter")
        
        negLogLiks = self.LG.results_history[-1]["logLiks"]
        negLogLiks_arr = negLogLiks.data
        self.assertEqual(negLogLiks_arr.shape, (2, 2))
        self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(negLogLiks_arr.dtype, jnp.float32)

        condLogLiks = self.LG.results_history[-1]["CLL"]
        condLogLiks_arr = condLogLiks.data
        self.assertEqual(condLogLiks_arr.shape, (2,2, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(condLogLiks_arr.dtype, jnp.float32)

        filter_mean = self.LG.results_history[-1]["filter_mean"]
        filter_mean_arr = filter_mean.data
        self.assertEqual(filter_mean_arr.shape, (2, 2, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
        self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)) 
        self.assertEqual(filter_mean_arr.dtype, jnp.float32)

        ess = self.LG.results_history[-1]["ESS"]
        ess_arr = ess.data
        self.assertEqual(ess_arr.shape, (2, 2, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
        self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
        self.assertEqual(ess_arr.dtype, jnp.float32)
        # all elements should be smaller than self.J and leq than 0
        self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr < self.J)))
        
        prediction_mean = self.LG.results_history[-1]["prediction_mean"]
        prediction_mean_arr = prediction_mean.data
        self.assertEqual(prediction_mean_arr.shape, (2, 2, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(prediction_mean_arr)))  
        self.assertTrue(jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)) 
        self.assertEqual(prediction_mean_arr.dtype, jnp.float32)



if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
