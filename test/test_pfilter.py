import jax
import unittest
import jax.numpy as jnp
import xarray as xr

from pypomp import pomp_class
from pypomp.LG import LG
from pypomp.pfilter import _vmapped_pfilter_internal2

class TestPfilter_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG(T=2)
        self.key = jax.random.key(111)
        self.J = 3
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas
    
    # default diagnostics = False
    def test_class_basic_default(self):
        self.LG.pfilter(J=self.J, key=self.key)
        val1 = self.LG.results_history[-1]["logLiks"]
        self.assertEqual(val1.shape, (1, 1))
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

        with self.assertRaises(KeyError):
            _ = self.LG.results_history[-1]["CLL"]

    def test_reps_default(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]
        self.LG.pfilter(J=self.J, key=self.key, theta=theta_list, reps=2)
        val1 = self.LG.results_history[-1]["logLiks"]
        self.assertEqual(val1.shape, (2, 2))

        with self.assertRaises(KeyError):
            _ = self.LG.results_history[-1]["CLL"]
    
    def test_class_basic_diagnostic(self):
        cases = [
            (False, False, False, False),
            #(True, False, False, False),
            #(False, True, False, False),
            #(False, False, True, False),
            #(False, False, False, True),
            (True, True, False, False),
            (False, False, True, True),
            #(True, True, True, True)
        ]
        for CLL, ESS, filter_mean, prediction_mean in cases:
            self.LG.results_history.clear()
            self.LG.pfilter(J=self.J, key=self.key, CLL=CLL, ESS=ESS, filter_mean=filter_mean, prediction_mean=prediction_mean)
            method = self.LG.results_history[-1]["method"]
            self.assertEqual(method, "pfilter")
            negLogLiks = self.LG.results_history[-1]["logLiks"]
            negLogLiks_arr = negLogLiks.data
            self.assertEqual(negLogLiks_arr.shape, (1, 1))
            self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
            self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 

            # CLL:
            if CLL:
                condLogLiks = self.LG.results_history[-1]["CLL"]
                condLogLiks_arr = condLogLiks.data
                self.assertEqual(condLogLiks_arr.shape, (1, 1, len(self.ys)))
                self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
                self.assertTrue(jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)) 
            else:
                with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["CLL"]
            
            # ESS:
            if ESS:
                ess = self.LG.results_history[-1]["ESS"]
                ess_arr = ess.data
                self.assertEqual(ess_arr.shape, (1, 1, len(self.ys)))
                self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
                self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
                # all elements should be smaller than self.J and leq than 0
                self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr < self.J)))
            else:
                with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["ESS"]
            
            # filter_mean:
            if filter_mean:
                filt_mean = self.LG.results_history[-1]["filter_mean"]
                filter_mean_arr = filt_mean.data
                self.assertEqual(filter_mean_arr.shape, (1, 1, len(self.ys), 2))
                self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
                self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)) 
            else:
                with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["filter_mean"]

            # prediction_mean:
            if prediction_mean:
                pred_mean = self.LG.results_history[-1]["prediction_mean"]
                prediction_mean_arr = pred_mean.data
                self.assertEqual(prediction_mean_arr.shape, (1, 1, len(self.ys), 2))
                self.assertTrue(jnp.all(jnp.isfinite(prediction_mean_arr)))  
                self.assertTrue(jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)) 
            else:
                 with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["prediction_mean"]

        
    def test_reps_diagnostic(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]

        cases = [
            (False, False, False, False),
            #(True, False, False, False),
            #(False, True, False, False),
            #(False, False, True, False),
            #(False, False, False, True),
            (True, True, False, False),
            (False, False, True, True),
            #(True, True, True, True)
        ]
        for CLL, ESS, filter_mean, prediction_mean in cases:
            self.LG.results_history.clear()
            self.LG.pfilter(J=self.J, key=self.key, theta=theta_list, reps=2,
                            CLL=CLL, ESS=ESS, filter_mean=filter_mean, prediction_mean=prediction_mean)
            method = self.LG.results_history[-1]["method"]
            self.assertEqual(method, "pfilter")
            negLogLiks = self.LG.results_history[-1]["logLiks"]
            negLogLiks_arr = negLogLiks.data
            self.assertEqual(negLogLiks_arr.shape, (2, 2))
            self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
            self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 

            # CLL:
            if CLL:
                condLogLiks = self.LG.results_history[-1]["CLL"]
                condLogLiks_arr = condLogLiks.data
                self.assertEqual(condLogLiks_arr.shape, (2, 2, len(self.ys)))
                self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
                self.assertTrue(jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)) 
            else:
                with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["CLL"]
            
            # ESS:
            if ESS:
                ess = self.LG.results_history[-1]["ESS"]
                ess_arr = ess.data
                self.assertEqual(ess_arr.shape, (2, 2, len(self.ys)))
                self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
                self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
                # all elements should be smaller than self.J and leq than 0
                self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr < self.J)))
            else:
                with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["ESS"]
            
            # filter_mean:
            if filter_mean:
                filt_mean = self.LG.results_history[-1]["filter_mean"]
                filter_mean_arr = filt_mean.data
                self.assertEqual(filter_mean_arr.shape, (2, 2, len(self.ys), 2))
                self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
                self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)) 
            else:
                with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["filter_mean"]

            # prediction_mean:
            if prediction_mean:
                pred_mean = self.LG.results_history[-1]["prediction_mean"]
                prediction_mean_arr = pred_mean.data
                self.assertEqual(prediction_mean_arr.shape, (2, 2, len(self.ys), 2))
                self.assertTrue(jnp.all(jnp.isfinite(prediction_mean_arr)))  
                self.assertTrue(jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)) 
            else:
                 with self.assertRaises(KeyError):
                    _ = self.LG.results_history[-1]["prediction_mean"]

if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)