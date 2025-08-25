import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG

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
    
    def test_diagnostics(self):
        # (theta, reps, expected shape)
        theta_cases = [
            (self.theta[0], 1, (1, 1)),  
            (
                [self.theta[0], {k: v * 2 for k, v in self.theta[0].items()}],
                2,
                (2, 2),                   
            ),
        ]

        bool_cases = [
            (False, False, False, False),
            (True, True, False, False),
            (False, False, True, True)
        ]

        for theta, reps, expected_shape in theta_cases:
            for CLL, ESS, filter_mean, prediction_mean in bool_cases:
                with self.subTest(theta=theta, 
                                  reps=reps, 
                                  CLL=CLL, 
                                  ESS=ESS, 
                                  filter_mean=filter_mean, 
                                  prediction_mean=prediction_mean):
                    self.LG.results_history.clear()
                    self.LG.pfilter(J=self.J, key=self.key, theta=theta, reps=reps,
                                    CLL=CLL, ESS=ESS, filter_mean=filter_mean, prediction_mean=prediction_mean)
                    method = self.LG.results_history[-1]["method"]
                    self.assertEqual(method, "pfilter")
                    negLogLiks = self.LG.results_history[-1]["logLiks"]
                    negLogLiks_arr = negLogLiks.data
                    self.assertEqual(negLogLiks_arr.shape, expected_shape)
                    self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
                    self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 

                    # CLL:
                    if CLL:
                        condLogLiks = self.LG.results_history[-1]["CLL"]
                        condLogLiks_arr = condLogLiks.data
                        self.assertEqual(condLogLiks_arr.shape, expected_shape + (len(self.ys),))
                        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
                        self.assertTrue(jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)) 
                    else:
                        with self.assertRaises(KeyError):
                            _ = self.LG.results_history[-1]["CLL"]
                    
                    # ESS:
                    if ESS:
                        ess = self.LG.results_history[-1]["ESS"]
                        ess_arr = ess.data
                        self.assertEqual(ess_arr.shape, expected_shape + (len(self.ys),))
                        self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
                        self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
                        # all elements should be  between 0 and self.J inclusive
                        self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr <= self.J)))
                    else:
                        with self.assertRaises(KeyError):
                            _ = self.LG.results_history[-1]["ESS"]
                    
                    # filter_mean:
                    if filter_mean:
                        filt_mean = self.LG.results_history[-1]["filter_mean"]
                        filter_mean_arr = filt_mean.data
                        self.assertEqual(filter_mean_arr.shape, expected_shape + (len(self.ys), 2))
                        self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
                        self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating))
                    else:
                        with self.assertRaises(KeyError):
                            _ = self.LG.results_history[-1]["filter_mean"]  
                    
                    # prediction_mean:
                    if prediction_mean: 
                        pred_mean = self.LG.results_history[-1]["prediction_mean"]
                        prediction_mean_arr = pred_mean.data
                        self.assertEqual(prediction_mean_arr.shape, expected_shape + (len(self.ys), 2))
                        self.assertTrue(jnp.all(jnp.isfinite(prediction_mean_arr)))  
                        self.assertTrue(jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)) 
                    else:
                        with self.assertRaises(KeyError):
                            _ = self.LG.results_history[-1]["prediction_mean"]

if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)