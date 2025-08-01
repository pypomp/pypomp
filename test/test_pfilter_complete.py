import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG

class TestPfilterComplete_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG() # LG_obj with T = 4
        self.key = jax.random.key(111)
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_class_basic(self):
        self.LG.pfilter_complete(J=self.J, key=self.key)
        method = self.LG.results_history[-1]["method"]
        self.assertEqual(method, "pfilter_complete")
        
        negLogLiks = self.LG.results_history[-1]["negLogLiks"]
        negLogLiks_arr = negLogLiks.data
        self.assertEqual(negLogLiks_arr.shape, (1, 1))
        self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(negLogLiks_arr.dtype, jnp.float32)

        meanLogLiks = self.LG.results_history[-1]["meanLogLiks"]
        meanLogLiks_arr = meanLogLiks.data
        self.assertEqual(meanLogLiks_arr.shape, (1, 1))
        self.assertTrue(jnp.all(jnp.isfinite(meanLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(meanLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(meanLogLiks_arr.dtype, jnp.float32)

        condLogLiks = self.LG.results_history[-1]["condLogLiks"]
        condLogLiks_arr = condLogLiks.data
        self.assertEqual(condLogLiks_arr.shape, (1, 1, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(meanLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(meanLogLiks_arr.dtype, jnp.float32)

        particles = self.LG.results_history[-1]["particles"]
        particles_arr = particles.data
        self.assertEqual(particles_arr.shape, (1, 1, len(self.ys), self.J, 2))
        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(meanLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(meanLogLiks_arr.dtype, jnp.float32)

        filter_mean = self.LG.results_history[-1]["filter_mean"]
        filter_mean_arr = filter_mean.data
        self.assertEqual(filter_mean_arr.shape, (1, 1, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
        self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)) 
        self.assertEqual(filter_mean_arr.dtype, jnp.float32)

        ess = self.LG.results_history[-1]["ess"]
        ess_arr = ess.data
        self.assertEqual(ess_arr.shape, (1, 1, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
        self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
        self.assertEqual(ess_arr.dtype, jnp.float32)
        # all elements should be smaller than self.J and leq than 0
        self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr < self.J)))

        filt_traj = self.LG.results_history[-1]["filt_traj"]
        filt_traj_arr = filt_traj.data
        self.assertEqual(filt_traj_arr.shape, (1, 1, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(filt_traj_arr)))  
        self.assertTrue(jnp.issubdtype(filt_traj_arr.dtype, jnp.floating)) 
        self.assertEqual(filt_traj_arr.dtype, jnp.float32)

    def test_reps(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]
        self.LG.pfilter_complete(J=self.J, key=self.key, reps=2, theta=theta_list)
        method = self.LG.results_history[-1]["method"]
        self.assertEqual(method, "pfilter_complete")
        
        negLogLiks = self.LG.results_history[-1]["negLogLiks"]
        negLogLiks_arr = negLogLiks.data
        self.assertEqual(negLogLiks_arr.shape, (2, 2))
        self.assertTrue(jnp.all(jnp.isfinite(negLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(negLogLiks_arr.dtype, jnp.float32)

        meanLogLiks = self.LG.results_history[-1]["meanLogLiks"]
        meanLogLiks_arr = meanLogLiks.data
        self.assertEqual(meanLogLiks_arr.shape, (2,2))
        self.assertTrue(jnp.all(jnp.isfinite(meanLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(meanLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(meanLogLiks_arr.dtype, jnp.float32)

        condLogLiks = self.LG.results_history[-1]["condLogLiks"]
        condLogLiks_arr = condLogLiks.data
        self.assertEqual(condLogLiks_arr.shape, (2,2, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(meanLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(meanLogLiks_arr.dtype, jnp.float32)

        particles = self.LG.results_history[-1]["particles"]
        particles_arr = particles.data
        self.assertEqual(particles_arr.shape, (2, 2, len(self.ys), self.J, 2))
        self.assertTrue(jnp.all(jnp.isfinite(condLogLiks_arr)))  
        self.assertTrue(jnp.issubdtype(meanLogLiks_arr.dtype, jnp.floating)) 
        self.assertEqual(meanLogLiks_arr.dtype, jnp.float32)

        filter_mean = self.LG.results_history[-1]["filter_mean"]
        filter_mean_arr = filter_mean.data
        self.assertEqual(filter_mean_arr.shape, (2, 2, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(filter_mean_arr)))  
        self.assertTrue(jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)) 
        self.assertEqual(filter_mean_arr.dtype, jnp.float32)

        ess = self.LG.results_history[-1]["ess"]
        ess_arr = ess.data
        self.assertEqual(ess_arr.shape, (2, 2, len(self.ys)))
        self.assertTrue(jnp.all(jnp.isfinite(ess_arr)))  
        self.assertTrue(jnp.issubdtype(ess_arr.dtype, jnp.floating)) 
        self.assertEqual(ess_arr.dtype, jnp.float32)
        # all elements should be smaller than self.J and leq than 0
        self.assertTrue(jnp.all((ess_arr >= 0) & (ess_arr < self.J)))

        filt_traj = self.LG.results_history[-1]["filt_traj"]
        filt_traj_arr = filt_traj.data
        self.assertEqual(filt_traj_arr.shape, (2, 2, len(self.ys), 2))
        self.assertTrue(jnp.all(jnp.isfinite(filt_traj_arr)))  
        self.assertTrue(jnp.issubdtype(filt_traj_arr.dtype, jnp.floating)) 
        self.assertEqual(filt_traj_arr.dtype, jnp.float32)

        if __name__ == "__main__":
           unittest.main(argv=[""], verbosity=2, exit=False)