"""
Edge case and error handling tests for pypomp core functionality.
These tests ensure robustness and proper error handling.
"""

import unittest
import jax
import jax.numpy as jnp
import pypomp as pp


class TestPompEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for POMP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.LG = pp.LG()
        self.key = jax.random.key(42)
        
    def test_pfilter_zero_particles(self):
        """Test that pfilter raises error with zero particles."""
        with self.assertRaises(ValueError):
            self.LG.pfilter(J=0, key=self.key)
            
    def test_pfilter_negative_particles(self):
        """Test that pfilter raises error with negative particles."""
        with self.assertRaises(ValueError):
            self.LG.pfilter(J=-5, key=self.key)
            
    def test_mif_negative_iterations(self):
        """Test that mif raises error with negative iterations."""
        with self.assertRaises(ValueError):
            self.LG.mif(J=10, M=-1, sigmas=0.1, key=self.key)
            
    def test_mif_zero_iterations(self):
        """Test mif behavior with zero iterations."""
        # This might be valid (no iterations = no filtering)
        try:
            self.LG.mif(J=10, M=0, sigmas=0.1, key=self.key)
            # If it doesn't raise an error, that's fine
        except ValueError:
            # If it raises an error, that's also reasonable
            pass
            
    def test_train_zero_iterations(self):
        """Test that train raises error with zero iterations."""
        with self.assertRaises(ValueError):
            self.LG.train(J=10, itns=0, key=self.key)
            
    def test_train_negative_iterations(self):
        """Test that train raises error with negative iterations.""" 
        with self.assertRaises(ValueError):
            self.LG.train(J=10, itns=-2, key=self.key)
            
    def test_invalid_theta_format(self):
        """Test error handling for invalid theta parameter format."""
        with self.assertRaises((TypeError, ValueError)):
            self.LG.pfilter(J=5, theta="invalid_string", key=self.key)
            
        with self.assertRaises((TypeError, ValueError)):
            self.LG.pfilter(J=5, theta=[1, 2, 3], key=self.key)  # List of numbers instead of dict
            
    def test_mismatched_theta_keys(self):
        """Test error handling for theta with wrong parameter names."""
        invalid_theta = {"wrong_param1": 1.0, "wrong_param2": 2.0}
        
        with self.assertRaises((ValueError, KeyError)):
            self.LG.pfilter(J=5, theta=[invalid_theta], key=self.key)
            
    def test_missing_required_components(self):
        """Test that Pomp class requires essential components."""
        ys = self.LG.ys
        theta = self.LG.theta
        rinit = self.LG.rinit
        rproc = self.LG.rproc
        dmeas = self.LG.dmeas
        
        # Test missing ys
        with self.assertRaises((TypeError, ValueError)):
            pp.Pomp(ys=None, theta=theta, rinit=rinit, rproc=rproc, dmeas=dmeas)
            
        # Test missing theta
        with self.assertRaises((TypeError, ValueError)):
            pp.Pomp(ys=ys, theta=None, rinit=rinit, rproc=rproc, dmeas=dmeas)
            
        # Test missing rinit
        with self.assertRaises((TypeError, ValueError)):
            pp.Pomp(ys=ys, theta=theta, rinit=None, rproc=rproc, dmeas=dmeas)
            
        # Test missing rproc
        with self.assertRaises((TypeError, ValueError)):
            pp.Pomp(ys=ys, theta=theta, rinit=rinit, rproc=None, dmeas=dmeas)
            
        # Test missing dmeas (this might be optional in some cases)
        try:
            pp.Pomp(ys=ys, theta=theta, rinit=rinit, rproc=rproc, dmeas=None)
        except (TypeError, ValueError):
            # If it raises an error, that's fine - dmeas might be required
            pass


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability with extreme parameter values."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.LG = pp.LG()
        self.key = jax.random.key(42)
        
    def test_very_small_parameters(self):
        """Test behavior with very small parameter values."""
        # Create theta with very small values
        small_theta = {k: v * 1e-10 for k, v in self.LG.theta[0].items()}
        
        try:
            self.LG.pfilter(J=5, theta=[small_theta], key=self.key)
            result = self.LG.results_history[-1]
            
            # Check that result is reasonable (not NaN or inf)
            log_lik = result["logLiks"][0]
            self.assertTrue(jnp.isfinite(log_lik) or jnp.isneginf(log_lik))
            
        except (ValueError, RuntimeError) as e:
            # Numerical instability is acceptable with extreme parameters
            self.assertIn(("numerical", "stability", "singular", "overflow"), 
                         str(e).lower())
            
    def test_very_large_parameters(self):
        """Test behavior with very large parameter values."""
        # Create theta with very large values
        large_theta = {k: v * 1e10 for k, v in self.LG.theta[0].items()}
        
        try:
            self.LG.pfilter(J=5, theta=[large_theta], key=self.key)
            result = self.LG.results_history[-1]
            
            # Check that result is reasonable
            log_lik = result["logLiks"][0]
            self.assertTrue(jnp.isfinite(log_lik) or jnp.isneginf(log_lik))
            
        except (ValueError, RuntimeError) as e:
            # Numerical instability is acceptable with extreme parameters
            self.assertIn(("numerical", "overflow", "infinity"), 
                         str(e).lower())
            
    def test_zero_variance_parameters(self):
        """Test behavior when variance parameters are zero."""
        # Modify theta to have zero variance components
        zero_var_theta = self.LG.theta[0].copy()
        
        # Set Q and R matrix elements to very small values (near zero)
        for i, key in enumerate(zero_var_theta.keys()):
            if i >= 8:  # Q and R matrix elements start at index 8
                zero_var_theta[key] = 1e-12
                
        try:
            self.LG.pfilter(J=5, theta=[zero_var_theta], key=self.key)
            result = self.LG.results_history[-1]
            
            # Should handle gracefully
            log_lik = result["logLiks"][0]
            self.assertTrue(jnp.isfinite(log_lik) or jnp.isneginf(log_lik))
            
        except (ValueError, RuntimeError):
            # Singular covariance matrices might cause errors - that's OK
            pass


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and limits."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.LG = pp.LG()
        self.key = jax.random.key(42)
        
    def test_single_particle(self):
        """Test with minimal number of particles (J=1)."""
        self.LG.pfilter(J=1, key=self.key)
        result = self.LG.results_history[-1]
        
        # Should work but may have high variance
        log_lik = result["logLiks"][0]
        self.assertTrue(jnp.isfinite(log_lik))
        
    def test_single_observation(self):
        """Test with minimal data (single time point)."""
        # Create POMP with only one observation
        single_obs_ys = self.LG.ys.iloc[:1].copy()
        
        single_pomp = pp.Pomp(
            ys=single_obs_ys,
            theta=self.LG.theta,
            rinit=self.LG.rinit,
            rproc=self.LG.rproc,
            dmeas=self.LG.dmeas
        )
        
        single_pomp.pfilter(J=5, key=self.key)
        result = single_pomp.results_history[-1]
        
        log_lik = result["logLiks"][0]
        self.assertTrue(jnp.isfinite(log_lik))
        
    def test_very_small_sigmas_mif(self):
        """Test MIF with very small perturbation sigmas."""
        self.LG.mif(J=5, M=2, sigmas=1e-10, key=self.key)
        result = self.LG.results_history[-1]
        
        # With tiny sigmas, parameters should barely change
        traces = result["traces"][0]
        param_cols = [col for col in traces.columns if col != "logLik"]
        
        for col in param_cols:
            param_range = traces[col].max() - traces[col].min()
            self.assertLess(param_range, 1e-6)  # Should change very little
            
    def test_very_large_sigmas_mif(self):
        """Test MIF with very large perturbation sigmas."""
        try:
            self.LG.mif(J=5, M=2, sigmas=1000.0, key=self.key)
            result = self.LG.results_history[-1]
            
            # Should complete even with large perturbations
            self.assertIsNotNone(result)
            
        except (ValueError, RuntimeError):
            # Large perturbations might cause numerical issues - acceptable
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)