"""
Tests for Linear Gaussian (LG) model utility functions.
These functions are currently not tested but are core to the LG model functionality.
"""

import unittest
import jax
import jax.numpy as jnp
import pypomp as pp


class TestLGUtilityFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.LG = pp.LG()
        self.key = jax.random.key(42)
        
    def test_get_thetas_shape(self):
        """Test that get_thetas returns matrices of correct shape."""
        theta_vec = jnp.arange(16, dtype=float)
        A, C, Q, R = pp.LG.get_thetas(theta_vec)
        
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(C.shape, (2, 2)) 
        self.assertEqual(Q.shape, (2, 2))
        self.assertEqual(R.shape, (2, 2))
        
    def test_get_thetas_values(self):
        """Test that get_thetas correctly parses the theta vector."""
        theta_vec = jnp.arange(16, dtype=float)
        A, C, Q, R = pp.LG.get_thetas(theta_vec)
        
        # Check that matrices contain expected values
        jnp.testing.assert_array_equal(A, jnp.array([[0, 1], [2, 3]]))
        jnp.testing.assert_array_equal(C, jnp.array([[4, 5], [6, 7]]))
        jnp.testing.assert_array_equal(Q, jnp.array([[8, 9], [10, 11]]))
        jnp.testing.assert_array_equal(R, jnp.array([[12, 13], [14, 15]]))
        
    def test_transform_thetas_shape(self):
        """Test that transform_thetas returns correct vector length."""
        A = jnp.ones((2, 2))
        C = jnp.ones((2, 2))
        Q = jnp.ones((2, 2))
        R = jnp.ones((2, 2))
        
        result = pp.LG.transform_thetas(A, C, Q, R)
        self.assertEqual(result.shape, (16,))
        
    def test_get_transform_roundtrip(self):
        """Test that get_thetas and transform_thetas are inverse operations."""
        original_theta = jnp.arange(16, dtype=float)
        
        # Forward transformation
        A, C, Q, R = pp.LG.get_thetas(original_theta)
        
        # Backward transformation
        reconstructed_theta = pp.LG.transform_thetas(A, C, Q, R)
        
        # Should get back original vector
        jnp.testing.assert_array_almost_equal(original_theta, reconstructed_theta)
        
    def test_rinit_output_shape(self):
        """Test that rinit returns correct state dimension."""
        theta = self.LG.theta[0]
        initial_state = pp.LG.rinit(theta, self.key)
        
        # Should return 2D state vector for LG model
        self.assertEqual(initial_state.shape, (2,))
        
    def test_rinit_randomness(self):
        """Test that rinit produces different outputs with different keys."""
        theta = self.LG.theta[0]
        
        key1 = jax.random.key(1)
        key2 = jax.random.key(2)
        
        state1 = pp.LG.rinit(theta, key1)
        state2 = pp.LG.rinit(theta, key2)
        
        # Should produce different states (with high probability)
        self.assertFalse(jnp.allclose(state1, state2))
        
    def test_rproc_output_shape(self):
        """Test that rproc returns correct next state dimension."""
        theta = self.LG.theta[0]
        current_state = jnp.array([1.0, 2.0])
        
        next_state = pp.LG.rproc(current_state, theta, self.key)
        
        # Should return same dimension as input state
        self.assertEqual(next_state.shape, current_state.shape)
        
    def test_rproc_deterministic_component(self):
        """Test that rproc applies state transition correctly on average."""
        theta = self.LG.theta[0]
        current_state = jnp.array([1.0, 2.0])
        
        # Generate many samples to estimate mean
        keys = jax.random.split(self.key, 1000)
        samples = jax.vmap(lambda k: pp.LG.rproc(current_state, theta, k))(keys)
        mean_next_state = jnp.mean(samples, axis=0)
        
        # Mean should be approximately A @ current_state
        A, _, _, _ = pp.LG.get_thetas(jnp.array(list(theta.values())))
        expected_mean = A @ current_state
        
        jnp.testing.assert_allclose(mean_next_state, expected_mean, atol=0.1)
        
    def test_dmeas_output_shape(self):
        """Test that dmeas returns scalar log-probability."""
        theta = self.LG.theta[0]
        state = jnp.array([1.0, 2.0])
        observation = jnp.array([0.5, 1.5])
        
        log_prob = pp.LG.dmeas(observation, state, theta)
        
        # Should return scalar
        self.assertEqual(log_prob.shape, ())
        
    def test_dmeas_finite_output(self):
        """Test that dmeas returns finite log-probabilities for reasonable inputs."""
        theta = self.LG.theta[0]
        state = jnp.array([1.0, 2.0])
        observation = jnp.array([0.5, 1.5])
        
        log_prob = pp.LG.dmeas(observation, state, theta)
        
        # Should be finite (not inf or nan)
        self.assertTrue(jnp.isfinite(log_prob))
        
    def test_rmeas_output_shape(self):
        """Test that rmeas returns correct observation dimension."""
        theta = self.LG.theta[0]
        state = jnp.array([1.0, 2.0])
        
        observation = pp.LG.rmeas(state, theta, self.key)
        
        # Should return same dimension as state for LG model
        self.assertEqual(observation.shape, state.shape)
        
    def test_rmeas_randomness(self):
        """Test that rmeas produces different observations with different keys."""
        theta = self.LG.theta[0]
        state = jnp.array([1.0, 2.0])
        
        key1 = jax.random.key(1)
        key2 = jax.random.key(2)
        
        obs1 = pp.LG.rmeas(state, theta, key1)
        obs2 = pp.LG.rmeas(state, theta, key2)
        
        # Should produce different observations (with high probability)
        self.assertFalse(jnp.allclose(obs1, obs2))


class TestLGEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for LG model."""
    
    def test_get_thetas_wrong_length(self):
        """Test error handling for incorrect theta vector length."""
        # Test with vector that's too short
        with self.assertRaises((ValueError, IndexError)):
            pp.LG.get_thetas(jnp.array([1, 2, 3]))  # Only 3 elements instead of 16
            
        # Test with vector that's too long
        with self.assertRaises((ValueError, IndexError)):
            pp.LG.get_thetas(jnp.arange(20))  # 20 elements instead of 16
            
    def test_transform_thetas_wrong_shapes(self):
        """Test error handling for incorrect matrix shapes."""
        A = jnp.ones((3, 2))  # Wrong shape
        C = jnp.ones((2, 2))
        Q = jnp.ones((2, 2))
        R = jnp.ones((2, 2))
        
        with self.assertRaises((ValueError, TypeError)):
            pp.LG.transform_thetas(A, C, Q, R)


if __name__ == "__main__":
    unittest.main(verbosity=2)