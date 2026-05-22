import os
import jax

# Configure JAX persistent compilation cache to avoid duplicate compilations
# across parallel xdist workers and subsequent pytest runs.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(project_root, ".pytest_cache", "jax_cache")

# Create the cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", cache_dir)

# Cache all compilations, including fast ones (default is 1.0 second min compile time)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
