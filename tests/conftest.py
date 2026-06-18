import os
import jax

# Configure JAX persistent compilation cache to avoid duplicate compilations
# across parallel xdist workers and subsequent pytest runs.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(project_root, ".pytest_cache", "jax_cache")

# Test writeability of the cache directory and only configure cache if successful
try:
    os.makedirs(cache_dir, exist_ok=True)
    test_file = os.path.join(cache_dir, ".write_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)

    # Update JAX configuration
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Cache all compilations, including fast ones (default is 1.0 second min compile time)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
except Exception:
    # If cache directory is not writeable (e.g. in sandbox environment), JAX caching is disabled.
    pass
