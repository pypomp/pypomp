import os

import jax

# Configure JAX persistent compilation cache to avoid duplicate compilations
# across parallel xdist workers and subsequent pytest runs.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(project_root, ".pytest_cache", "jax_cache")

# Test writeability of the cache directory and only configure cache if successful
try:
    if os.environ.get("DISABLE_JAX_CACHE") != "1":
        os.makedirs(cache_dir, exist_ok=True)
        test_file = os.path.join(cache_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        # Update JAX configuration
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache all compilations, including fast ones (default is 1.0 second min compile time)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # Bound the cache size to enable LRU eviction. This also makes JAX's
        # LRUCache take a real file lock around every read/write; without it
        # (max_size == -1, the default), concurrent xdist workers can write
        # the same cache entry at once with no locking and corrupt it,
        # producing spurious "Error reading persistent compilation cache
        # entry ... ZstdError" warnings.
        jax.config.update("jax_compilation_cache_max_size", 5 * 1024**3)
except Exception:
    # If cache directory is not writeable (e.g. in sandbox environment), JAX caching is disabled.
    pass
