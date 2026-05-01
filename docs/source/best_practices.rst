Best Practices and Performance Guidelines
=========================================

To ensure that **pypomp** operates at optimal speed and efficiency, it is important to follow a few key best practices. 
Pypomp's primary advantage is its performance, particularly on GPUs, and deviating from these guidelines may significantly degrade execution speed.

Essential Guidelines
--------------------

Using pypomp Random Variate Samplers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When defining your model's random processes, always use the random variate samplers provided in ``pypomp.random`` instead of their equivalents in ``jax.random`` whenever possible. 

The list of optimized samplers includes:

.. autosummary::
   :nosignatures:

   pypomp.random.fast_poisson
   pypomp.random.fast_binomial
   pypomp.random.fast_multinomial
   pypomp.random.fast_gamma
   pypomp.random.fast_nbinomial

Many ``jax.random`` functions utilize extensive branching logic, which must be executed sequentially on GPUs, severely impacting performance. 
In contrast, the samplers in ``pypomp.random`` utilize approximate inverse cumulative distribution functions (CDFs) that avoid most of this branching logic. 
Using the standard JAX samplers can result in execution speeds up to 25 times slower, potentially making the code slower than its R ``pomp`` counterpart. 

.. note::
   The functions ``jax.random.normal`` and ``jax.random.uniform`` are among the exceptions that are highly optimized; they are perfectly fine to use.

In addition, if your model requires sampling from a given distribution multiple times sequentially, it is significantly faster to concatenate your input parameters and make a single vectorized call to the sampling function, rather than invoking the function multiple separate times in a loop.

Vectorizing Across Replicates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some methods such as :meth:`pypomp.Pomp.pfilter`, :meth:`pypomp.Pomp.mif`, and :meth:`pypomp.Pomp.train` accept a collection of parameter sets, ``theta``, as an argument and are designed to process the collection simultaneously. 

By passing multiple parameter sets directly into ``theta``, these methods automatically vectorize operations across the replicates. 
This can result in execution speeds up to `R` times faster, where `R` is the number of parameter sets passed in ``theta``. 
Furthermore, the results of these vectorized operations are stored efficiently within the object, reducing both RAM and disk memory usage. 
This also simplifies downstream analysis, as methods like :meth:`pypomp.Pomp.results` can return a single tidy data frame encompassing all replicates.

Creating a separate :class:`pypomp.Pomp` or :class:`pypomp.PanelPomp` object for each parameter set negates these performance and structural advantages, so it is generally not recommended.

Optimizing CPU Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

When using JAX on CPUs, parallelizing across individual particles can be inefficient due to inter-core communication overhead (e.g., utilizing 36 cores may only be as fast as using 2 cores). 
To resolve this, **pypomp** functions like :meth:`pypomp.Pomp.mif` and :meth:`pypomp.Pomp.pfilter` are optimized to force parallelization across replications instead.

For this optimization to take effect, you must manually set the number of JAX devices to match your available CPU cores **before** importing JAX. 
You can do this by adding the following snippet at the beginning of your script:

.. code-block:: python

    import os

    # Set JAX platform before importing JAX
    USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
    if USE_CPU:
        os.environ["JAX_PLATFORMS"] = "cpu"
        if "SLURM_CPUS_PER_TASK" in os.environ:
            os.environ["XLA_FLAGS"] = (
                os.environ.get("XLA_FLAGS", "")
                + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
            )

The current implementation is experimental and subject to change.
