Best Practices and Performance Guidelines
=========================================

To ensure that Pypomp operates at optimal speed and efficiency, it is important to follow a few key best practices.
Pypomp's primary advantage over the original R :code:`pomp` package (see the `R pomp website <https://kingaa.github.io/pomp/>`_ and `GitHub repository <https://github.com/kingaa/pomp>`_) is its performance, particularly on GPUs, and deviating from these guidelines may significantly degrade execution speed and thus negate the benefits of using Pypomp.

Essential Guidelines
--------------------

Using Pypomp Random Variate Samplers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When defining a model's random processes, always use the random variate samplers provided in ``pypomp.random`` instead of their equivalents in the `jax.random <https://jax.readthedocs.io/en/latest/jax.random.html>`_ module whenever possible.

The list of optimized samplers includes:

.. autosummary::
   :nosignatures:

   pypomp.random.fast_poisson
   pypomp.random.fast_binomial
   pypomp.random.fast_multinomial
   pypomp.random.fast_gamma
   pypomp.random.fast_nbinomial

Many ``jax.random`` functions utilize extensive branching logic, which ends up being executed sequentially on GPUs, severely impacting performance.
In contrast, the samplers in ``pypomp.random`` utilize approximate inverse cumulative distribution functions (CDFs) that avoid most of this branching logic.
Using the standard JAX samplers can make the code significantly slower than its R :code:`pomp` counterpart.

.. note::
   The functions ``jax.random.normal`` and ``jax.random.uniform`` are among the exceptions that are highly optimized; they are perfectly fine to use.

In addition, if a model requires sampling from a given distribution multiple times sequentially, it is significantly faster to concatenate the input parameters and make a single vectorized call to the sampling function, rather than invoking the function multiple separate times in a loop.

Vectorizing Across Replicates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some methods such as :meth:`~pypomp.core.pomp.Pomp.pfilter`, :meth:`~pypomp.core.pomp.Pomp.mif`, and :meth:`~pypomp.core.pomp.Pomp.train` accept a collection of parameter sets, ``theta``, as an argument and are designed to vectorize across the parameter sets so a GPU can process them in parallel.
Unless there are memory constraints, this approach is far faster than processing each parameter set sequentially.
Furthermore, the results of these vectorized operations are stored efficiently within the object, reducing both RAM and disk memory usage.
This also simplifies downstream analysis, as methods like :meth:`~pypomp.core.pomp.Pomp.results` can return a single tidy data frame encompassing all replicates.

Creating a separate :class:`~pypomp.core.pomp.Pomp` or :class:`~pypomp.panel.panel.PanelPomp` object for each parameter set negates these performance and structural advantages, so it is generally not recommended.

Optimizing CPU Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

When using JAX on CPUs, parallelizing across individual particles can be inefficient due to inter-core communication overhead (e.g., utilizing 36 cores may only be as fast as using 2 cores).
To resolve this, Pypomp methods like :meth:`~pypomp.core.pomp.Pomp.mif` and :meth:`~pypomp.core.pomp.Pomp.pfilter` automatically shard parameter sets across available devices.

For this optimization to take effect, you must manually set the number of JAX devices to match your available CPU cores **before** importing JAX.
You can do this by adding the following snippet at the beginning of your script:

.. code-block:: python

    import os
    # Set JAX platform before importing JAX
    os.environ["JAX_PLATFORMS"] = "cpu"
    cpus = 8
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={cpus}"
    )

The current implementation is experimental and subject to change.

As an alternative to the above, you can use the functions under ``pypomp.functional`` (see :doc:`api/functional`), which are flexible, JAX-compatible implementations of the core methods.
