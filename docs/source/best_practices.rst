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


Avoiding 2-D Temporaries Under IF2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under :meth:`~pypomp.core.pomp.Pomp.mif`, parameters are perturbed per particle and arrive as ``(J,)`` arrays rather than scalars.
Any intermediate built inside an ``rproc`` by combining several state variables or parameters into an array therefore becomes a genuine 2-D buffer, ``(k, J)``, rebuilt on every Euler sub-step.
On JAX >= 0.10.2, the compiler no longer fuses that pattern away, which can cause significant slowdowns (2x or more).
The same code under :meth:`~pypomp.core.pomp.Pomp.pfilter` is free because parameters there are scalars.

This applies to both default (vmapped) and :func:`~pypomp.core.model_mechanics.vectorized` process functions.

**The rule:** inside an ``rproc``, never allocate a buffer whose length is a compile-time constant. Use that many named variables instead.

Concretely:

.. code-block:: python

    # Slow under mif (creates 2-D buffers and scatter targets)
    pts = jnp.stack([X_["R1"], X_["R2"], X_["R3"]], axis=0)
    passages = jnp.zeros((4, J))
    passages = passages.at[0].set(gamma * I)
    count = count + jnp.any(jnp.stack([S, I, Y]) < 0, axis=0)

    # Fast (elementwise operations on named variables)
    q0, q1, q2 = X_["R1"], X_["R2"], X_["R3"]
    pass0 = gamma * I
    count = count + ((S < 0) | (I < 0) | (Y < 0))

Key patterns to avoid:

- Scatter targets like ``passages.at[i].set(...)`` inside the time-stepping loop.
- Array reductions across states/parameters like ``jnp.any(jnp.stack(...), axis=0)``.
- Intermediate state arrays like ``pts = jnp.stack([R1, R2, R3])`` that are updated or reduced across.


.. _cute-backend:

The CuTe Backend
^^^^^^^^^^^^^^^^

On an NVIDIA GPU, the CuTe backend translates ``rproc``, and ``dmeas`` where it can, into one CUDA kernel per observation interval.
Each thread advances one particle through every Euler step of the interval, for all the filters of a call at once, so the particles' states stay in registers instead of passing through GPU memory at each step.
It is opt-in: a new model uses XLA until you choose CuTe with :meth:`~pypomp.core.pomp.Pomp.set_backend`.

.. code-block:: python

    model.set_backend("cute")  # warns and uses XLA if CuTe can't run the model
    model.set_backend("auto")  # CuTe where it can run, XLA otherwise, quietly

CuTe needs the optional ``nvidia-cutlass-dsl`` package (``pip install pypomp[cute]``), an NVIDIA GPU from Ampere on, float32 particles and JAX's default threefry keys.
Where any of these is missing, the model runs on XLA.

**When it is faster.**
The kernel pays off most for models with many Euler steps per observation and with the branching samplers of ``pypomp.random`` (binomial, Poisson), whose branches the kernel takes per particle instead of computing every branch for every particle as XLA does.
For the UK measles model (``pypomp.models.UKMeasles``, model ``001b``), advancing 1296 filters of 5000 particles over one observation interval takes about 11 ms on an A40 against 69 ms on XLA, and 3.6 ms on an RTX Pro 6000 against 10.5 ms.
For a larger panel model on the RTX Pro 6000, :meth:`~pypomp.panel.panel.PanelPomp.pfilter` ran 4.7 times and :meth:`~pypomp.panel.panel.PanelPomp.mif` 2.7 times faster.
The work between intervals, such as normalizing the weights and deciding whether to resample, gains less, so a whole filter speeds up less than its process model.

**Startup cost.**
The first call after :meth:`~pypomp.core.pomp.Pomp.set_backend` translates the model, compiles the kernels and checks them against XLA, which takes tens of seconds: for the UK measles model on an A40, the first filter or iterated filtering call took 30 to 50 s longer than on XLA.
The kernels belong to the model object: copying it (``deepcopy``, pickling) builds them again, and nothing is cached between Python sessions.
For a short exploratory run, XLA can finish first; CuTe is for long runs such as iterated filtering searches and many filter replicates.

**How results differ from XLA's.**
With ``rng="threefry"``, the default, the kernel draws the same random numbers as XLA for the same key, so the two backends agree up to float rounding.
Rounding can still flip a resampling decision now and then, which moves a log-likelihood by about its Monte Carlo error, so results are not identical across backends.
``rng="philox"`` draws pypomp's own Philox stream instead, which takes about 20% less kernel time for the UK measles model but gives different numbers from XLA's for the same key.

**What can go wrong.**

- The check runs the first observation interval on the kernel and on XLA at ``model.theta``, once, and keeps XLA (with a warning) if they differ. A kernel that differs from XLA only at other parameters goes unnoticed. ``set_backend(..., check=False)`` skips the check.
- The translator handles most of what an ``rproc`` written for one particle uses, but not everything: while loops with a runtime trip count, dynamic updates at a runtime index and float64, among others, keep the model on XLA. A ``@vectorized`` ``rproc`` that couples particles, for example through a mean over them, stays on XLA too.
- :meth:`~pypomp.core.pomp.Pomp.train` differentiates through a second generated kernel. Optimizers that need Hessians (:class:`~pypomp.Newton`, :class:`~pypomp.WeightedNewton`) run on XLA.
- A :class:`~pypomp.panel.panel.PanelPomp` runs its first unit's kernel for every unit, checked on the first unit.
- On several GPUs, :meth:`~pypomp.core.pomp.Pomp.pfilter`, :meth:`~pypomp.core.pomp.Pomp.mif` and :meth:`~pypomp.core.pomp.Pomp.train` spread the parameter sets over the GPUs, and each GPU launches the kernel for its own. With ``pypomp.functional``, run the call under ``jax.set_mesh(mesh)`` with the parameter sets sharded over the mesh for the same effect; otherwise every GPU runs every filter.

**Checking a model.**
:meth:`~pypomp.core.pomp.Pomp.backend_report` says which parts run on the kernel, and names each operation that keeps a part off it with the line of the model's code that uses it:

.. code-block:: python

    print(model.backend_report(backend="cute"))

It runs on any machine, including one without a GPU or ``nvidia-cutlass-dsl``, where it says which parts the translator would accept.

**Writing for the kernel.**
The kernel runs ``rproc`` for one particle, so a ``lax.cond`` in it is a real branch that computes one side; ``jnp.where`` still computes both.
The samplers in ``pypomp.random`` are written this way, which is one more reason to prefer them.


Optimizing CPU Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

When using JAX on CPUs, parallelizing across individual particles can be inefficient due to inter-core communication overhead (e.g., utilizing 36 cores may only be as fast as using 2 cores).
To resolve this, Pypomp methods like :meth:`~pypomp.core.pomp.Pomp.mif` and :meth:`~pypomp.core.pomp.Pomp.pfilter` automatically shard parameter sets across available devices.

.. note::
    It is specifically the *parameter sets* that are sharded, so the speedup scales with how many of them you pass in ``theta``.
    The ``reps`` argument is evaluated sequentially (via ``jax.lax.map``) underneath the sharding to conserve memory and does not spread work across devices: eight parameter sets with ``reps=1`` parallelizes across eight devices, whereas a single parameter set with ``reps=8`` runs on one.
    Passing fewer parameter sets than there are devices simply leaves the surplus devices idle.

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
