Computation archives
====================

``bake`` and ``stew`` evaluate trusted Python source blocks and save their
results locally, following the computation-archiving workflow of R pomp.
The first sections describe the native default; the explicit R mode is
documented below. An existing archive is reused only when the expression, explicitly declared
dependencies, and seed match. A hit does not execute the block again.

Both functions are implemented in ``pypomp/bake.py``, with the explicit R
adapters in ``pypomp/_bake_compat.py``. The public calls remain ``pypomp.bake``
and ``pypomp.stew``. Compatibility import aliases retain the earlier module
names when loading existing Python pickles; the on-disk archive format is unchanged.

Use ``file`` for the archive path and ``expr`` for trusted Python source text.
``dependson`` records explicitly declared inputs, while ``envir`` supplies an
explicit workspace dictionary. ``dir`` selects the archive directory,
``seed`` controls reproducible draws, and ``info`` / ``timing`` select returned
metadata. ``compatibility="R"`` enables the documented R behavior;
``kind`` and ``normal_kind`` accept the supported R generator metadata.

.. autofunction:: pypomp.bake

.. autofunction:: pypomp.stew

One value
---------

At script or notebook top level, the caller's workspace is used automatically::

    import pypomp as pp

    a = 2
    value = pp.bake("archives/value.pkl", "a * 10", dependson=a)
    assert value == 20

The same call restores the value. Changing ``a`` invalidates this archive
because it is an explicit dependency. Statements can precede the final
expression. A ``None`` result warns and returns an empty list, as R bake does
for ``NULL``. Python assignments have no value: an empty block or a final
assignment also produces this warning. End with ``x`` to return an assigned
value, for example ``"x = 5\nx"``.

Named outputs
-------------

``stew`` saves assignments automatically, restores visible names into the
workspace, and returns their sorted names::

    names = pp.stew("archives/bundle.pkl", """
        x = [1, 2, 3]
        y = [2 * item for item in x]
        _scratch = 99
    """)
    assert names == ["x", "y"]
    assert y == [2, 4, 6]

The block's final value is ignored. Underscore-prefixed local names are saved
but not copied back as ordinary outputs. Names already present in the parent
workspace are not saved merely because the block reads them; rebinding a name
creates a local output. Mutating a referenced object can still affect the
parent object, and those side effects are not replayed on a hit.

Inside a function or class, pass a dictionary explicitly::

    def experiment(a):
        workspace = {"a": a}
        pp.stew("archives/scaled.pkl", "x = a * 10",
                dependson=a, envir=workspace)
        return workspace["x"]

Restoring into a dictionary does not create optimized Python local variables;
use the dictionary to read outputs. ``expr`` is source text, not a callable,
and executes with ordinary Python privileges. Do not accept it from untrusted
input or assume that the child namespace is a security sandbox.

Reproducible JAX computations
-----------------------------

An integer ``seed`` temporarily supplies ``_key``, a Threefry JAX key::

    import jax

    draw = pp.bake("archives/draw.pkl",
                   "jax.random.normal(_key, (1000,))", seed=123)

Use ``jax.random.split`` for independent draws within a block. With no seed,
no key is installed; a block can instead use an explicit key already in its
workspace, declaring it as a dependency when it should invalidate the cache.
The seed does not set NumPy's or Python's global random generators. A seeded
call restores or removes its temporary ``_key`` binding even if evaluation fails.

For object-oriented pypomp work, remember that inference methods mutate the
model and return ``None``. Return the model as the final expression in bake,
or assign it locally in stew::

    fit = pp.bake("archives/fit.pkl", """
        model = build_model()
        model.pfilter(J=J, key=_key)
        model
    """, seed=123,
        dependson={"J": J, "data": data_digest, "model": model_revision})

Use a new model or an intentional copy inside the block if the existing model
must remain unchanged. Full-model loading uses pypomp's existing pickle hooks;
it does not repair an unsupported model serialization or preserve compiled JIT
executables across processes.
In the current pypomp model hooks, parameter transforms must be importable
module-level functions: lambda/local transforms can reload as the identity
transform. Check transform behavior before relying on a full-model archive.

Dependencies and metadata
-------------------------

Expression hashing ignores whitespace, comments, and source positions.
External data, imported helper implementations, configuration and runtime
changes are tracked only when explicitly included in ``dependson``. A path
string tracks the path, not the file contents; supply the data or a content
digest when necessary. An imported function may pickle by reference, so
include its source digest or a stable revision to track body changes.

Serialization is not canonical mathematical equality: equivalent custom
objects can produce different hashes. Prefer deliberate data/configuration
dependencies to a live model with changing keys and history.

Both timing and ingredients are always archived. ``info=True`` exposes
``_ingredients`` in the workspace; ``timing=True`` (the default) exposes
``_system_time`` there. Disabling a flag does not remove metadata left by an
earlier call. These names, ``_key``, and ``__builtins__`` are reserved.

``_system_time`` contains ``user.self``, ``sys.self``, ``elapsed``,
``user.child``, and ``sys.child`` in seconds. It measures evaluation and
completion of ordinary JAX output pytrees, excluding serialization and file
I/O. Opaque asynchronous objects must be synchronized explicitly in the
block. On a hit the timing describes the original computation, not load time.
Synchronization visits each object once and handles cyclic containers,
mixed dictionary key types, and registered pytree children without recursively
flattening the entire object graph.

Explicit R compatibility
------------------------

Use ``compatibility="R"`` when the R recipe contract is required. The default
remains ``"native"``. For example::

    with pp.archive_directory("r-cache"):
        uniforms = pp.bake(
            "uniform.pkl", "_rng.uniform(size=4)", seed=7,
            compatibility="R", envir={}, timing=False,
        )
        details = pp.bake(
            "uniform.pkl", "_rng.uniform(size=4)", seed=7,
            compatibility="R", envir={}, info=True,
        )
    assert uniforms.shape == (4,)
    assert (uniforms == details.value).all()
    assert len(details.ingredients) == 5

``archive_directory`` is a context manager that restores the previous option,
including after an exception. ``dir`` overrides it. R mode concatenates paths
like R's ``file.path``; for an absolute file path, pass ``dir=""``. It emits a
message when creating a directory, creates files under the current umask, and
writes through symlinks. **R mode writes directly and can leave partial files
on failure. It restores seeded RNG state only after successful evaluation.**
These reproduce the pinned R implementation; native mode keeps atomic writes
and exception-safe key cleanup.

R-mode bake returns ``ArchiveValue`` when either metadata flag is enabled.
Use ``.value`` for the original payload, ``.ingredients`` and ``.system_time``
for selected attributes. With both flags false it returns the payload directly.
Bake does not overwrite caller metadata. Stew exposes metadata in the workspace.
Five ingredients track source, dependencies, typed seed, ``kind`` and
``normal_kind``. R's ``normal.kind`` spelling maps to ``normal_kind``. Seed
metadata retains Python type and serialized value for exact invalidation.
Recognized Python legacy bake records with timing metadata and legacy stew
binding dictionaries are upgraded; **RDS/RData files are not decoded**.

``freeze`` evaluates without an archive and preserves a ``None`` result::

    workspace = {}
    frozen_value = pp.freeze("x=5", compatibility="R", envir=workspace)
    assert frozen_value == workspace["x"] == 5
    assert pp.freeze("None", compatibility="R", envir={}) is None

R-mode ``seed`` uses the first integer-coerced element, accepting negative
signed integers, booleans, numeric strings and truncated floats. Empty seeds
leave the ambient stream alone; overflow and missing values raise an error.
``_rng.uniform`` reproduces R's default Mersenne-Twister ``runif`` sequence.
The public ``r_uniform`` object also permits direct seeding and drawing.
NumPy's ambient RNG is seeded/restored for its native repeatability contract;
**NumPy normal draws and JAX draws are not R draws**. ``kind`` and
``normal_kind`` support only the default MT/Inversion selector metadata;
other R RNG kinds, exact normal streams and cross-runtime model trajectory
identity are outside this adapter. These global RNG controls require sequential
use, as in R; do not run overlapping seeded R-mode calls in threads.

R-mode final assignments return their RHS, evaluated once. Explicit Python
workspaces still resolve their supplied inputs; implicit function-local lookup
is not inferred. Source blocks remain trusted Python, not translated R code.

.. autofunction:: pypomp.freeze

.. autofunction:: pypomp.archive_directory

.. autoclass:: pypomp.ArchiveValue

Compatibility and cost
----------------------

The native default retains these differences; the explicit mode above covers
the tested R recipe contracts:

* Archives are Python pickle files, not RDS/RDA, and old R archives are not read.
* JAX seeds do not reproduce R random streams, RNG selectors or coercion rules.
* Bake returns raw values and exposes metadata in the workspace rather than
  attaching R-style attributes to every possible Python value.
* Hidden names use ``_``; Python returns the stew names normally rather than
  invisibly. Function/class callers use explicit workspaces.
* The child workspace can read the supplied parent dictionary. The pinned R
  stew has a different miss-path environment chain and can fail to find a
  function caller's local variables.
* The default directory is the working directory; use ``dir`` or
  ``archive_directory`` for a different default.
* A failed seeded evaluation cleans up the temporary key, and writes use
  atomic replacement. These differ from failure paths in the pinned R source.

Only load trusted archives: unpickling can execute code. The format tag
identifies the operation and layout; it does not guarantee cross-version
compatibility. A corrupt, unsupported, or wrong-operation archive raises an
error. In native mode, failed recomputation preserves an existing archive, but arbitrary
side effects of the computation itself cannot be rolled back.

Use one writer per archive path. Native atomic replacement avoids exposing a partly
written file; it does not lock computations or coordinate competing writers.
Native archives use owner-only permissions (0600). Replacing a symbolic-link
archive path replaces the link itself and leaves its old target unchanged;
use the target path explicitly when that is the file you intend to update.
Hashing dependencies and reading the complete archive happen even on a hit.
Stale archives are fully loaded before their ingredients are rejected, and
stew sorts its local names. Small computations or large dependency objects
can therefore be faster without caching; measure completed work and archive
I/O separately when comparing performance.

Reference: `R pomp bake/stew source, commit 7cfb3f9
<https://github.com/kingaa/pomp/blob/7cfb3f9aa84c85de687b82d71b081016b9cc5762/R/bake.R>`_.
