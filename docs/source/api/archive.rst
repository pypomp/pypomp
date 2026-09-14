Computation Archives
====================

.. currentmodule:: pypomp

:func:`bake` and :func:`stew` save the result of an expensive computation to
a file and load it on later runs, so a notebook or script can be re-run
without repeating long fits. An archive is reused only while the
computation's source code, its declared dependencies, and its seed are
unchanged.

.. autofunction:: bake

.. autofunction:: stew
