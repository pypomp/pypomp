CuTe Backend
============

The experimental CuTe backend runs a model's process model as one CUDA kernel per observation interval on an NVIDIA GPU.
Choose it with :meth:`~pypomp.core.pomp.Pomp.set_backend`, and see :ref:`cute-backend` for when it is faster and how its results differ from XLA's.
:meth:`~pypomp.core.pomp.Pomp.backend_report` returns a :class:`BackendReport` of what runs on the kernel.

.. currentmodule:: pypomp.cute

.. autosummary::
   :toctree: generated/
   :nosignatures:

   available
   BackendReport
   Part
   Problem
