Results
=======

Results objects in **pypomp** are structured dataclasses that store the output of algorithmic methods, including log-likelihood estimates, parameter traces, and diagnostic information. 

Each :class:`~pypomp.core.pomp.Pomp` and :class:`~pypomp.panel.panel.PanelPomp` object maintains a :class:`ResultsHistory` that automatically records the outcome of every method call (e.g., ``pfilter``, ``mif``, ``train``). 
In addition to accessing the stored values directly, you can access these results as tidy pandas DataFrames via the :meth:`~pypomp.core.pomp.Pomp.results` and :meth:`~pypomp.core.pomp.Pomp.traces` methods.

.. currentmodule:: pypomp.core.results

.. rubric:: Results History
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ResultsHistory

.. rubric:: Pomp Results
.. autosummary::
   :toctree: generated/
   :nosignatures:

   PompPFilterResult
   PompMIFResult
   PompTrainResult

.. rubric:: PanelPomp Results
.. autosummary::
   :toctree: generated/
   :nosignatures:

   PanelPompPFilterResult
   PanelPompMIFResult
   PanelPompTrainResult