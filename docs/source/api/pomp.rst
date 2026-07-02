Pomp Class
==========

.. currentmodule:: pypomp.core.pomp

.. autoclass:: Pomp
   :no-members:


.. rubric:: Attributes

.. autoattribute:: Pomp.ys
.. autoattribute:: Pomp.theta
.. autoattribute:: Pomp.canonical_param_names
.. autoattribute:: Pomp.statenames
.. autoattribute:: Pomp.t0
.. autoattribute:: Pomp.rinit
.. autoattribute:: Pomp.rproc
.. autoattribute:: Pomp.dmeas
.. autoattribute:: Pomp.rmeas
.. autoattribute:: Pomp.par_trans
.. autoattribute:: Pomp.covars
.. autoattribute:: Pomp.accumvars
.. autoattribute:: Pomp.results_history
.. autoattribute:: Pomp.fresh_key
.. autoattribute:: Pomp.metadata


.. rubric:: Core Algorithmic Methods
.. autosummary::
   :toctree: generated/

   ~Pomp.simulate
   ~Pomp.pfilter
   ~Pomp.mif
   ~Pomp.train
   ~Pomp.dpop_train
   ~Pomp.bif
   ~Pomp.pmcmc
   ~Pomp.abc

.. rubric:: Supporting Methods
.. autosummary::
   :toctree: generated/

   ~Pomp.sample_params
   ~Pomp.prune
   ~Pomp.results
   ~Pomp.traces
   ~Pomp.CLL
   ~Pomp.ESS
   ~Pomp.time
   ~Pomp.arma
   ~Pomp.negbin
   ~Pomp.probe
   ~Pomp.print_summary
   ~Pomp.print_metadata
   ~Pomp.plot_traces
   ~Pomp.plot_simulations
   ~Pomp.merge
