Pomp Class
==========

.. currentmodule:: pypomp.core.pomp

.. autoclass:: Pomp
   :no-members:


.. rubric:: Attributes
.. autosummary::
   :toctree: generated/

   ~Pomp.ys
   ~Pomp.theta
   ~Pomp.canonical_param_names
   ~Pomp.statenames
   ~Pomp.t0
   ~Pomp.rinit
   ~Pomp.rproc
   ~Pomp.dmeas
   ~Pomp.rmeas
   ~Pomp.par_trans
   ~Pomp.covars
   ~Pomp.accumvars
   ~Pomp.results_history
   ~Pomp.fresh_key
   ~Pomp.metadata

.. rubric:: Core Algorithmic Methods
.. autosummary::
   :toctree: generated/

   ~Pomp.simulate
   ~Pomp.pfilter
   ~Pomp.mif
   ~Pomp.train
   ~Pomp.dpop_train

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
