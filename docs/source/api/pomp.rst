Pomp Class
==========

.. currentmodule:: pypomp

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
   ~Pomp.pmcmc
   ~Pomp.abc

.. rubric:: Results
.. autosummary::
   :toctree: generated/

   ~Pomp.results
   ~Pomp.traces
   ~Pomp.CLL
   ~Pomp.ESS

.. rubric:: Other Supporting Methods
.. autosummary::
   :toctree: generated/

   ~Pomp.sample_params
   ~Pomp.to_struct
   ~Pomp.prune
   ~Pomp.time
   ~Pomp.arma
   ~Pomp.negbin
   ~Pomp.probe
   ~Pomp.print_summary
   ~Pomp.print_metadata
   ~Pomp.merge

.. rubric:: Visualization
.. autosummary::
   :toctree: generated/

   ~Pomp.plot_traces
   ~Pomp.plot_simulations
