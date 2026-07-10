PanelPomp Class
===============

.. currentmodule:: pypomp

.. autoclass:: PanelPomp
   :no-members:


.. rubric:: Attributes

.. autoattribute:: PanelPomp.ys
.. autoattribute:: PanelPomp.theta
.. autoattribute:: PanelPomp.unit_objects
.. autoattribute:: PanelPomp.canonical_param_names
.. autoattribute:: PanelPomp.canonical_shared_param_names
.. autoattribute:: PanelPomp.canonical_unit_param_names
.. autoattribute:: PanelPomp.results_history
.. autoattribute:: PanelPomp.fresh_key
.. autoattribute:: PanelPomp.metadata


.. rubric:: Core Algorithmic Methods
.. autosummary::
   :toctree: generated/

   ~PanelPomp.simulate
   ~PanelPomp.pfilter
   ~PanelPomp.mif
   ~PanelPomp.train
   ~PanelPomp.dpop_train

.. rubric:: Results
.. autosummary::
   :toctree: generated/

   ~PanelPomp.results
   ~PanelPomp.CLL
   ~PanelPomp.ESS
   ~PanelPomp.traces


.. rubric:: Other Supporting Methods
.. autosummary::
   :toctree: generated/

   ~PanelPomp.get_unit_names
   ~PanelPomp.get_unit_parameters
   ~PanelPomp.sample_params
   ~PanelPomp.to_struct
   ~PanelPomp.prune
   ~PanelPomp.mix_and_match
   ~PanelPomp.time
   ~PanelPomp.probe
   ~PanelPomp.arma
   ~PanelPomp.negbin
   ~PanelPomp.print_metadata
   ~PanelPomp.print_summary
   ~PanelPomp.merge

.. rubric:: Visualization
.. autosummary::
   :toctree: generated/

   ~PanelPomp.plot_traces
   ~PanelPomp.plot_simulations
