PanelPomp Class
===============

.. currentmodule:: pypomp.panel.panel

.. autoclass:: PanelPomp
   :no-members:


.. rubric:: Attributes

.. autoattribute:: PanelPomp.ys
.. autoattribute:: PanelPomp.unit_objects
.. autoattribute:: PanelPomp.theta
.. autoattribute:: PanelPomp.results_history
.. autoattribute:: PanelPomp.fresh_key
.. autoattribute:: PanelPomp.metadata
.. autoattribute:: PanelPomp.canonical_param_names
.. autoattribute:: PanelPomp.canonical_shared_param_names
.. autoattribute:: PanelPomp.canonical_unit_param_names


.. rubric:: Core Algorithmic Methods
.. autosummary::
   :toctree: generated/

   ~PanelPomp.simulate
   ~PanelPomp.pfilter
   ~PanelPomp.mif
   ~PanelPomp.train

.. rubric:: Supporting Methods
.. autosummary::
   :toctree: generated/

   ~PanelPomp.get_unit_names
   ~PanelPomp.get_unit_parameters
   ~PanelPomp.sample_params
   ~PanelPomp.prune
   ~PanelPomp.mix_and_match
   ~PanelPomp.results
   ~PanelPomp.CLL
   ~PanelPomp.ESS
   ~PanelPomp.time
   ~PanelPomp.traces
   ~PanelPomp.probe
   ~PanelPomp.arma
   ~PanelPomp.negbin
   ~PanelPomp.plot_traces
   ~PanelPomp.plot_simulations
   ~PanelPomp.print_metadata
   ~PanelPomp.print_summary
   ~PanelPomp.merge
