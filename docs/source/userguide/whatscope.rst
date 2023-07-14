:parenttoc: True


What is ``skscope``?
=====================================

``skscope`` is an open-source Python package designed for efficiently solving sparsity-constrained optimization (SCO) problems. Since SCO can handle many problems in machine learning, statistics, signal processing, and related application domains, ``skscope`` has many applications in these fields. For examples, variable selection (a.k.a., feature selection or compress sensing) is one of the most classical SCO problem. Below we highlight some interesting problems that ``skscope`` can handle.

1. `Robust variable selection <examples/LinearModelAndVariants/robust-regression.html>`__

.. image:: figure/variable_selection.png
  :width: 300
  :align: center

2. `Nonlinear variable selection <examples/Miscellaneous/hsic-splicing.html>`__

.. image:: figure/nonlinear_variable_selection.png
  :width: 600
  :align: center


3. `Spatial trend filtering <examples/FusionModels/spatial-trend-filtering.html>`__

.. image:: figure/trend_filter.png
  :width: 600
  :align: center

4. `Network reconstruction <examples/GraphicalModels/sparse-gaussian-precision.html>`__

.. image:: figure/precision_matrix.png
  :width: 600
  :align: center

5. `Portfolio selection <examples/Miscellaneous/portfolio-selection.html>`__

.. image:: figure/precision_matrix.png
  :width: 600
  :align: center


Beyond that, we currently provide more than 20 examples in our `example gallery <examples/index.html>`__.


How does ``skscope`` work? 
--------------------------

