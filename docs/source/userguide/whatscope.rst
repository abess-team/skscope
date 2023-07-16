:parenttoc: True


What is ``skscope``?
=====================================

``skscope`` is a powerful open-source Python package specifically developed to tackle sparsity-constrained optimization (SCO) problems with utmost efficiency. With SCO's broad applicability in machine learning, statistics, signal processing, and other related domains, ``skscope`` can find extensive usage in these fields. For example, it excels in solving classic SCO problems like variable selection (also known as feature selection or compress sensing). Even more impressively, it goes beyond that and handles a diverse range of intriguing real-world problems:

1. `Robust variable selection <examples/LinearModelAndVariants/robust-regression.html>`__

.. image:: figure/variable_selection.png
  :width: 300
  :align: center

2. `Nonlinear variable selection <examples/Miscellaneous/hsic-splicing.html>`__

.. image:: figure/nonlinear_variable_selection.png
  :width: 666
  :align: center


3. `Spatial trend filtering <examples/FusionModels/spatial-trend-filtering.html>`__

.. image:: figure/trend_filter.png
  :width: 666
  :align: center

4. `Network reconstruction <examples/GraphicalModels/sparse-gaussian-precision.html>`__

.. image:: figure/precision_matrix.png
  :width: 666
  :align: center

5. `Portfolio selection <examples/Miscellaneous/portfolio-selection.html>`__

.. image:: figure/portfolio_selection.png
  :width: 300
  :align: center


These above examples represent just a glimpse of the practical problems that ``skscope`` can effectively address. With its efficient optimization algorithms and versatility, ``skscope`` proves to be an invaluable tool for a wide range of disciplines. Currently, we offer over 20 examples in our comprehensive `example gallery <examples/index.html>`__.


How does ``skscope`` work? 
--------------------------

Specifically, ``skscope`` aims to tackle this problem: 

.. math:: 
   \min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s,

where :math:`f(x)` is a differential objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.