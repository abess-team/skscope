:parenttoc: True

Quick Start for Practitioner
============================

Usually, practitioners are likely to use fully implemented end-to-end methods. We have designed :ref:`skscope <skscope_package>` to support powerful SCO methods for non-trivially and practically important problems to accommodate this need. :ref:`skscope <skscope_package>` serves as a machine learning factory continuously producing easy-to-use end-to-end sparse learning algorithms.

.. figure:: ../contribute/figure/architecture-scope.png
   :width: 100%
   :align: center
   :alt: The software architecture of :ref:`skscope <skscope_package>`.
   
   The software architecture of :ref:`skscope <skscope_package>`.

Specifically, the submodule ``skscope.skmodel`` in :ref:`skscope <skscope_package>` includes the implementation of practically valued SCO methods that can be directly used by practitioners. More importantly, these methods are designed to be compatible with the ``sklearn`` library. This allows Python users to use a familiar ``sklearn`` API to train models and easily create ``sklearn`` pipelines incorporating these models. Currently, these end-to-end methods supported by :ref:`skscope <skscope_package>` are summarized in this table:

.. list-table:: Some application-oriented interfaces implemented in the module ``skscope.skmodel`` in :ref:`skscope <skscope_package>`.
   :header-rows: 1

   * - **skmodel**
     - **Description**
   * - PortfolioSelection
     - Construct sparse Markowitz portfolio
   * - NonlinearSelection
     - Select relevant features with nonlinear effect
   * - RobustRegression
     - A robust regression dealing with outliers
   * - MultivariateFailure
     - Multivariate failure time model in survival analysis
   * - IsotonicRegression
     - Fit the data with a non-decreasing curve

