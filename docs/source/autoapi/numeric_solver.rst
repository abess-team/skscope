(Nonlinear) Optimization Functions
========================================

The implementation for solver 

.. math::
    \arg\min_{\theta \in R^s} f(\theta),

where

- :math:`\theta` is a :math:`s`-dimensional parameter vector (note that :math:`s` is the desired sparsity in sparsity-constraint optimization)

- :math:`f(\theta)` is the objective function.

Functions
---------------

.. autoapisummary::

    skscope.numeric_solver.convex_solver_nlopt

.. autoapimodule:: skscope.numeric_solver
    :members: 