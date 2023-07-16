:parenttoc: True

Variants of Sparsity-Constraint Optimization
====================

Here we will introduce some helpful variants of sparsity-constraint optimization (SCO) that can be addressed by ``skscope``.

Group-structured parameters
----------------------------

We may encounter group-structured parameters, i.e., all parameters are divided into several non-overlapping parameters groups. Typical examples are group variable selection, `multitask learning <../userguide/examples/GeneralizedLinearModels/multiple-response-linear-regression.html>`__, and so on. 

When encountering group-structured parameters, we treated each parameter group as a unit, and we will select (or unselect) all of parameters in the same group at the same time. We named this problem as group SCO (GSCO)

If we want to select :math:`s` groups of parameters among :math:`q` parameters, the group variable selection can be formulated as this:

Let :math:`G=\{g_1, \dots, g_q\}` be a partition of the index set :math:`\{1, \dots, p\}`, where :math:`g_i \subset \{1, \dots, p\}` for :math:`i=1, \dots, q`, `g_i \cap g_j = \emptyset` for :math:`i \neq j` and :math:`\bigcup_{i=1}^q g_i = \{1, \dots, p\}`. Then optimize the objective function with constraints:


.. math::
	\min_{x \in R^p} L(x),\operatorname{ s.t. } \sum_{i=1}^q I({\|x}_{g_i}\|\neq 0) \leq s,

where :math:`s` is the group sparsity level, i.e., the number of groups to be selected. If :math:`q=p`, then :math:`g_i = \{i\}` for :math:`i=1, \dots, q`, so the group variable selection is equivalent to the original variable selection. 

In this wide-appeared case, we cooperate a ``group`` parameter in the solvers in ``skscope``. ``group`` is an incremental integer array of length ``dimensionality`` starting from 0 without gap, which means the variables in the same group must be adjacent, and they will be selected together or not.

Here are wrong examples: [0,2,1,2] (not incremental), [1,2,3,3] (not start from 0), [0,2,2,3] (there is a gap).

Note that ``params`` (parameters) must be a vector not matrix and ``sparsity`` represents **the number of groups** to be selected here.


.. code-block:: python

    p, s, m = 3, 2, 2
    params_true = [1, 10, 0, 0, -1, 5]
    ScopeSolver(
            dimensionality=p * m, 
            sparsity=s,
            group=[0,0,1,1,2,2]
        )


search support size
-------------------------

In the previous section, we have introduced how to set the sparsity level. However, sometimes we do not know the sparsity level and need to search it. In this case, we can set ``sparsity`` as a list of int, and the solver will search the best sparsity level from the given list.

Note that ``sample_size`` must be offered to ``ScopeSolver`` when ``sparsity`` is a list.


.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n ## the number of samples
    )


Preselected Non-sparse Parameters
--------------------------------

:ref:`skscope <skscope_package>` allows users to specify some parameter which must has a non-zero value. This widely occurs scenario in modelling. For example, 

- in linear model with intercept, people often assume the intercept is a non-sparse parameter

- in Ising model, the diagonal entries of corresponding matrix represents the strength of external magnetic which is often consider as a non-zero entries. 

We can use ``always_select`` to set the parameters that we want to select. ``always_select`` is a list of ``int``, and the solver will always select these parameters.

Here is an example:

.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p,      ## there are p parameters
        always_select=[0, 1],  ## we want to select the first two variables
    )


Flexible Optimization Interface
---------------------------------------------------------

For all solvers in ``skscope`` (except ``IHTSolver``), one of indispensible step in these solvers is solving a optimization problem:

.. math::
    \arg\min_{\theta \in R^s} f(\theta),

where

- :math:`\theta` is a :math:`s`-dimensional parameter vector that needs to be optimized

- :math:`f(\theta)` is the objective function; 

All solvers in :ref:`skscope <skscope_package>` use `nlopt <https://nlopt.readthedocs.io/en/latest/>`_ as the default numeric optimization solver for this problem. 

In some cases, there are additional constraint for the intrinsic structure for :math:`\theta`, which is formulated as a set :math:`\mathcal{C}`:

.. math::
    \arg\min_{\theta \in R^s, \theta \in \mathcal{C}} f(\theta).

A typical example is the Gaussian graphical model for continuous random variables, which constrain the :math:`\theta` within a symmetric positive-definitive space (see `this example <../userguide/examples/GraphicalModels/sparse-gaussian-precision-matrix.html>`__). Though ``nlopt`` cannot solve this problem, ``skscope`` provide a flexible interface that can replace it. Specifically, users can change the default numeric optimization solver by properly setting the ``numeric_solver`` in the solver. 

    > Notice that, the accepted input of ``numeric_solver`` should have the same interface as ``skscope.numeric_solver.convex_solver_nlopt``.


.. code-block:: python

    from skscope import ScopeSolver
    def custom_numeric_solver(*args, **kwargs):
        params = []
        # do something about params
        return params

    p, k = 10, 3
    solver = ScopeSolver(p, k, numeric_solver=custom_numeric_solver)

This feature definitely borden the application range of the ``skscope`` by cooperating ``skscope`` with the other powerful optimization toolkits in Python.
We just briefly introduce some examples:

- ``cvxpy``: an open source Python-embedded modeling language for convex optimization problems. Its `official website <https://www.cvxpy.org/>`__ supplies powerful features (such as semi-definite program) that can be .

- ``scipy.optimize``: includes solvers for nonlinear problems, linear programming, constrained and nonlinear least-squares, root finding, and curve fitting. Its documentation can be found `here <https://docs.scipy.org/doc/scipy/reference/optimize.html/>`__.
