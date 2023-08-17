:parenttoc: True

Variants of Sparsity-Constraint Optimization
=====================================================

In addition to standard sparsity-constrained optimization (SCO) problems, ``skscope`` also provides support for several helpful variants of SCO.

Group-structured parameters
----------------------------

In certain cases, we may encounter group-structured parameters where all parameters are divided into non-overlapping groups. Examples of such scenarios include group variable selection under linear model `[1]`_, `multitask learning <../userguide/examples/GeneralizedLinearModels/multiple-response-linear-regression.html>`__, and so on. 

When dealing with group-structured parameters, we treat each parameter group as a unit, selecting or deselecting all the parameters in the group simultaneously. This problem is referred to as group SCO (GSCO).

For instance, if we aim to select :math:`s` groups of parameters from a total of :math:`q` parameters group, the GSCO problem can be formulated as follows:

.. math::
	\min_{\theta \in R^p} f(x),\operatorname{ s.t. } \sum_{i=1}^q I({\|\theta}_{g_i}\|\neq 0) \leq s,

where :math:`G=\{g_1, \dots, g_q\}` is a partition of the index set satisfying: 

- :math:`g_i \subseteq \{1, \dots, p\}` for :math:`i=1, \dots, q`,

- :math:`g_i \cap g_j = \emptyset` for :math:`i \neq j`,

and 

-  :math:`\bigcup_{i=1}^q g_i = \{1, \dots, p\}`. 

Particularly, if :math:`q=p` and :math:`g_i = \{i\}` (for :math:`i=1, \dots, q`), then the GSCO is equivalent to SCO. Thus, GSCO is a generalization of SCO. 

When encountering GSCO, we utilize the ``group`` parameter in the solvers provided by ``skscope``. The ``group`` parameter is an incremental integer array of length dimensionality, starting from 0 without gaps. This means that the variables within the same group must be adjacent, and they will be selected or deselected together. Here are some examples of invalid ``group`` parameter arrays: 

- ``group = [0, 2, 1, 2]`` (not incremental), 

- ``group = [1, 2, 3, 3]`` (doesn't start from 0), 

- ``group = [0, 2, 2, 3]`` (contains a gap).

When solving GSCO with ``skscope``, please note that 

- the ``dimensionality`` parameter is the number of all parameters,  

- the ``sparsity`` parameter represents **the number of parameter groups** to be selected.

.. code-block:: python

    from skscope import ScopeSolver

    q, s, m = 3, 2, 2
    params_true = [1, 10, 0, 0, -1, 5]
    ScopeSolver(
        dimensionality=q * m,     ## the total number of parameters 
        sparsity=s,               ## the number of parameter groups to be selected
        group=[0, 0, 1, 1, 2, 2], ## specify group structure
    )


Preselected Non-sparse Parameters
--------------------------------------

When using various model, it is common to have certain parameters that must have a non-zero value. For example:

- in linear models with an intercept, the intercept term is often assumed to be non-sparse;

- in Ising models, the diagonal entries of the corresponding matrix represent the strength of external magnetic fields and are typically considered non-zero.

Let :math:`\mathcal{P}` be a set that represents pre-selecting parameters, the generalized SCO is formulated as:

.. math::

    \arg\min\limits_{\theta \in R^p} f(\theta) \text{ s.t. } ||\theta_{\mathcal{P}^c}||_0 \leq s. 

:ref:`skscope <skscope_package>` allows users to specify such preselected non-sparse parameters using the ``preselect`` parameter. This parameter is a list of integers, and the solver will always select these parameters.

Here is an illustrative example for the usage of ``preselect``:

.. code-block:: python
    
    from skscope import ScopeSolver

    solver = ScopeSolver(
        dimensionality=10,      ## 10 parameters in total
        sparsity=3,             ## 3 non-sparse parameters to be selected
        preselect=[0, 1],   ## always select the first two parameters as non-sparse values
    )


Re-parameterization by Layers
---------------------------------------------------------
In practice, we may need certain requirements for parameters, which can be achieved through re-parameterization. 
For example, in some cases, we want the parameters corresponding to the non-selected features to be equal to a constant rather than zero.

.. math:: 
    
        \arg\min_{\theta \in R^p} f(\theta) \text{ s.t. } ||\theta - \mu||_0 \leq s, 
    
where :math:`\mu \in R^p` is a offset vector. For this, we can re-parameterize the original problem as follows: 

.. math::
        
        \arg\min_{\theta' \in R^p} f(\theta' + \mu) \text{ s.t. } ||\theta'||_0 \leq s,
    
which the parameters are re-parameterized before entering the objective function.

In :ref:`skscope <skscope_package>`, we can achieve this by ``layers`` parameter in the ``solve`` method of sparse solvers.

.. code-block:: python

    from skscope import ScopeSolver
    from skscope.layer import OffsetSparse
    from jax import numpy as jnp

    X = jnp.array([[1, 2, 3], [4, 5, 6]])
    y = jnp.array([1, 2])

    def loss(params):
        return jnp.sum((X @ params - y) ** 2)

    solver = ScopeSolver(3, 1)

    solver.solve(
        loss,
        layers=[OffsetSparse(dimensionality=3, offset=1)], 
    )

    print(solver.get_estimated_params())

Let ``params`` pass through an offset-layer ``OffsetSparse`` before entering ``loss``. 
In this way, the parameters corresponding to the non-selected features will be equal to 1 rather than zero.

Further, we can use several layers at the same time to achieve more complex re-parameterization.
``layers`` is a list of layers, and the parameters will pass through these layers in order before entering ``loss``.

In ``skscope.layer``, we provide several layers for re-parameterization: ``NonNegative``, ``LinearConstraint``, ``SimplexConstraint`` and ``BoxConstraint``.
In addition, users can also define their own layers by inheriting the ``skscope.layer.Identity`` class.


Flexible Optimization Interface
---------------------------------------------------------

For all solvers in ``skscope`` (except ``IHTSolver``), an indispensable step in these solvers is solving an optimization problem:

.. math::
    \arg\min_{\theta \in R^s} f(\theta),

where

- :math:`\theta` is a :math:`s`-dimensional parameter vector (note that :math:`s` is the desired sparsity in SCO)

- :math:`f(\theta)` is the objective function; 

All solvers in :ref:`skscope <skscope_package>` use `nlopt <https://nlopt.readthedocs.io/en/latest/>`_ as the default numeric optimization solver for this problem. 

In some cases, there may be additional constraints on the intrinsic structure of :math:`\theta`, which can be formulated as a set :math:`\mathcal{C}`:

.. math::
    \arg\min_{\theta \in R^s, \theta \in \mathcal{C}} f(\theta).

A typical example is the Gaussian graphical model for continuous random variables, which constrains :math:`\theta` on symmetric positive-definite spaces (see this example `<../userguide/examples/GraphicalModels/sparse-gaussian-precision-matrix.html>`__). Although ``nlopt`` cannot solve this problem, ``skscope`` provides a flexible interface that allows for its replacement. Specifically, users can change the default numerical optimization solver by properly setting the ``numeric_solver`` in the solver. 

    > Notice that, the accepted input of ``numeric_solver`` should have the same interface as ``skscope.numeric_solver.convex_solver_nlopt``.


.. code-block:: python

    from skscope import ScopeSolver
    def custom_numeric_solver(*args, **kwargs):
        params = []
        # do something about params
        return params

    p, k = 10, 3
    solver = ScopeSolver(p, k, numeric_solver=custom_numeric_solver)

This feature significantly expands the application range of ``skscope`` by allowing it to cooperate with other powerful optimization toolkits in Python.
We will briefly introduce some examples:

- ``cvxpy``: an open source Python-embedded modeling language for convex optimization problems. Its `official website <https://www.cvxpy.org/>`__ provides powerful features (such as semi-definite programs).

- ``scipy.optimize``: includes solvers for nonlinear problems, linear programming, constrained and nonlinear least-squares, root finding, and curve fitting. Its documentation can be found `here <https://docs.scipy.org/doc/scipy/reference/optimize.html/>`__.

Reference
---------------------------------------------------------

- _`[1]` Zhang, Y., Zhu, J., Zhu, J., & Wang, X. (2023). A splicing approach to best subset of groups selection. INFORMS Journal on Computing, 35(1), 104-119.