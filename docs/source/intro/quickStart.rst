:parenttoc: True

Quick Start
=============

Here, we will take linear regression as an example, introduce the basic usage of :ref:`scope <scope_package>`. to solve a variables selection problem.

Suppose we collect :math:`n` independent observations for a response variable and :math:`p` explanatory variables, say :math:`y \in R^n` and :math:`X \in R^{n\times p}`. Let :math:`\epsilon_1, \ldots, \epsilon_n` be i.i.d zero-mean random noises and :math:`\epsilon = (\epsilon_1, \ldots, \epsilon_n)`, the linear model has a form:

.. math::
    y=X \beta^{*} +\epsilon.

.. code-block:: python

    from sklearn.datasets import make_regression
    ## generate data
    n, p, k= 10, 5, 3
    x, y, true_params = make_regression(n_samples=n, n_features=p, n_informative=k, coef=True) 


write the objective function with ``JAX`` package
------------------------------------------------------

Formalize the problem as a sparse convex optimization problem:

.. math::
    arg\min_{\beta \in R^p}L(\beta) := ||y-X \beta||^{2} s.t. ||\beta||_0 \leq s,

where objective function :math:`L(\beta)` is our optimization objective and the parameter, denoted as ``params``, is a vector which SCOPE needs to optimize and is imposed by sparse-constrain. 

The length of ``params`` is actually the dimension of the optimization problem so denoted as ``dimensionality``. The number of nonzero parameters is the sparsity level and denote as ``sparsity``.

From the perspective of variable selection, each parameter corresponds to a variable, and the nonzero parameters correspond to the selected variables. 

In the above example, :math:`\beta` is the parameters, so ``dimensionality`` is :math:`p` and ``sparsity`` is :math:`s`.

``JAX`` is Autograd version of ``Numpy`` for high-performance machine learning research. It is a Python library that provides a NumPy-compatible multidimensional array API and automatic differentiation. As the usage of ``JAX`` is similar to ``Numpy``, we will not introduce it here. For more information, please refer to `JAX <https://jax.readthedocs.io/en/latest/index.html>`_.

In the above example, we can define the objective function :math:`L(\beta)` as ``custom_objective(params)``:

.. code-block:: python

    import jax.numpy as jnp

    def custom_objective(params):
        return jnp.sum(
            jnp.square(y - x @ params)
        )
 

initialize ``ScopeSolver`` and solve the problem
------------------------------------------------------

Those concepts are introduced in the previous section. 

- ``dimensionality`` is the number of parameters and must be offered.
- ``sparsity`` is the sparsity level, int or list of int. If it is an int, the solver will use the given sparsity level, otherwise, the solver will search the best sparsity level from the given list which will be introduced in the next section.

.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=k ## we want to select k variables
    )

``solve`` is the main function of SCOPE, it takes the objective function as optimization objective and commands the algorithm to begin the optimization process. 

.. code-block:: python

    solver.solve(custom_objective)



get results
------------------------------------------------------

- ``get_estimated_params`` returns the optimized parameters.
- ``get_support`` returns the index of selected variables (nonzero parameters).

.. code-block:: python

    beta = solver.get_estimated_params()
    support_set = solver.get_support()
