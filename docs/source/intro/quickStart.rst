:parenttoc: True

Quick Start
=============

Here, we will take linear regression as an example, introduce the basic usage of :ref:`scope <scope_package>`. to solve a variables selection problem.

Suppose we collect :math:`n` independent observations for a response variable and :math:`p` explanatory variables, say :math:`y \in R^n` and :math:`X \in R^{n\times p}`. Let :math:`\epsilon_1, \ldots, \epsilon_n` be i.i.d zero-mean random noises and :math:`\epsilon = (\epsilon_1, \ldots, \epsilon_n)`, the linear model has a form:

.. math::
    y=X \beta^{*} +\epsilon.

First, we need to define a loss function as the optimization objective. 

1. find the loss function in math form
------------------------------------------------------

Formalize the problem as a sparse convex optimization problem:

.. math::
    arg\min_{\beta \in R^p}L(\beta) := ||y-X \beta||^{2} s.t. ||\beta||_0 \leq s,

where loss function :math:`L(\beta)` is our optimization objective. 

2. decide parameters
------------------------------------------------------

Parameters, denoted as ``para``, are vector which SCOPE needs to optimize and is imposed by sparse-constrain. The number of ``para`` is actually the dimension of the optimization problem so denoted as ``dimensionality``.

From the perspective of variable selection, each parameter corresponds to a variable, and the nonzero parameters correspond to the selected variables. The number of nonzero parameters is the sparsity level and denote as ``sparsity_level``.

In the above example, :math:`\beta` is the parameters, so ``dimensionality`` is :math:`p` and ``sparsity_level`` is :math:`s`.

3. make custom data
------------------------------------------------------

We can pass any additional data through to loss function. The data may be a user-defined class containing information your loss function needs and this loss is the only function that can access the custom type, so it can be anything you want.

In the above example, the loss function need explanatory variables :math:`X` and response variable :math:`y`, so we can define a class ``CustomData`` to store them.

.. code-block:: python

    class CustomData:
        def __init__(self, X, y):
            self.X = X
            self.y = y
    
    train_data = CustomData(data_set.x, data_set.y)


4. use ``JAX`` package
------------------------------------------------------

``JAX`` is Autograd version of ``Numpy`` for high-performance machine learning research. It is a Python library that provides a NumPy-compatible multidimensional array API and automatic differentiation. As the usage of ``JAX`` is similar to ``Numpy``, we will not introduce it here. For more information, please refer to `JAX <https://jax.readthedocs.io/en/latest/index.html>`_.

In the above example, we can define the loss function :math:`L(\beta)` as ``custom_loss(para)``:

.. code-block:: python

    import jax.numpy as jnp

    def custom_loss(para):
        return jnp.sum(
            jnp.square(train_data.y - train_data.X @ para)
        )


Then we can initialize the solver and set the loss function. 

5. initialize ``ConvexSparseSolver``
------------------------------------------------------

Those concepts are introduced in the previous section. 

- ``dimensionality`` is the number of parameters and must be offered.
- ``sparsity_level`` is the sparsity level, int or list of int. If it is an int, the solver will use the given sparsity level, otherwise, the solver will search the best sparsity level from the given list which will be introduced in the next section.

.. code-block:: python

    solver = ConvexSparseSolver(
        dimensionality=p, ## there are p parameters
        sparsity_level=k ## we want to select k variables
    )

Finally, we can solve the problem and get the result.

6. begin to solve 
------------------------------------------------------

``solve`` is the main function of SCOPE, it takes the loss function as optimization objective and commands the algorithm to begin the optimization process. 

.. code-block:: python

    solver.solve(custom_loss)

7. get results
------------------------------------------------------

- ``get_parameters`` returns the optimized parameters.
- ``get_selected_variables`` returns the index of selected variables (nonzero parameters).

.. code-block:: python

    beta = solver.get_parameters()
    support_set = solver.get_selected_variables()
