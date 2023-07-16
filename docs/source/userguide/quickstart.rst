:parenttoc: True

Quick Example for Beginner
============================

Here, we provide a beginner-friendly example that introduces the basic usage of :ref:`skscope <skscope_package>` for feature/variable selection in linear regression.

Introduction
-----------------

Let's consider a dataset with :math:`n=100` independent observations of an interesting response variable and :math:`p=10` predictive variables:

.. code-block:: python

    from sklearn.datasets import make_regression

    ## generate data
    n, p, s = 100, 10, 3
    X, y, true_coefs = make_regression(n_samples=n, n_features=p, n_informative=s, coef=True, random_state=0) 
    
    print("X shape:", X.shape)
    >>> X shape: (100, 10)
    
    print("Y shape:", y.shape)
    >>> Y shape: (100,)

In this scenario, our goal is to explain the variation in the response variable using the available predictive variables. However, we do not have knowledge of the true parameters represented by ``true_coefs``. We only know that out of the :math:`p=10` predictive variables, only :math:`s=3` variables actually influence the response variable. Furthermore, we assume a linear relationship between the predictive variables and the response variable:

.. math::
    y=X \theta^{*} +\epsilon.

where 

- :math:`y \in R^n` and :math:`X \in R^{n\times p}` represent the observations on the response variable and the predictive variables, respectively;

- :math:`\epsilon = (\epsilon_1, \ldots, \epsilon_n)` is a vector consisting of :math:`n` i.i.d zero-mean random noises;

- :math:`\theta^{*}` is an unknown coefficient vector that needs to be estimated.

With ``skscope``, users can effectively estimate :math:`\theta^{*}` and identify the influential predictive variables through variable selection. Below presents a step-by-step procedure to achieve this.

A Step-by-Step Procedure
-------------------------------

Sparsity-Constrained Optimization Perspective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Let's begin by formulating the variable selection problem as an optimization problem:

.. math::
    \arg\min_{\theta \in R^p} ||y-X \theta||^{2} s.t. ||\theta||_0 \leq s,

where:

- :math:`\theta` is a coefficient vector to be optimized.

- :math:`||y-X \theta||^{2}` is the objective function that quantifies the quality of the coefficient vector :math:`\theta`.

- :math:`||\theta||_0` counts the number of non-zero elements in :math:`\theta`. The sparsity constraint :math:`||\theta||_0 \leq s` reflects our prior knowledge that the number of influential predictive variables is less than or equal to :math:`s`.

The intuitive explanation for this optimization problem is to find a small subset of predictors that result in a linear model with the most desirable prediction accuracy. By doing so, we leverage the information in the observed data and our prior knowledge effectively.


Solve SCO via ``skscope``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimization problem described above is a sparsity-constrained optimization (SCO) problem. Since ``skscope`` is designed for general SCO, it can solve this problem with ease. The beauty of using skscope is its simplicity --- it can solve SCO problems as long as the objective function :math:`||y-X \theta||^{2}` is properly implemented. For the present example, we will program the objective function as a Python function named  ``objective_function(coefs)``:

.. code-block:: python

    import jax.numpy as jnp

    def objective_function(coefs):
        return jnp.linalg.norm(y - X @ coefs)

Note that we import the required jax library [*]_ on the first line. This library provides a ``numpy``-compatible multidimensional array API and supports automatic differentiation. As the usage of ``jax.numpy`` is very similar to ``numpy``, here, we can understand ``jax.numpy`` as being equivalent to ``numpy`` [*]_. Through the code snippet above, the objective_function is implemented exactly as the mathematical function :math:`||y-X \theta||^{2}`.

Next, we input the ``objective_function`` into one solver in ``skscope`` to get the practice solution of the SCO problem. Below, we import the ``ScopeSolver`` [*]_ and properly configure it:  

.. The length of ``coefs`` is actually the dimension of the optimization problem so denoted as ``dimensionality``. The number of nonzero parameters is the sparsity level and denote as ``sparsity``.

.. From the perspective of variable selection, each parameter corresponds to a variable, and the nonzero parameters correspond to the selected variables.

.. code-block:: python

    from skscope import ScopeSolver

    scope_solver = ScopeSolver(
        dimensionality=p,  ## there are p parameters
        sparsity=s,        ## we want to select s variables
    )

In the above configuration, we set:

- ``dimensionality``: the number of parameters, which must be specified;

and 

- ``sparsity``: the desired sparsity level. 

In the above example, :math:`\theta` is the parameters, so ``dimensionality`` is :math:`p` and ``sparsity`` is :math:`s`.

With ``scope_solver`` and ``objective_function`` defined, we can solve the SCO problem using a single command:

.. code-block:: python

    scope_solver.solve(objective_function)


The ``solve`` method is the main method for the solvers in ``skscope``. It takes the objective function as the optimization objective and instructs the algorithm to conduct the optimization process.

Retrieving the Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the solvers in ``skscope`` come with necessary functions to extract results, we can obtain the desired outcomes with just one line of code. For instance, if we are interested in retrieving the effective variables, we can use the ``get_support`` method:


.. code-block:: python

    import numpy as np

    estimated_support_set = scope_solver.get_support()
    print("Estimated effective predictors:", estimated_support_set)
    >>> Estimated effective predictors: [3 5 6]
    print("True effective predictors:", np.nonzero(true_coefs)[0])
    >>> True effective predictors: [3 5 6]

We can observe that the estimated effective predictive variables match the true ones, indicating the accuracy of the solver in ``skscope``.

Additionally, we may be interested in the regression coefficients:

- the ``get_estimated_params`` method returns the optimized coefficients.

.. code-block:: python

    est_coefs = scope_solver.get_estimated_params()
    print("Estimated coefficient:", np.around(est_coefs, 2))
    >>> Estimated coefficient: [ 0.    0.    0.   82.19  0.   88.31 70.05  0.    0.    0.  ]
    print("True coefficient:", np.around(true_coefs, 2))
    >>> True coefficient: [ 0.    0.    0.   82.19  0.   88.31 70.05  0.    0.    0.  ]

In the output, we can observe that the estimated coefficients closely resemble the true coefficients.

Further reading
---------------------------

- `JAX library <https://jax.readthedocs.io/en/latest/index.html>`__

- A bunch of `machine learning methods <examples/index.html>`__ implemented on the ``skscope``

- More `advanced features <../feature/index.html>`__ implemented in ``skscope`` 

Footnotes
---------------------------

.. [*] For simplicity, we just introduce the purpose of ``JAX`` library. For more information, please refer to `JAX <https://jax.readthedocs.io/en/latest/index.html>`__. 

.. [*] If you know nothing about ``numpy``, we can turn to `this introduction <https://numpy.org/doc/stable/user/whatisnumpy.html>`__.

.. [*] We skip the algorithmic detail about ``scopeSolver``. Please turn the paper "sparsity-constrained optimization via splicing iteration" if your are interested in. 