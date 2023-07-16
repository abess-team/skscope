:parenttoc: True

Quick Example for Beginner
============================

Introduction
-----------------

.. Here, we will take linear regression as an start-up example to introduce the basic usage of :ref:`scope <skscope_package>`.

Presume we collect a dataset with :math:`n=100` independent observations on a interesting response variable and :math:`p=10` predictive variables: 

.. code-block:: python

    from sklearn.datasets import make_regression
    ## generate data
    n, p, s = 100, 10, 3
    X, y, true_coefs = make_regression(n_samples=n, n_features=p, n_informative=s, coef=True, random_state=0) 
    print("X shape:", X.shape)
    >>> X shape: (100, 10)
    print("Y shape:", y.shape)
    >>> Y shape: (100,)

We are interested in explaining the variation of the response variable with :math:`p` predictive variable, but in practice, the underlying true parameters ``true_coefs`` is unknown. Our only knowledge is that only :math:`s=3` predictive variables would influence the response variable, and the relationship between the predictive variables and the response variable has a form of linear model: 

.. math::
    y=X \theta^{*} +\epsilon.

where 

- :math:`y \in R^n` and :math:`X \in R^{n\times p}` records the observations on the response variable and the predictive variables, respectively;

- :math:`\epsilon = (\epsilon_1, \ldots, \epsilon_n)` is a vector that contact :math:`n` i.i.d zero-mean random noises;

- :math:`\theta^{*}` is an unknown coefficient vector to be estimated.  

With ``skscope``, users can well estimate the :math:`\theta^{*}` and find out the predictive variables that is influential of the response variable, so called variables selection in literature. 
To see this, a present a step-by-step procedure below. 

A Step-by-step procedure
-------------------------------

Sparsity-constrained optimization perspective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first formulate the variables selection problem as an optimization problem:

.. math::
    \arg\min_{\theta \in R^p} ||y-X \theta||^{2} s.t. ||\theta||_0 \leq s,

where

- :math:`\theta` is a coefficient vector that needs to be optimized

- :math:`||y-X \theta||^{2}` is the objective function that quantitatively measures the quality of the coefficient vector :math:`\theta`; 

- :math:`||\theta||_0` counts the number of non-zero element in :math:`\theta`. And thus, the sparsity constraint :math:`||\theta||_0 \leq s` reflects the prior knowledge that the number of effective predictive variable is less than :math:`s`.

The intuitive explanation for the optimization problem is: finding a small subset of predictors, so that the resulting linear model is expected to have the most desirable prediction accuracy. Therefore, it fully leverages information the observed data and our prior knowledge. 


Solve SCO via ``skscope``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above optimization problem is a sparsity-constrained optimization (SCO) problem. Since ``skscope`` is designed for general sparsity-constrained optimization, it can definitively solve the SCO problem. More importantly, ``skscope`` is extremely easy to use --- it can solve the SCO problem so long as the objective function :math:`||y-X \theta||^{2}` is properly implemented. Next, for the above example, the objective function is programmed into a python function ``objective_function(coefs)``:

.. code-block:: python

    import jax.numpy as jnp
    def objective_function(coefs):
        return jnp.linalg.norm(y - X @ coefs)

Notice that, the first line imports the only required ``jax`` library [*]_, which is a Python library that provides a NumPy-compatible multidimensional array API and automatic differentiation. As the usage of ``jax`` is very similar to ``numpy``, we can simply understand ``jax.numpy`` is equivalent to ``numpy`` [*]_. 

Moreover, we can see that the programmed ``objective_function`` is exactly match to the mathematics function :math:`||y-X \theta||^{2}`. 

Then, we will input ``objective_function`` into one solver in ``skscope`` to get the practice solution of the SCO problem. 

.. The length of ``coefs`` is actually the dimension of the optimization problem so denoted as ``dimensionality``. The number of nonzero parameters is the sparsity level and denote as ``sparsity``.

.. From the perspective of variable selection, each parameter corresponds to a variable, and the nonzero parameters correspond to the selected variables.

Here, we import the ``ScopeSolver`` [*]_ and properly configure it. 

.. code-block:: python

    from skscope import ScopeSolver
    scope_solver = ScopeSolver(
        dimensionality=p,  ## there are p parameters
        sparsity=s,        ## we want to select s variables
    )

In our configuration, we set:

- ``dimensionality``, i.e., is the number of parameters and must be offered.

and 

- ``sparsity``, i.e., the desired sparsity level. 

In the above example, :math:`\beta` is the parameters, so ``dimensionality`` is :math:`p` and ``sparsity`` is :math:`s`.

Those concepts are introduced in the previous section. 

With ``scope_solver`` and ``objective_function``, we use one-line command to solve the SCO:

.. code-block:: python

    scope_solver.solve(objective_function)


``solve`` is the main method of solver in ``skscope``, it takes the objective function as optimization objective and commands the algorithm to conduct the optimization process. 

Get solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the solvers in ``skscope`` are all coupled with necessary functions to extract, we can also use one line to get the result. For example, we may interested in obtaining the effective variables, which can be extract by using 
the ``get_support`` method. The code is below. 

.. code-block:: python

    import numpy as np
    est_support_set = scope_solver.get_support()
    print("Estimated effective predictors:", est_support_set)
    >>> Estimated effective predictors: [3 5 6]
    print("True effective predictors:", np.nonzero(true_coefs)[0])
    >>> True effective predictors: [3 5 6]

We can see that the estimated effective predictive variables are exactly the true one, which reflects the accuracy of the solver in ``skscope``.

Else, we may interested in the regression coefficient:

- ``get_estimated_params`` returns the optimized coefficient.

.. code-block:: python

    est_coefs = scope_solver.get_estimated_params()
    print("Estimated coefficient:", np.around(est_coefs, 2))
    >>> Estimated coefficient: [ 0.    0.    0.   82.19  0.   88.31 70.05  0.    0.    0.  ]
    print("True coefficient:", np.around(true_coefs, 2))
    >>> True coefficient: [ 0.    0.    0.   82.19  0.   88.31 70.05  0.    0.    0.  ]

For the output, we see that the estimated coefficient approaches to the underlying true coefficient. 

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