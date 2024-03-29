Features for Data Science
=========================

Data science has emerged as a highly sought-after field due to its immense potential in extracting valuable insights and making informed decisions from large volumes of data. It combines various disciplines such as statistics, mathematics, computer science, and domain knowledge to analyze complex datasets and uncover patterns, trends, and correlations.

Currently, ``skscope`` develops two features that is helpful for users in data science community. 

- The first feature facilitate people use the machine/statistical learning method developed upon ``skscope``. 

- The second feature provides benchmarked methods for selecting the optimal support size :math:`s` according to the collected data. 

Data-dependence Objective Function
------------------------------------

Imagine that you have developed a novel statistical/machine learning method based on ``skscope``, and you want data scientists to be able to use this method. If the objective function is programmed with parameters as input and requires the data to be provided within its definition, then this programming style would raise two main issues. First, users will need to understand your specific programming implementation, which can be time-consuming if the objective function is complex. Second, users will have to modify your objective function based on their own data, which can be even more cumbersome and error-prone, as small mistakes can lead to failures in the programming. Both of these issues waste users' time and result in additional inconvenience.

To make machine learning methods based on ``skscope`` more accessible to the data science community, ``skscope`` is designed to allow the use of different datasets with the same objective function. Here's an example where the objective function takes both parameters and data as input:

.. code-block:: python
    
    import jax.numpy as jnp
    from sklearn.datasets import make_regression
    from skscope import ScopeSolver

    p, k = 100, 10
    X, y = make_regression(n_features=p, n_informative=k)
    your_data = (X, y)
    ## define objective function
    def custom_objective(params, data):
        return jnp.sum(
            jnp.square(data[1] - data[0] @ params)
        )
    solver = ScopeSolver(p, k)
    your_params = solver.solve(custom_objective, your_data)

For other users, they can directly use your implementation even if their dataset has a different size (e.g., the new dataset includes 200 predictors):

.. code-block:: python

    from sklearn.datasets import make_regression
    from skscope import ScopeSolver

    p, k = 200, 5
    X, y = make_regression(n_features=p, n_informative=k)
    new_data = (X, y)
    solver = ScopeSolver(p, k)
    new_params = solver.solve(custom_objective, new_data)

You can further wrap your implementation to make it even easier to use:

.. code-block:: python

    ## define a wrapped machine learning method:
    def SparseRegressor(data, sparsity):
        p = data[0].shape[1]
        solver = ScopeSolver(p, sparsity)
        est_params = solver.solve(custom_objective, data)
        return est_params

With this wrapped machine learning method, users can apply it to their dataset using a one-line Python command:

.. code-block:: python

    SparseRegressor(new_data, sparsity=5)


Optimal Support Size Searching
------------------------------

In other places, we presume the sparsity level would be appropriate set. However, there are cases where we do not know the optimal sparsity level and need to search for it. In such cases, we can set the ``sparsity`` parameter as a list of integers, and the solver will search for the best sparsity level from the given list.

Note that when using a list for ``sparsity``, the ``sample_size`` parameter must also be provided to the solver in ``skscope``.


.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p,         ## there are p parameters
        sparsity=[1, 2, 3, 4, 5], ## the candidate support sizes
        sample_size=n,            ## the number of samples
    )


There are two ways to evaluate sparsity levels: `Information Criterion`_ and `Cross Validation`_.


Information Criterion
^^^^^^^^^^^^^^^^^^^^^^^^^


Information criterion is a statistical measure used to assess the goodness of fit of a model while penalizing model complexity. It helps in selecting the optimal model from a set of competing models. In the context of sparsity-constrained optimization, information criterion can be used to evaluate different sparsity levels and identify the most suitable support size.
.. There is another way to evaluate sparsity levels, which is information criterion. The larger the information criterion, the better the model. 
There are four types of information criterion can be used in ``skscope``: Akaike information criterion `[1]`_, Bayesian information criterion (BIC, `[2]`_), extend BIC `[3]`_, and special information criterion (SIC `[4]`_). 
.. If sparsity is list and ``cv=None``, the solver will use information criterions to evaluate the sparsity level. 
The input parameter ``ic`` in the solvers of skscope can be used to choose the information criterion. The default value is ``ic='sic'``. Here is an example using SIC to find the optimal support size.

.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p,        
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n,           ## the number of samples
        ic='sic',                ## use SIC to evaluate sparsity levels
    )


Cross Validation
^^^^^^^^^^^^^^^^^^^^

Cross-validation is a technique used to assess the performance and generalization capability of a machine learning model. It involves partitioning the available data into multiple subsets, or folds, to train and test the model iteratively.

To utilizing cross validation `[5]`_, there are some requirements:
    
1. The objective function must take data as input.
    
.. code-block:: python

    import jax.numpy as jnp
    from sklearn.datasets import make_regression

    ## generate data
    n, p, k= 10, 5, 3
    X, y, true_params = make_regression(n_samples=n, n_features=p, n_informative=k, coef=True)
    ## define objective function
    def custom_objective(params, data):
        return jnp.sum(
            jnp.square(data[1] - data[0] @ params)
        )
    
    
2. The data needs to be split into training and validation sets. The ``split_method`` parameter is used to define the split method. The split method must be a function that takes two arguments: ``data`` and ``index``, and returns a new data object. The ``index`` parameter represents the indices of the training set.
    
.. code-block:: python

    def split_method(data, index):
        return (data[0][index, :], data[1][index])
    
3. When initializing solvers, ``sample_size`` and ``cv`` must be offered. Notice that, ``cv`` represents the number of folds in cross validation.
   
.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p,          ## there are p parameters
        sparsity=[1, 2, 3, 4, 5],  ## we want to select 1-5 variables
        sample_size=n,             ## the number of samples
        split_method=split_method, ## use split_method to split data
        cv=10,                     ## use 10-fold cross validation
    )

    params = solver.solve(custom_objective, data = (X, y))

There is a simpler way to use cross validation: let objective function comprise a additional parameter ``index``. In this case, we do not need to set ``split_method``. Below is the illustrative example for this usage. 

.. code-block:: python
    
    import jax.numpy as jnp
    from sklearn.datasets import make_regression

    ## generate data
    n, p, k= 10, 5, 3
    X, y, true_params = make_regression(n_samples=n, n_features=p, n_informative=k, coef=True)

    def custom_objective(params, index):
        return jnp.sum(
            jnp.square(y[index] - X[index,:] @ params)
        )
    
    solver = ScopeSolver(
        dimensionality=p,        ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n,           ## the number of samples
        cv=10,                   ## 10-folds use cross validation
    )

    params = solver.solve(custom_objective)


Reference
------------------------------

- _`[1]` Akaike, H. (1998). Information theory and an extension of the maximum likelihood principle. In Selected papers of hirotugu akaike (pp. 199-213). New York, NY: Springer New York.

- _`[2]` Schwarz, G. (1978). Estimating the dimension of a model. The annals of statistics, 461-464.

- _`[3]` Chen, J., & Chen, Z. (2008). Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3), 759-771.

- _`[4]` Zhu, J., Wen, C., Zhu, J., Zhang, H., & Wang, X. (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52), 33117-33123.

- _`[5]` Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.