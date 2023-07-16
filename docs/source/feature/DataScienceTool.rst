Features for Data Science
=========================

Data science has emerged as a highly sought-after field due to its immense potential in extracting valuable insights and making informed decisions from large volumes of data. It combines various disciplines such as statistics, mathematics, computer science, and domain knowledge to analyze complex datasets and uncover patterns, trends, and correlations.

Currently, ``skscope`` develops two features that is helpful for users in data science community. 

- The first feature facilitate the machine/statistical learning method developed upon ``skscope`` can be directly used by more users. 

- The second feature provides benchmarked method for selecting the optimal support size :math:`s` according to the collected data. 

Data-dependence Objective Function
------------------------------------

Imaging you have developed a novel statistical/machine learning method based on ``skscope``, and you are willing that your method can be used by other users. If the objective function programmed with parameters as input and data in the context of its definition, it will cause many inconvenience. We can see this from two aspects. First, users shall always go to correct understand your programming which is time consuming if the objective function is complicated. Second, users have to modify your objective function based on their data. This is even more inconvenience because some subtle reason can cause the programming fail. Moreover, both of two aspects wastes a lot of time of users, which leads to even more inconvenience of users. 

To make the method machine learning method based on ``skscope`` being more accessible to data science community, we carefully design ``skscope`` such that it can use different data with the same objective function. 

Below gives an example where the objective function take both parameters and data as input. 

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

For other users, they can directly use your implementation even their dataset has different size (below the new dataset includes 200 predictors):

.. code-block:: python

    from sklearn.datasets import make_regression
    from skscope import ScopeSolver
    p, k = 200, 5
    X, y = make_regression(n_features=p, n_informative=k)
    new_data = (X, y)
    solver = ScopeSolver(p, k)
    new_params = solver.solve(custom_objective, new_data)

You can further wrap your implementation so that the usage is even more easier. 

.. code-block:: python

    ## program a wrapped machine learning method:
    def SparseRegressor(data, sparsity):
        p = data[0].shape[1]
        solver = ScopeSolver(p, sparsity)
        est_params = solver.solve(custom_objective, data)
        return est_params

With this wrapped machine learning method, users can use one-line python command to apply it on their dataset:

.. code-block:: python

    SparseRegressor(new_data, sparsity=5)


Optimal Support Size Searching
------------------------------

In the previous section, we have introduced how to set the sparsity level. However, sometimes we do not know the sparsity level and need to search it. In this case, we can set ``sparsity`` as a list of int, and the solver will search the best sparsity level from the given list.

Note that ``sample_size`` must be offered to ``ScopeSolver`` when ``sparsity`` is a list.


.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p,         ## there are p parameters
        sparsity=[1, 2, 3, 4, 5], ## the candidate support sizes
        sample_size=n,            ## the number of samples
    )


There are two ways to evaluate sparsity levels: `Information Criterion`_ and `Cross Validation`_.


Information Criterion
^^^^^^^^^^^^^^^^^^^^^^^^^

There is another way to evaluate sparsity levels, which is information criterion. The larger the information criterion, the better the model. There are four types of information criterion can be used in ``skscope``: Akaike information criterion `[1]`_, Bayesian information criterion (BIC, `[2]`_), extend BIC `[3]`_, and special information criterion (SIC `[4]`_). 

If sparsity is list and ``cv=None``, the solver will use information criterions to evaluate the sparsity level. We can use the input parameter ``ic`` in the solvers of ``skscope`` to choose information criterions, default is ``ic='gic'``. Here is an example that using the SIC to find the optimal support size. 

.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p,        
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n,           ## the number of samples
        ic='gic'                 ## use GIC to evaluate sparsity levels
    )


Cross Validation
^^^^^^^^^^^^^^^^^^^^

For cross validation `[5]`_, there are some requirements:
    
1. The objective function must take data as input.
    
.. code-block:: python

    import numpy as np
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
    
    
2. The data needs to be split into training and validation set. We can use ``set_split_method`` to set the split method. The split method must be a function that takes two arguments: ``data`` and ``index``, and returns a new data object. The ``index`` is the index of training set.
    
.. code-block:: python

    def split_method(data, index):
        return (data[0][index, :], data[1][index])
    
1. When initializing solvers, ``sample_size`` and ``cv`` must be offered. If ``cv`` is not None, the solver will use cross validation to evaluate the sparsity level. ``cv`` is the number of folds.
   
.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5], ## we want to select 1-5 variables
        sample_size=n, ## the number of samples
        split_method=split_method, ## use split_method to split data
        cv=10 ## use cross validation
    )

    params = solver.solve(custom_objective, data = (X, y))

There is a simpler way to use cross validation: let custom data be indeies of training set. In this case, we do not need to set ``split_method``.

.. code-block:: python
    
    import numpy as np
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
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n, ## the number of samples
        cv=10 ## use cross validation
    )

    params = solver.solve(custom_objective)


- _`[1]` Akaike, H. (1998). Information theory and an extension of the maximum likelihood principle. In Selected papers of hirotugu akaike (pp. 199-213). New York, NY: Springer New York.

- _`[2]` Schwarz, G. (1978). Estimating the dimension of a model. The annals of statistics, 461-464.

- _`[3]` Chen, J., & Chen, Z. (2008). Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3), 759-771.

- _`[4]` Zhu, J., Wen, C., Zhu, J., Zhang, H., & Wang, X. (2020). A polynomial algorithm for best-subset selection problem. Proceedings of the National Academy of Sciences, 117(52), 33117-33123.

- _`[5]` Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.