:parenttoc: True

Variants
====================

Here we will introduce some variants of :ref:`scope <scope_package>`.

group variables selection
----------------------------

In the group variable selection, variables are divided into several non-overlapping groups and each group is treated as a whole, i.e., selected at the same time or not selected at the same time. If we want to select :math:`s` groups of variables among :math:`q` groups, the group variable selection can be formulated as this:

Let :math:`G=\{g_1, \dots, g_q\}` be a partition of the index set :math:`\{1, \dots, p\}`, where :math:`g_i \subset \{1, \dots, p\}` for :math:`i=1, \dots, q`, `g_i \cap g_j = \emptyset` for :math:`i \neq j` and :math:`\bigcup_{i=1}^q g_i = \{1, \dots, p\}`. Then optimize the objective function with constraints:


.. math::
	\min_{x \in R^p} L(x),\operatorname{ s.t. } \sum_{i=1}^q I({\|x}_{g_i}\|\neq 0) \leq s,

where :math:`s` is the group sparsity level, i.e., the number of groups to be selected. If :math:`q=p`, then :math:`g_i = \{i\}` for :math:`i=1, \dots, q`, so the group variable selection is equivalent to the original variable selection. 

In this case, we need to offer group information by ``group`` parameter. ``group`` is an incremental integer array of length ``dimensionality`` starting from 0 without gap, which means the variables in the same group must be adjacent, and they will be selected together or not.

Here are wrong examples: [0,2,1,2] (not incremental), [1,2,3,3] (not start from 0), [0,2,2,3] (there is a gap).

Note that ``params`` (parameters) must be a vector not matrix and ``sparsity`` represents the number of groups to be selected here.


.. code-block:: python

    p, s, m = 3, 2, 2
    params_true = [1, 10, 0, 0, -1, 5]
    ScopeSolver(
            dimensionality=p * m, 
            sparsity=s,
            group=[0,0,1,1,2,2]
        )


Search support size
-------------------------

In the previous section, we have introduced how to set the sparsity level. However, sometimes we do not know the sparsity level and need to search it. In this case, we can set ``sparsity`` as a list of int, and the solver will search the best sparsity level from the given list.

Note that ``sample_size`` must be offered to ``ScopeSolver`` when ``sparsity`` is a list.


.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n ## the number of samples
    )


There are two ways to evaluate sparsity levels:

cross validation
^^^^^^^^^^^^^^^^^^^^

For cross validation, there are some requirements:
1. When initializing ``ScopeSolver``, ``sample_size`` and ``cv`` must be offered. If ``cv`` is not None, the solver will use cross validation to evaluate the sparsity level. ``cv`` is the number of folds.
   
.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n, ## the number of samples
        cv=10 ## use cross validation
    )
    
2. The objective function must take data as input.
    
.. code-block:: python

        def custom_objective(params, data):
            return jnp.sum(
                jnp.square(data.y - data.X @ params)
            )
    
    If there are auxiliary parameters, the data must be the last argument.
    
.. code-block:: python

        def custom_objective(params, aux_params, data):
            return jnp.sum(
                jnp.square(data.y - data.X @ params - aux_params)
            )
    
3. The data needs to be split into training and validation set. We can use ``set_split_method`` to set the split method. The split method must be a function that takes two arguments: ``data`` and ``index``, and returns a new data object. The ``index`` is the index of training set.
    
.. code-block:: python

        def split_method(data, index):
            return CustomData(data.x[index, :], data.y[index])
        solver.set_split_method(split_method)
    


information criterion
^^^^^^^^^^^^^^^^^^^^^^^^^

There is another way to evaluate sparsity levels, which is information criterion. The larger the information criterion, the better the model. There are four types of information criterion can be used in SCOPE: 'aic', 'bic', 'gic', 'ebic'. If sparsity is list and ``cv`` is ``None``, the solver will use cross validation to evaluate the sparsity level. We can use ``ic`` to choose information criterions, default is 'gic'.

Here is an example:

.. code-block:: python

    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n, ## the number of samples
        ic='gic' ## use default way gic to evaluate sparsity levels
    )


The way of defining objective function is the same as common way.

always select some variables
--------------------------------