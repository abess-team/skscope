:parenttoc: True

Computational Tips
=============================


Just In Time Compilation
--------------------------------------------------

`Just In Time Compilation (JIT) <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_ is a technique used in the ``JAX`` library to optimize and accelerate the execution of numerical computations. And thus, it can make solvers in ``skscope`` be executed more efficiently. We can easily use JIT to speed up the execution of solvers by setting the ``jit=True`` in the ``solve`` method: 

.. code-block:: python
    
    from skscope import ScopeSolver

    def objective_fn(params):
        value = 0.0
        # do somethings
        return value
    solver = ScopeSolver(
        dimensionality=10,  ## there are p parameters
        sparsity=3,         ## the candidate support sizes
    )
    solver.solve(objective_fn, jit=True)

The runtime comparison on the JIT mode is on or off shows that, JIT generally speedup computation, ranging from 2 to 30 times. Here are the ratios of the runtime of non-JIT mode to JIT mode for different solvers on different problems:

+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+
|             | Linear regression | Logistic regression | Multi-task learning | Nonlinear feature selection | Trend filtering | Ising model |
+=============+===================+=====================+=====================+=============================+=================+=============+
| FobaSolver  | 11.93             | 19.16               | 7.02                | 4.32                        | 2.97            | 14.76       |
+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+
| GraspSolver | 5.76              | 31.63               | 6.73                | 1.07                        | 1.81            | 10.34       |
+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+
| HTPSolver   | 5.34              | 13.55               | 11.16               | 1.21                        | 0.89            | 13.26       |
+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+
| IHTSolver   | 1.06              | 3.28                | 1.84                | 0.53                        | 0.25            | 4.89        |
+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+
| OMPSolver   | 11.33             | 20.88               | 9.82                | 2.83                        | 0.9             | 16.16       |
+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+
| ScopeSolver | 5.24              | 17.26               | 2.06                | 2.01                        | 3.21            | 11.21       |
+-------------+-------------------+---------------------+---------------------+-----------------------------+-----------------+-------------+




    > Note that JIT need additional requirements on the programming of objective function. More details can be found in `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_.


Support GPU device
--------------------------------------------------

``skscope`` does not exclude the use of GPU devices for computation. 
In fact, when users correctly install the matching JAX with the physical device, they can use GPU for computation without any additional commands.

    > JAX runs transparently on the GPU or TPU (falling back to CPU if you don't have one).

In order to ensure universality, ``skscope`` relies on the only CPU version of the JAX. 
Therefore, for users who want to use GPUs, they only need to follow the `instructions <https://jax.readthedocs.io/en/latest/installation.html>`_ 
and correctly install the JAX version that matches the physical device. For example:

.. code-block:: Bash

    pip install -U "jax[cuda12]"



Support Sparse Matrices
--------------------------------------------------

Thanks to ``jax``, ``skscope`` supports input matrices as sparse matrices. Although using sparse matrices increases the time required for automatic differentiation, 
it can significantly reduce memory usage. Below, we provide an example of linear regression to demonstrate this functionality. First, we import the necessary libraries and filter out warnings for cleaner output.

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from jax.experimental import sparse
    from skscope import ScopeSolver
    import scipy.sparse as sp

    import warnings
    warnings.filterwarnings('ignore')

Next, we generate a random sparse matrix using ``scipy.sparse``, convert it to a dense matrix with ``JAX``, and then convert it to a BCOO format sparse matrix. We also create a target vector based on a predefined vector with some added noise.

.. code-block:: python

    n, p = 150, 30
    np.random.seed(0)
    random_sparse_matrix = sp.random(n, p, density=0.1, format='coo')
    dense_matrix = jnp.array(random_sparse_matrix.toarray())
    X = sparse.BCOO.fromdense(dense_matrix)
    beta = np.zeros(p)
    beta[:3] = [1, 2, 3]
    y = X @ beta + np.random.normal(0, 0.1, n)

We define a simple ordinary least squares (OLS) loss function to be minimized by ``ScopeSolver``.

.. code-block:: python

    def ols_loss(params):
        loss = jnp.mean((y - X @ params) ** 2)
        return loss

Finally, we initialize the ``ScopeSolver``, specifying the number of features to select, and solve for the optimal parameters.

.. code-block:: python

    solver = ScopeSolver(p, sparsity=3)
    params_skscope = solver.solve(ols_loss, jit=True)

Then, we can get ``params_skscope`` as the result of the subset selection.

.. Build with C++
.. -------------------