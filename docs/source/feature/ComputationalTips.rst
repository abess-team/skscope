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


.. Build with C++
.. -------------------