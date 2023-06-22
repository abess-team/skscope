:parenttoc: True

Computational Tips
=============================


Just In Time Compilation
--------------------------------------------------

`Just In Time Compilation (JIT) <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_ is a technology which can make Python function be executed more efficiently.

We can easily use jit to speed up the execution of ``solve`` by setting the ``jit`` parameter to ``True``.

.. code-block:: python

    solver.solve(objective_fn, jit=True)


Note that JIT need additional requirements for the objective function and makes it difficult to debug. 
Details can be found in `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_.

Build with C++
-------------------


change numeric solver platform
---------------------------------------------------------

All solvers in :ref:`scope <scope_package>` use `nlopt <https://nlopt.readthedocs.io/en/latest/>`_ as the default numeric solver platform, but users can change it 
by setting the ``numeric_solver`` parameter to a function which has the same interface as ``scope.numeric_solver.convex_solver_nlopt``.

.. code-block:: python

    from scope import ScopeSolver
    from scope.numeric_solver import convex_solver_nlopt
    def custom_numeric_solver(*args, **kwargs):
        # do something
        return convex_solver_nlopt(*args, **kwargs)

    p, k = 10, 3
    solver = ScopeSolver(p, k, numeric_solver=custom_numeric_solver)