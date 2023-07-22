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

The runtime comparison on the JIT mode is on or off shows that, JIT generally speedup 10 times. 

+-----------------------+---------------+----------------+---------------+---------------+---------------+---------------+
| Method                | linear        | logistic       | trend\_filter | multitask     | ising         | non\_linear   |
+=======================+===============+================+===============+===============+===============+===============+
| FobaSolver(non-JIT)   | 132.04(17.53) | 116.23(33.13)  | 5.46(2.24)    | 46.59(6.96)   | 165.58(42.08) | 132.27(32.48) |
| FobaSolver(JIT)       | 11.7(2.9)     | 6.31(2.15)     | 1.89(0.84)    | 6.65(0.65)    | 11.59(3.55)   | 31.26(8.8)    |
| GraspSolver(non-JIT)  | 6.57(2.35)    | 408.14(246.79) | 3.14(1.95)    | 5.95(2.75)    | 10.62(5.77)   | 7.65(3.05)    |
| GraspSolver(JIT)      | 1.16(0.38)    | 12.7(8.2)      | 1.74(0.74)    | 0.87(0.31)    | 1.02(0.44)    | 7.34(2.75)    |
| HTPSolver(non-JIT)    | 21.7(8.55)    | 32.13(17.28)   | 1.46(0.82)    | 235.89(64.94) | 70.71(34.32)  | 12.61(8.33)   |
| HTPSolver(JIT)        | 4.14(1.25)    | 2.37(0.92)     | 1.71(0.85)    | 21.5(4.56)    | 5.26(2.03)    | 10.82(7.86)   |
| IHTSolver(non-JIT)    | 3.54(1.11)    | 3.59(3.77)     | 0.68(0.35)    | 2.26(0.49)    | 15.23(5.69)   | 3.36(2.06)    |
| IHTSolver(JIT)        | 3.42(0.88)    | 1.06(0.67)     | 2.8(1.35)     | 1.24(0.23)    | 3.24(1.43)    | 6.37(2.32)    |
| OMPSolver(non-JIT)    | 26.74(5.99)   | 32.78(10.32)   | 1.43(0.68)    | 39.8(5.22)    | 45.64(14.47)  | 32.61(10.47)  |
| OMPSolver(JIT)        | 2.45(0.68)    | 1.66(0.67)     | 1.6(0.65)     | 4.07(0.48)    | 2.86(0.86)    | 11.53(3.61)   |
| ScopeSolver(non-JIT)  | 10.73(3.36)   | 53.88(38.12)   | 8.93(3.54)    | 14.02(9.26)   | 18.65(7.79)   | 17.03(5.52)   |
| ScopeSolver(JIT)      | 2.11(0.7)     | 3.24(2.67)     | 2.85(1.29)    | 7.28(4.84)    | 1.69(0.67)    | 8.6(2.7)      |
+-----------------------+---------------+----------------+---------------+---------------+---------------+---------------+



    > Note that JIT need additional requirements on the programming of objective function. More details can be found in `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_.


.. Build with C++
.. -------------------