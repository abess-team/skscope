:parenttoc: True

Computational Tips
=============================


Just In Time Compilation
--------------------------------------------------

`Just In Time Compilation (JIT) <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_ is a technology which can make solvers in ``skscope`` be executed more efficiently. We can easily use JIT to speed up the execution of solvers by setting the ``jit=True`` in the ``solve`` method: 

.. code-block:: python

    solver.solve(objective_fn, jit=True)

The runtime comparison on the JIT mode is on or off shows that, JIT generally speedup 10 times. 

    > Note that JIT need additional requirements on the programming of objective function. More details can be found in `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#>`_.


.. Build with C++
.. -------------------