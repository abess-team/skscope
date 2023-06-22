:py:mod:`scope.model`
=====================

.. py:module:: scope.model


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   scope.model.quadratic_objective



.. py:function:: quadratic_objective(Q, p, autodiff=False)

   
   Create a model of quadratic objective function which is $L(x) = <x, Qx> / 2 + <p, x>$.

   :param + Q: The matrix of quadratic term.
   :type + Q: array-like, shape (n_features, n_features)
   :param + p: The vector of linear term.
   :type + p: array-like, shape (n_features,)
   :param + autodiff: Whether to return a overloaded function of objective function for cpp library `autodiff`.
   :type + autodiff: bool, default False

   :returns: * A dict of quadratic model for `solve()`, which may contain part of the following keys
             * **+ objective** (*function('params': array, 'data': Any) ->  float*) -- The objective function.
             * **+ gradient** (*function('params': array, 'data': Any) -> 1-D array*) -- The gradient of objective function.
             * **+ hessian** (*function('params': array, 'data': Any) -> 2-D array*) -- The hessian of objective function.
             * **+ data** (*Any*) -- The data for above functions.

   .. rubric:: Examples

   ```
   import numpy as np
   from scope import ScopeSolver, GraspSolver
   from scope.model import quadratic_objective

   model = quadratic_objective(np.eye(5), np.ones(5))
   solver1 = ScopeSolver(dimensionality=5)
   solver1.solve(
       model["objective"],
       model["data"],
       gradient=model["gradient"],
       hessian=model["hessian"],
   )
   print(solver1.get_result())

   solver2 = GraspSolver(dimensionality=5)
   solver2.solve(
       model['objective'],
       model['data'],
       gradient = model['gradient'],
   )
   print(solver2.get_result())
   ```















   ..
       !! processed by numpydoc !!

