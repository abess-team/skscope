:py:mod:`scope.numeric_solver`
==============================

.. py:module:: scope.numeric_solver


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   scope.numeric_solver.convex_solver_nlopt



.. py:function:: convex_solver_nlopt(loss_fn, value_and_grad, params, optim_variable_set, data)

   
   A wrapper of nlopt solver for convex optimization.

   Nlopt often throws RuntimeError even if the optimization is nearly successful.
   This function is used to cache the best result and return it.

   :param loss_fn: The loss function.
   :type loss_fn: Callable[[Sequence[float], Any], float]
   :param value_and_grad: The function to compute the loss and gradient.
   :type value_and_grad: Callable[[Sequence[float], Any], Tuple[float, Sequence[float]]]
   :param params: The complete initial parameters.
   :type params: Sequence[float]
   :param optim_variable_set: The index of variables to be optimized.
   :type optim_variable_set: Sequence[int]
   :param data: The data passed to loss_fn and value_and_grad.
   :type data: Any

   :returns: * **loss** (*float*) -- The loss of the optimized parameters, i.e., `loss_fn(params, data)`.
             * **optimized_params** (*Sequence[float]*) -- The optimized parameters.















   ..
       !! processed by numpydoc !!

