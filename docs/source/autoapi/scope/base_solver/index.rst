:py:mod:`scope.base_solver`
===========================

.. py:module:: scope.base_solver


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scope.base_solver.BaseSolver




.. py:class:: BaseSolver(dimensionality, sparsity=None, sample_size=1, *, always_select=[], numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`sklearn.base.BaseEstimator`

   
   # attributes
   params: ArrayLike | None
   support_set: ArrayLike | None
   value_of_objective: float
   eval_objective: float

       dimensionality: int,
       sparsity: int | ArrayLike | None = None,
       sample_size: int = 1,
       *,
       max_iter: int = 1000,
       ic_type: str = "aic",
       ic_coef: float = 1.0,
       metric_method: Callable = None,
           (loss: float, p: int, s: int, n: int) -> float
       cv: int = 1,
       split_method: Callable[[Any, ArrayLike], Any] | None = None,
       random_state: int | np.random.RandomState | None  = None,















   ..
       !! processed by numpydoc !!
   .. py:method:: get_config(deep=True)


   .. py:method:: set_config(**params)


   .. py:method:: solve(objective, data=(), gradient=None, init_support_set=None, init_params=None, jit=False)

      
      Set the optimization objective and begin to solve

      :param + objective: Defined the objective of optimization, must be written in JAX if gradient is not provided.
      :type + objective: function('params': array of float(, 'data': custom class)) ->  float
      :param + gradient: Defined the gradient of objective function, return the gradient of parameters.
      :type + gradient: function('params': array of float(, 'data': custom class)) -> array of float
      :param + init_support_set: The index of the variables in initial active set.
      :type + init_support_set: array-like of int, optional, default=[]
      :param + init_params: An initial value of parameters.
      :type + init_params: array-like of float, optional, default is an all-zero vector
      :param + data: The data that objective function should be known, like samples, responses, weights, etc, which is necessary for cross validation. It can be any class which is match to objective function.
      :type + data: custom class, optional, default=None
      :param + jit: just-in-time compilation with XLA, but it should be a pure function.
      :type + jit: bool, optional, default=False















      ..
          !! processed by numpydoc !!

   .. py:method:: get_result()

      
      Get the solution of optimization, include the parameters ...
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_estimated_params()

      
      Get the parameters of optimization.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_support()

      
      Get the support set of optimization.
















      ..
          !! processed by numpydoc !!


