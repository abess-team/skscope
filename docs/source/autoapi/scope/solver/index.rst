:py:mod:`scope.solver`
======================

.. py:module:: scope.solver


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scope.solver.ScopeSolver
   scope.solver.HTPSolver
   scope.solver.IHTSolver
   scope.solver.GraspSolver
   scope.solver.FobaSolver
   scope.solver.ForwardSolver
   scope.solver.OMPSolver




.. py:class:: ScopeSolver(dimensionality, sparsity=None, sample_size=1, *, always_select=[], numeric_solver=convex_solver_nlopt, max_iter=20, ic_type='aic', ic_coef=1.0, cv=1, split_method=None, deleter=None, cv_fold_id=None, group=None, warm_start=True, important_search=128, screening_size=-1, max_exchange_num=5, is_dynamic_max_exchange_num=True, greedy=True, splicing_type='halve', path_type='seq', gs_lower_bound=None, gs_upper_bound=None, use_hessian=False, thread=1, jax_platform='cpu', random_state=None, console_log_level='off', file_log_level='off', log_file_name='logs/scope.log')

   Bases: :py:obj:`sklearn.base.BaseEstimator`

   
   Get sparse optimal solution of convex objective function by sparse-Constrained Optimization via Splicing Iteration (SCOPE) algorithm, which also can be used for variables selection.
   Specifically, ScopeSolver aims to tackle this problem: min_{x} f(x) s.t. ||x||_0 <= s, where f(x) is a convex objective function and s is the sparsity level. Each element of x can be seen as a variable, and the nonzero elements of x are the selected variables.

   :param + dimensionality: Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as p.
   :type + dimensionality: int
   :param + sparsity: The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as s. If sparsity is an array-like, it should be a list of integers.
                      default is `range(min(n, int(n/(log(log(n))log(p)))))`
                      Used only when path_type = "seq".
   :type + sparsity: int or array-like, optional, default=None
   :param + sample_size: sample size, only used in the selection of support size, denoted as n.
   :type + sample_size: int, optional, default=1
   :param + always_select: An array contains the indexes of variables which must be selected.
   :type + always_select: array-like, optional, default=[]
   :param + group: The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
                   The variables in the same group must be adjacent, and they will be selected together or not.
                   Here are wrong examples: [0,2,1,2] (not incremental), [1,2,3,3] (not start from 0), [0,2,2,3] (there is a gap).
                   It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example, "sparsity=[3]" means there will be 3 groups of variables selected rather than 3 variables,
                   and "always_include=[0,3]" means the 0-th and 3-th groups must be selected.
   :type + group: array-like with shape (p,), optional, default=range(p)
   :param + warm_start: When tuning the optimal parameter combination, whether to use the last solution as a warm start to accelerate the iterative convergence of the splicing algorithm.
   :type + warm_start: bool, optional, default=True
   :param + important_search: The number of important variables which need be splicing.
                              This is used to reduce the computational cost. If it's too large, it would greatly increase runtime.
   :type + important_search: int, optional, default=128
   :param + screening_size: The number of variables remaining after the screening before variables select. Screening is used to reduce the computational cost.
                            `screening_size` should be a non-negative number smaller than p,
                            but larger than any value in sparsity.
                            - If screening_size=-1, screening will not be used.
                            - If screening_size=0, screening_size will be set as
                              `min(p, int(n / (log(log(n))log(p))))`.
   :type + screening_size: int, optional, default=-1
   :param + max_iter: Maximum number of iterations taken for the
                      splicing algorithm to converge.
                      The limitation of objective reduction can guarantee the convergence.
                      The number of iterations is only to simplify the implementation.
   :type + max_iter: int, optional, default=20
   :param + max_exchange_num: Maximum exchange number when splicing.
   :type + max_exchange_num: int, optional, default=5
   :param + splicing_type: The type of reduce the exchange number in each iteration
                           from max_exchange_num.
                           "halve" for decreasing by half, "taper" for decresing by one.
   :type + splicing_type: {"halve", "taper"}, optional, default="halve"
   :param + path_type: The method to be used to select the optimal support size.
                       - For path_type = "seq", we solve the problem for all sizes in `sparsity` successively.
                       - For path_type = "gs", we solve the problem with support size ranged between `gs_lower_bound` and `gs_upper_bound`, where the specific support size to be considered is determined by golden section.
   :type + path_type: {"seq", "gs"}, optional, default="seq"
   :param + gs_lower_bound: The lower bound of golden-section-search for sparsity searching.
                            Used only when path_type = "gs".
   :type + gs_lower_bound: int, optional, default=0
   :param + gs_upper_bound: The higher bound of golden-section-search for sparsity searching.
                            Used only when path_type = "gs".
   :type + gs_upper_bound: int, optional, default=`min(n, int(n/(log(log(n))log(p))))`
   :param + ic_type: The type of information criterion for choosing the support size.
                     Used only when `cv` = 1.
   :type + ic_type: {'aic', 'bic', 'gic', 'ebic'}, optional, default='gic'
   :param + ic_coef: The coefficient of information criterion.
                     Used only when `cv` = 1.
   :type + ic_coef: float, optional, default=1.0
   :param + cv: The folds number when use the cross-validation method.
                - If `cv`=1, cross-validation would not be used.
                - If `cv`>1, support size will be chosen by CV's test objective,
                  instead of information criterion.
   :type + cv: int, optional, default=1
   :param + cv_fold_id: An array indicates different folds in CV.
                        Samples in the same fold should be given the same number.
                        The number of different masks should be equal to `cv`.
                        Used only when `cv` > 1.
   :type + cv_fold_id: array-like with shape (n,), optional, default=None
   :param + regular_coef: L2 regularization coefficient for computational stability.
                          Note that if `regular_coef` is not 0 and the length of `sparsity` is not 1, algorithm will compute full hessian matrix of objective function, which is time-consuming.
   :type + regular_coef: float, optional, default=0.0
   :param + thread: Max number of multithreads. Only used for cross-validation.
                    - If thread = 0, the maximum number of threads supported by
                      the device will be used.
   :type + thread: int, optional, default=1
   :param + console_log_level: The level of output log to console, which can be "off", "error", "warning", "debug". For example, if it's "warning", only error and warning log will be output to console.
   :type + console_log_level: str, optional, default="off"
   :param + file_log_level: The level of output log to file, which can be "off", "error", "warning", "debug". For example, if
                            it's "off", no log will be output to file.
   :type + file_log_level: str, optional, default="off"
   :param + log_file_name: The name (relative path) of log file, which is used to store the log information.
   :type + log_file_name: str, optional, default="logs/scope.log"

   .. attribute:: params

      The sparse optimal solution

      :type: array-like, shape(p, )

   .. attribute:: cv_test_loss

      If cv=1, it stores the score under chosen information criterion.
      If cv>1, it stores the test objective under cross-validation.

      :type: float

   .. attribute:: cv_train_loss

      The objective on training data.

      :type: float

   .. attribute:: value_of_objective

      The value of objective function on the solution.

      :type: float

   .. rubric:: Examples

   .. rubric:: References

   - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang.
     A polynomial algorithm for best-subset selection problem.
     Proceedings of the National Academy of Sciences,
     117(52):33117-33123, 2020.















   ..
       !! processed by numpydoc !!
   .. py:method:: get_config(deep=True)


   .. py:method:: set_config(**params)


   .. py:method:: get_estimated_params()

      
      Get the parameters of optimization.
















      ..
          !! processed by numpydoc !!

   .. py:method:: get_support()

      
      Get the support set of optimization.
















      ..
          !! processed by numpydoc !!

   .. py:method:: solve(objective, data=(), init_support_set=None, init_params=None, gradient=None, hessian=None, cpp=False, jit=False)

      
      Set the optimization objective and begin to solve

      :param + objective: The objective function to be minimized: ``objective(params, *data) -> float``
                          where ``params`` is a 1-D array with shape (dimensionality,) and
                          ``data`` is a tuple of the fixed parameters needed to completely specify the function.
                          ``objective`` must be written in JAX if gradient and hessian are not provided.
                          If `cpp` is `True`, `objective` can be a wrap of Cpp overloaded function which defined the objective of optimization with Cpp library `autodiff`, examples can be found in https://github.com/abess-team/scope_example.
      :type + objective: callable
      :param + data: Extra arguments passed to the objective function and its derivatives (if existed).
      :type + data: tuple, optional
      :param + init_support_set: The index of the variables in initial active set.
      :type + init_support_set: array-like of int, optional, default=[]
      :param + init_params: An initial value of parameters.
      :type + init_params: array-like of float, optional, default is an all-zero vector
      :param + gradient: Defined the gradient of objective function, return the gradient of parameters in `compute_index`.
      :type + gradient: function('params': array, 'data': custom class, 'compute_index': array) -> array
      :param + hessian: Defined the hessian of objective function, return the hessian matrix of the parameters in `compute_index`.
      :type + hessian: function('params': array, 'data': custom class, 'compute_index': array) -> 2D array
      :param + cpp: If `cpp` is `True`, `objective` must be a wrap of Cpp overloaded function which defined the objective of optimization with Cpp library `autodiff`, examples can be found in https://github.com/abess-team/scope_example.
      :type + cpp: bool, optional, default=False















      ..
          !! processed by numpydoc !!

   .. py:method:: get_result()

      
      Get the solution of optimization, include the parameters ...
















      ..
          !! processed by numpydoc !!


.. py:class:: HTPSolver(dimensionality, sparsity=None, sample_size=1, *, always_select=[], step_size=0.005, numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`scope.base_solver.BaseSolver`

   
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

.. py:class:: IHTSolver(dimensionality, sparsity=None, sample_size=1, *, always_select=[], step_size=0.005, numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`HTPSolver`

   
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

.. py:class:: GraspSolver(dimensionality, sparsity=None, sample_size=1, *, always_select=[], numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`scope.base_solver.BaseSolver`

   
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

.. py:class:: FobaSolver(dimensionality, sparsity=None, sample_size=1, *, use_gradient=False, threshold=0.0, foba_threshold_ratio=0.5, strict_sparsity=True, always_select=[], numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`scope.base_solver.BaseSolver`

   
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

.. py:class:: ForwardSolver(dimensionality, sparsity=None, sample_size=1, *, use_gradient=False, threshold=0.0, strict_sparsity=True, always_select=[], numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`FobaSolver`

   
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

.. py:class:: OMPSolver(dimensionality, sparsity=None, sample_size=1, *, threshold=0.0, strict_sparsity=True, always_select=[], numeric_solver=convex_solver_nlopt, max_iter=100, group=None, ic_type='aic', ic_coef=1.0, metric_method=None, cv=1, cv_fold_id=None, split_method=None, jax_platform='cpu', random_state=None)

   Bases: :py:obj:`ForwardSolver`

   
   Forward-gdt is equivalent to the orthogonal matching pursuit.
















   ..
       !! processed by numpydoc !!

