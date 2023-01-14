from .BaseSolver import BaseSolver
import numpy as np
import importlib
from .pybind_cabess import pywrap_Universal, UniversalModel, init_spdlog


class ScopeSolver(BaseSolver):
    r"""
    Get sparse optimal solution of convex objective function by sparse-Constrained Optimization via Splicing Iteration (SCOPE) algorithm, which also can be used for variables selection.
    Specifically, ScopeSolver aims to tackle this problem: min_{x} f(x) s.t. ||x||_0 <= s, where f(x) is a convex objective function and s is the sparsity level. Each element of x can be seen as a variable, and the nonzero elements of x are the selected variables.

    Parameters
    ----------
    + dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as p.
    + sparsity : int or array-like, optional, default=None
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as s. If sparsity is an array-like, it should be a list of integers. 
        default is `range(min(n, int(n/(log(log(n))log(p)))))`
        Used only when path_type = "seq".
    + aux_params_size : int, optional, default=0
        The total number of auxiliary variables, which means that they need to be considered in optimization but always be selected.
        This is for the convenience of some models, for example, the intercept in linear regression is an auxiliary variable.
    + sample_size : int, optional, default=1
        sample size, only used in the selection of support size, denoted as n.
    + always_select : array-like, optional, default=[]
        An array contains the indexes of variables which must be selected.
        Its effect is simillar to see these variables as auxiliary variables and set `aux_params_size`. 
    + group : array-like with shape (p,), optional, default=range(p)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: [0,2,1,2] (not incremental), [1,2,3,3] (not start from 0), [0,2,2,3] (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example, "sparsity=[3]" means there will be 3 groups of variables selected rather than 3 variables,
        and "always_include=[0,3]" means the 0-th and 3-th groups must be selected.
    + warm_start : bool, optional, default=True
        When tuning the optimal parameter combination, whether to use the last solution as a warm start to accelerate the iterative convergence of the splicing algorithm.
    + important_search : int, optional, default=128
        The number of important variables which need be splicing. 
        This is used to reduce the computational cost. If it's too large, it would greatly increase runtime.
    + screening_size : int, optional, default=-1
        The number of variables remaining after the screening before variables select. Screening is used to reduce the computational cost.
        `screening_size` should be a non-negative number smaller than p,
        but larger than any value in sparsity.
        - If screening_size=-1, screening will not be used.
        - If screening_size=0, screening_size will be set as
          `min(p, int(n / (log(log(n))log(p))))`.
    + max_iter : int, optional, default=20
        Maximum number of iterations taken for the
        splicing algorithm to converge.
        The limitation of objective reduction can guarantee the convergence.
        The number of iterations is only to simplify the implementation.
    + max_exchange_num : int, optional, default=5
        Maximum exchange number when splicing.
    + splicing_type : {"halve", "taper"}, optional, default="halve"
        The type of reduce the exchange number in each iteration
        from max_exchange_num.
        "halve" for decreasing by half, "taper" for decresing by one.
    + path_type : {"seq", "gs"}, optional, default="seq"
        The method to be used to select the optimal support size.
        - For path_type = "seq", we solve the problem for all sizes in `sparsity` successively.
        - For path_type = "gs", we solve the problem with support size ranged between `gs_lower_bound` and `gs_higher_bound`, where the specific support size to be considered is determined by golden section.
    + gs_lower_bound : int, optional, default=0
        The lower bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    + gs_higher_bound : int, optional, default=`min(n, int(n/(log(log(n))log(p))))`
        The higher bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    + ic_type : {'aic', 'bic', 'gic', 'ebic'}, optional, default='gic'
        The type of information criterion for choosing the support size.
        Used only when `cv` = 1.
    + ic_coef : float, optional, default=1.0
        The coefficient of information criterion.
        Used only when `cv` = 1.
    + cv : int, optional, default=1
        The folds number when use the cross-validation method.
        - If `cv`=1, cross-validation would not be used.
        - If `cv`>1, support size will be chosen by CV's test objective,
          instead of information criterion.
    + cv_fold_id: array-like with shape (n,), optional, default=None
        An array indicates different folds in CV.
        Samples in the same fold should be given the same number.
        The number of different masks should be equal to `cv`.
        Used only when `cv` > 1.
    + regular_coef : float, optional, default=0.0
        L2 regularization coefficient for computational stability.
        Note that if `regular_coef` is not 0 and the length of `sparsity` is not 1, algorithm will compute full hessian matrix of objective function, which is time-consuming.
    + thread : int, optional, default=1
        Max number of multithreads. Only used for cross-validation.
        - If thread = 0, the maximum number of threads supported by
          the device will be used.

    Attributes
    ----------
    params : array-like, shape(p, )
        The sparse optimal solution
    aux_params : array-like, shape(`aux_params_size`,)
        The aux_params of the model.
    eval_objective : float
        If cv=1, it stores the score under chosen information criterion.
        If cv>1, it stores the test objective under cross-validation.
    train_objective : float
        The objective on training data.
    value_objective: float
        The value of objective function on the solution.
    Examples
    --------


    References
    ----------
    - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang.
      A polynomial algorithm for best-subset selection problem.
      Proceedings of the National Academy of Sciences,
      117(52):33117-33123, 2020.
    """

    # attributes
    aux_params = None
    eval_objective = None
    train_objective = None

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        aux_params_size=0,
        sample_size=1,
        always_select=None,
        group=None,
        warm_start=True,
        important_search=128,
        screening_size=-1,
        max_iter=20,
        max_exchange_num=5,
        splicing_type="halve",
        path_type="seq",
        gs_lower_bound=0,
        gs_higher_bound=None,
        ic_type="gic",
        ic_coef=1.0,
        cv=1,
        cv_fold_id=None,
        regular_coef=0.0,
        thread=1,
    ):
        self.model = UniversalModel()
        self.dimensionality = dimensionality
        self.aux_params_size = aux_params_size
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.max_exchange_num = max_exchange_num
        self.splicing_type = splicing_type
        self.path_type = path_type
        self.sparsity = sparsity
        self.gs_lower_bound = gs_lower_bound
        self.gs_higher_bound = gs_higher_bound
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.cv_fold_id = cv_fold_id
        self.regular_coef = regular_coef
        self.always_select = always_select
        self.screening_size = screening_size
        self.important_search = important_search
        self.group = group
        self.warm_start = warm_start
        self.thread = thread

    def solve(self, 
        objective, 
        init_support_set=None,
        init_params=None,
        init_aux_params=None,
        gradient=None,
        hessian=None,
        autodiff=False,
        data=None,
        console_log_level="off", file_log_level="off", log_file_name="logs/scope.log"):
        r"""
        Set the optimization objective and begin to solve

        Parameters
        ----------
        + objective : function('params': array, ('aux_params': array), ('data': custom class)) ->  float 
            Defined the objective of optimization, must be written in JAX if gradient and hessian are not provided.
            If `autodiff` is `True`, `objective` can be a wrap of Cpp overloaded function which defined the objective of optimization with Cpp library `autodiff`, examples can be found in https://github.com/abess-team/scope_example.
        + init_support_set : array-like of int, optional, default=[]
            The index of the variables in initial active set.
        + init_params : array-like of float, optional, default is an all-zero vector
            An initial value of parameters.
        + init_aux_params : array-like of float, optional, default is an all-zero vector
            An initial value of auxiliary parameters.
        + gradient : function('params': array, 'aux_params': array, 'data': custom class, 'compute_index': array) -> array
            Defined the gradient of objective function, return the gradient of `aux_params` and the parameters in `compute_index`.
        + hessian : function('params': array, 'aux_params': array, 'data': custom class, 'compute_index': array) -> 2D array
            Defined the hessian of objective function, return the hessian matrix of the parameters in `compute_index`.
        + autodiff : bool, optional, default=False
            If `autodiff` is `True`, `objective` must be a wrap of Cpp overloaded function which defined the objective of optimization with Cpp library `autodiff`, examples can be found in https://github.com/abess-team/scope_example.
        + data : custom class, optional, default=None
            Any class which is match to objective function. It can cantain all data that objective should be known, like samples, responses, weights, etc.
        + console_log_level : str, optional, default="off"
            The level of output log to console, which can be "off", "error", "warning", "debug". For example, if it's "warning", only error and warning log will be output to console.
        + file_log_level : str, optional, default="off"
            The level of output log to file, which can be "off", "error", "warning", "debug". For example, if 
            it's "off", no log will be output to file.
        + log_file_name : str, optional, default="logs/scope.log"
            The name (relative path) of log file, which is used to store the log information.
        """
        # log level
        if console_log_level == "off":
            console_log_level = 6
        elif console_log_level == "error":
            console_log_level = 4
        elif console_log_level == "warning":
            console_log_level = 3
        elif console_log_level == "debug":
            console_log_level = 1
        elif isinstance(console_log_level, int) and console_log_level >= 0 and console_log_level <= 6:
            pass
        else:
            raise ValueError("console_log_level must be in 'off', 'error', 'warning', 'debug'")

        if file_log_level == "off":
            file_log_level = 6
        elif file_log_level == "error":
            file_log_level = 4
        elif file_log_level == "warning":
            file_log_level = 3
        elif file_log_level == "debug":
            file_log_level = 1
        elif isinstance(file_log_level, int) and file_log_level >= 0 and file_log_level <= 6:
            pass
        else:
            raise ValueError("file_log_level must be in 'off', 'error', 'warning', 'debug'")

        if not isinstance(log_file_name, str):
            raise ValueError("log_file_name must be a string")

        init_spdlog(console_log_level, file_log_level, log_file_name)

        # dimensionality
        p = self.dimensionality
        self.__check_positive_integer(p, "dimensionality")

        # sample_size
        n = self.sample_size
        self.__check_positive_integer(n, "sample_size")

        # aux_params_size
        m = self.aux_params_size
        self.__check_non_negative_integer(m, "aux_params_size")

        # max_iter
        self.__check_non_negative_integer(self.max_iter, "max_iter")

        # max_exchange_num
        self.__check_positive_integer(self.max_exchange_num, "max_exchange_num")

        # path_type
        if self.path_type == "seq":
            path_type = 1
        elif self.path_type == "gs":
            path_type = 2
        else:
            raise ValueError("path_type should be 'seq' or 'gs'")

        # ic_type
        if self.ic_type == "aic":
            ic_type = 1
        elif self.ic_type == "bic":
            ic_type = 2
        elif self.ic_type == "gic":
            ic_type = 3
        elif self.ic_type == "ebic":
            ic_type = 4
        else:
            raise ValueError('ic_type should be "aic", "bic", "ebic" or "gic"')

        # cv
        self.__check_positive_integer(self.cv, "cv")
        if self.cv > n:
            raise ValueError("cv should not be greater than sample_size")

        # group
        if self.group is None:
            group = np.array(range(p), dtype="int32")
            group_num = p  # len(np.unique(group))
        else:
            group = np.array(self.group)
            if group.ndim > 1:
                raise ValueError("Group should be an 1D array of integers.")
            if group.size != p:
                raise ValueError("The length of group should be equal to dimensionality.")
            group_num = len(np.unique(group))
            if group[0] != 0:
                raise ValueError("Group should start from 0.")
            if any(group[1:] - group[:-1] < 0):
                raise ValueError("Group should be an incremental integer array.")
            if not group_num == max(group) + 1:
                raise ValueError("There is a gap in group.")
            group = np.array(
                [np.where(group == i)[0][0] for i in range(group_num)], dtype="int32"
            )

        # sparsity
        if self.path_type == "gs":
            sparsity = np.array([0], dtype="int32")
        else:
            if self.sparsity == None:
                if n == 1 or group_num == 1:
                    sparsity = np.array([0, 1], dtype="int32")
                else:
                    sparsity = np.array(
                        range(
                            max(
                                1,
                                min(
                                    group_num,
                                    int(n / np.log(np.log(n)) / np.log(group_num)),
                                ),
                            )
                        ),
                        dtype="int32",
                    )
            else:
                if isinstance(self.sparsity, (int, float)):
                    sparsity = np.array([self.sparsity], dtype="int32")
                else:
                    sparsity = np.array(self.sparsity, dtype="int32")
                sparsity = np.sort(np.unique(sparsity))
                if sparsity[0] < 0 or sparsity[-1] > group_num:
                    raise ValueError(
                        "All sparsity should be between 0 and dimensionality"
                    )

        # regular_coef
        if self.regular_coef == None:
            regular_coef = np.array([0.0], dtype=float)
        else:
            if isinstance(self.regular_coef, (int, float)):
                regular_coef = np.array([self.regular_coef], dtype=float)
            else:
                regular_coef = np.array(self.regular_coef, dtype=float)
            if any(regular_coef < 0.0):
                raise ValueError("regular_coef should be positive.")

        # gs_bound
        if self.path_type == "seq":
            gs_lower_bound = gs_higher_bound = 0
        else:
            if self.gs_lower_bound is None:
                gs_lower_bound = 0
            else:
                gs_lower_bound = self.gs_lower_bound
            if self.gs_higher_bound is None:
                gs_higher_bound = min(
                    group_num, int(n / (np.log(np.log(n)) * np.log(group_num)))
                )
            else:
                gs_higher_bound = self.gs_higher_bound
            if gs_lower_bound > gs_higher_bound:
                raise ValueError(
                    "gs_higher_bound should be larger than gs_lower_bound."
                )

        # screening_size
        if self.screening_size == -1:
            screening_size = -1
        elif self.screening_size == 0:
            screening_size = min(
                group_num,
                max(
                    max(sparsity[-1], gs_higher_bound),
                    int(n / (np.log(np.log(n)) * np.log(group_num))),
                ),
            )
        else:
            screening_size = self.screening_size
            if screening_size > group_num or screening_size < max(
                sparsity[-1], gs_higher_bound
            ):
                raise ValueError(
                    "screening_size should be between max(sparsity) and dimensionality."
                )

        # always_select
        if self.always_select is None:
            always_select = np.array([], dtype="int32")
        else:
            always_select = np.sort(np.array(self.always_select, dtype="int32"))
            if len(always_select) > 0 and (
                always_select[0] < 0 or always_select[-1] >= group_num
            ):
                raise ValueError("always_select should be between 0 and dimensionality.")

        # thread
        self.__check_non_negative_integer(self.thread, "thread")

        # splicing_type
        if self.splicing_type == "halve":
            splicing_type = 0
        elif self.splicing_type == "taper":
            splicing_type = 1
        else:
            raise ValueError('splicing_type should be "halve" or "taper".')

        # important_search
        self.__check_non_negative_integer(self.important_search, "important_search")

        # cv_fold_id
        if self.cv_fold_id is None:
            cv_fold_id = np.array([], dtype="int32")
        else:
            cv_fold_id = np.array(cv_fold_id, dtype="int32")
            if cv_fold_id.ndim > 1:
                raise ValueError("group should be an 1D array of integers.")
            if cv_fold_id.size != n:
                raise ValueError("The length of group should be equal to X.shape[0].")
            if len(set(cv_fold_id)) != self.cv:
                raise ValueError(
                    "The number of different masks should be equal to `cv`."
                )

        # init_support_set
        if init_support_set is None:
            init_support_set = np.array([], dtype="int32")
        else:
            init_support_set = np.array(init_support_set, dtype="int32")
            if init_support_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be " "an 1D array of integers."
                )
            if init_support_set.min() < 0 or init_support_set.max() >= p:
                raise ValueError("init_support_set contains wrong index.")

        # init_params
        if init_params is None:
            init_params = np.zeros(p, dtype=float)
        else:
            init_params = np.array(init_params, dtype=float)
            if init_params.shape != (p,):
                raise ValueError("The length of init_params must match `dimensionality`!")

        # init_aux_params
        if init_aux_params is None:
            init_aux_params = np.zeros(m, dtype=float)
        else:
            init_aux_params = np.array(init_aux_params, dtype=float)
            if init_aux_params.shape != (m,):
                raise ValueError("The length of init_aux_params must match `aux_params_size`!")

        # set optimization objective
        if autodiff:
            self.__set_objective_autodiff(objective)
        elif gradient is not None and hessian is not None:
            self.__set_objective_custom(objective, gradient, hessian)
        else:
            objective = ScopeSolver.__objective_decorator(objective, self.aux_params_size)
            self.__set_objective_jax(objective)    

        result = pywrap_Universal(
            data,
            self.model,
            p,
            n,
            m,
            self.max_iter,
            self.max_exchange_num,
            path_type,
            self.warm_start,
            ic_type,
            self.ic_coef,
            self.cv,
            sparsity,
            regular_coef,
            gs_lower_bound,
            gs_higher_bound,
            screening_size,
            group,
            always_select,
            self.thread,
            splicing_type,
            self.important_search,
            cv_fold_id,
            init_support_set,
            init_params,
            init_aux_params,
        )

        self.params = result[0]
        self.aux_params = result[1].squeeze()
        self.train_objective = result[2]
        self.eval_objective = result[4] if self.cv==1 else result[3]
        self.value_objective = objective(self.params, self.aux_params, data)

    def set_split_method(self, spliter, deleter=None):
        r"""
        Register `spliter` as a callback function to split data into training set and validation set for cross-validation.

        Parameters
        ----------
        + spliter : function {'data': custom class, 'index': array-like, 'return': custom class}
            Filter samples in the `data` within `index`.
        + deleter : function {'data': custom class}
            If the custom class `data` is defined in Cpp, it is necessary to register its destructor as callback function `deleter` to avoid memory leak. This isn't necessary for Python class because Python has its own garbage collection mechanism.

        Examples
        --------
            class CustomData:
                def __init__(self, X, y):
                    self.X = X
                    self.y = y
            
            solver = ScopeSolver(dimensionality=5, cv=10)
            solver.set_split_method(lambda data, index:  Data(data.x[index, :], data.y[index]))
        """
        self.model.set_slice_by_sample(spliter)
        if deleter is not None:
            self.model.set_deleter(deleter)

    def set_init_params_of_sub_optim(self, func):
        r"""
        Register a callback function to initialize parameters and auxiliary parameters for each sub-problem of optimization.

        Parameters
        ----------
        + func : function {'params': array-like, 'aux_params': array-like, 'data': custom class, 'active_index': array-like, 'return': tuple of array-like}
            - `params` and `aux_params` are the default initialization of parameters and auxiliary parameters.
            - `data` is the training set of sub-problem.
            - `active_index` is the index of parameters needed initialization, the parameters not in `active_index` must be zeros. 
            - The function should return a tuple of array-like, the first element is the initialization of parameters and the second element is the initialization of auxiliary parameters.
        """
        self.model.set_init_params(func)

    def __objective_decorator(objective, aux_params_size):
        if objective.__code__.co_argcount == 3:
            return objective
        elif objective.__code__.co_argcount == 2 and aux_params_size > 0:
            def __objective(params, aux_params, data):
                    return objective(params, aux_params)
            return __objective
        elif objective.__code__.co_argcount == 2 and aux_params_size == 0:
            def __objective(params, aux_params, data):
                    return objective(params, data)
            return __objective
        elif objective.__code__.co_argcount == 1:
            def __objective(params, aux_params, data):
                return objective(params)
            return __objective
        else:
            raise ValueError("The objective function should have 1, 2 or 3 arguments.")

    def __set_objective_autodiff(self, objective_overloaded):
        r"""
        Register objective function as callback function. This method only can register objective function with Cpp library `autodiff`.

        Parameters
        ----------
        + objective_overloaded : a wrap of Cpp overloaded function which defined the objective of optimization, examples can be found in https://github.com/abess-team/scope_example.
        """
        self.model.set_loss_of_model(objective_overloaded)
        self.model.set_gradient_autodiff(objective_overloaded)
        self.model.set_hessian_autodiff(objective_overloaded)        

    def __set_objective_jax(self, objective):
        r"""
        Register objective function as callback function. This method only can register objective function with Python package `JAX`.

        Parameters
        ----------
        + objective : function('params': jax.numpy.DeviceArray, 'aux_params': jax.numpy.DeviceArray, 'data': custom class) ->  float or function('params': jax.numpy.DeviceArray, 'data': custom class) -> float
            Defined the objective of optimization, must be written in JAX.

        Examples
        --------
            import jax.numpy as jnp
            from abess import ScopeSolver
            
            class CustomData:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            def linear_no_intercept(params, data):
                return jnp.sum(jnp.square(data.x @ params - data.y))

            def linear_with_intercept(params, aux_params, data):
                return jnp.sum(jnp.square(data.x @ params + aux_params - data.y))
            
            solver1 = ScopeSolver(10)
            solver1.set_objective_jax(linear_no_intercept)

            solver2 = ScopeSolver(10, aux_params_size=1)
            solver2.set_objective_jax(linear_with_intercept)
        """
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")

        # the function for differential
        def func_(params_compute, aux_params, params, ind, data):
            params_complete = params.at[ind].set(params_compute)
            return objective(params_complete, aux_params, data)

        def grad_(params, aux_params, data, compute_params_index):
            params_j = jnp.array(params)
            aux_params_j = jnp.array(aux_params)
            params_compute_j = jnp.array(params[compute_params_index])
            return np.array(
                jnp.append(
                    *jax.grad(func_, (1, 0))(
                        params_compute_j, aux_params_j, params_j, compute_params_index, data
                    )
                )
            )

        def hessian_(params, aux_params, data, compute_params_index):
            params_j = jnp.array(params)
            aux_params_j = jnp.array(aux_params)
            params_compute_j = jnp.array(params[compute_params_index])
            return np.array(
                jax.jacfwd(jax.jacrev(func_))(
                    params_compute_j, aux_params_j, params_j, compute_params_index, data
                )
            )

        self.model.set_loss_of_model(objective)
        self.model.set_gradient_user_defined(grad_)
        self.model.set_hessian_user_defined(hessian_)
    
    def __set_objective_custom(self, objective, gradient, hessian):
        r"""
        Register objective function and its gradient and hessian as callback function.

        Parameters
        ----------
        + objective : function {'params': array-like, 'aux_params': array-like, 'data': custom class, 'return': float}
            Defined the objective of optimization.
        + gradient : function {'params': array-like, 'aux_params': array-like, 'data': custom class, 'compute_index': array-like, 'return': array-like}
            Defined the gradient of objective function, return the gradient of `aux_params` and the parameters in `compute_index`.
        + hessian : function {'params': array-like, 'aux_params': array-like, 'data': custom class, 'compute_index': array-like, 'return': 2D array-like}
            Defined the hessian of objective function, return the hessian matrix of the parameters in `compute_index`.

        Examples
        --------
            import numpy as np
            def objective(params, aux_params, data):
                return np.sum(np.square(data.y - data.x @ params))
            def grad(params, aux_params, data, compute_params_index):
                return -2 * data.x[:,compute_params_index].T @ (data.y - data.x @ params)
            def hess(params, aux_params, data, compute_params_index):
                return 2 * data.x[:,compute_params_index].T @ data.x[:,compute_params_index]

            model.set_objective_custom(objective=objective, gradient=grad, hessian=hess)
        """
        self.model.set_loss_of_model(objective)
        # NOTE: Perfect Forwarding of grad and hess is neccessary for func written in Pybind11_Cpp code 
        self.model.set_gradient_user_defined(
            lambda arg1, arg2, arg3, arg4: gradient(arg1, arg2, arg3, arg4)
        )

        self.model.set_hessian_user_defined(
            lambda arg1, arg2, arg3, arg4: hessian(arg1, arg2, arg3, arg4)
        )

    @staticmethod
    def __check_positive_integer(var, name: str):
        if (not isinstance(var, int) or var <= 0):
            raise ValueError("{} should be an positive integer.".format(name))

    @staticmethod
    def __check_non_negative_integer(var, name: str):
        if (not isinstance(var, int) or var < 0):
            raise ValueError("{} should be an non-negative integer.".format(name))    

class GrahtpSolver(BaseSolver):
    def __init__(self,
        dimensionality,
        sparsity,
        fast=False,
        final_support_size=-1,
        init_params=None,
        step_size=0.005,
        max_iter=100,
    ):
        self.dimensionality=dimensionality
        self.sparsity=sparsity
        self.fast=fast
        self.final_support_size=final_support_size
        self.init_params=init_params
        self.step_size=step_size
        self.max_iter=max_iter

    def solve(self, 
        objective, 
        gradient=None,
    ):
        nlopt = importlib.import_module("nlopt")
            
        if gradient is None:
            jax = importlib.import_module("jax")
            jnp = importlib.import_module("jax.numpy")

            loss_fn_jax = objective
            objective = lambda x: loss_fn_jax(x).item()

            def func_(para_compute, para, index):
                para_complete = para.at[index].set(para_compute)
                return loss_fn_jax(para_complete)

            def gradient(para, compute_para_index):
                para_j = jnp.array(para)
                para_compute_j = jnp.array(para[compute_para_index])
                return np.array(
                    jax.jacfwd(
                        func_
                    )(  ## forward mode automatic differentiation is faster than reverse mode
                        para_compute_j, para_j, compute_para_index
                    )
                )

        if self.init_params is None:
            self.init_params = np.zeros(self.dimensionality)

        if final_support_size < 0:
            final_support_size = self.sparsity

        # init
        x_old = self.init_params
        support_old = np.argpartition(np.abs(x_old), -self.sparsity)[
            -self.sparsity:
        ]  # the index of self.sparsity largest entries

        for iter in range(self.max_iter):
            # S1: gradient descent
            x_bias = x_old - self.step_size * gradient(x_old, np.arange(self.dimensionality))
            # S2: Gradient Hard Thresholding
            support_new = np.argpartition(np.abs(x_bias), -self.sparsity)[-self.sparsity:]
            # S3: debise
            if self.fast:
                x_new = np.zeros(self.dimensionality)
                x_new[support_new] = x_bias[support_new]
            else:
                def opt_f(x, gradient):
                    x_full = np.zeros(self.dimensionality)
                    x_full[support_new] = x
                    if gradient.size > 0:
                        gradient[:] = gradient(x_full, support_new)
                    return objective(x_full)

                opt = nlopt.opt(nlopt.LD_SLSQP, self.sparsity)
                opt.set_min_objective(opt_f)
                opt.set_ftol_rel(0.001)
                x_new = np.zeros(self.dimensionality)
                try:
                    x_new[support_new] = opt.optimize(x_bias[support_new])
                except RuntimeError:
                    x_new[support_new] = opt.last_optimize_result()
            # terminating condition
            if np.all(set(support_old) == set(support_new)):
                break
            x_old = x_new
            support_old = support_new

        final_support = np.argpartition(np.abs(x_new), -final_support_size)[
            -final_support_size:
        ]
        final_estimator = np.zeros(self.dimensionality)
        final_estimator[final_support] = x_new[final_support]

        self.params = final_estimator
        self.value_objective = objective(final_estimator)        

class GraspSolver(BaseSolver):
    def __init__(self,
        dimensionality,
        sparsity,
        max_iter=100,
    ):
        self.dimensionality=dimensionality
        self.sparsity=sparsity
        self.max_iter=max_iter
    
    def solve(self, 
        objective=None, 
        gradient=None,
    ):
        nlopt = importlib.import_module("nlopt")

        if gradient is None:
            jax = importlib.import_module("jax")
            jnp = importlib.import_module("jax.numpy")

            loss_fn_jax = objective
            objective = lambda x: loss_fn_jax(x).item()

            def func_(para_compute, para, index):
                para_complete = para.at[index].set(para_compute)
                return loss_fn_jax(para_complete)

            def gradient(para, compute_para_index):
                para_j = jnp.array(para)
                para_compute_j = jnp.array(para[compute_para_index])
                return np.array(
                    jax.jacfwd(func_)( ## forward mode automatic differentiation is faster than reverse mode
                        para_compute_j, para_j, compute_para_index
                    )
                )
        # init
        x_old = np.zeros(self.dimensionality)
    
        for iter in range(self.max_iter):
            # compute local gradient 
            z = gradient(x_old, np.arange(self.dimensionality))

            # identify directions
            if 2*self.sparsity < self.dimensionality:
                Omega = [idx for idx in np.argpartition(np.abs(z), -2*self.sparsity)[-2*self.sparsity:] if z[idx] != 0.0] # supp of top 2k largest absolute values of gradient
            else:
                Omega = np.nonzero(z)[0] # supp(z)

            # merge supports
            support_new = np.unique(np.append(Omega, x_old.nonzero()[0])) 
            
            # minimize 
            def opt_f(x, gradient):
                x_full = np.zeros(self.dimensionality)
                x_full[support_new] = x
                if gradient.size > 0:
                    gradient[:] = gradient(x_full, support_new)
                return objective(x_full)    

            opt = nlopt.opt(nlopt.LD_SLSQP, support_new.size)
            opt.set_min_objective(opt_f)
            opt.set_ftol_rel(0.001)
            x_tem = np.zeros(self.dimensionality)
            try:
                x_tem[support_new] = opt.optimize(x_old[support_new])
            except RuntimeError:
                x_tem[support_new] = opt.last_optimize_result()
            
            # prune estimate
            x_supp = np.argpartition(np.abs(x_tem), -self.sparsity)[-self.sparsity:]
            x_new = np.zeros(self.dimensionality)
            x_new[x_supp] = x_tem[x_supp]

            # update
            x_old = x_new
            support_old = support_new

            # terminating condition
            if np.all(set(support_old) == set(support_new)):
                break
        
        self.params = x_new
        self.value_objective = objective(x_new)

class IHTSolver(GrahtpSolver):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        self.fast = True
