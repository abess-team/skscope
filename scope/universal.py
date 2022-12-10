from sklearn.base import BaseEstimator
import numpy as np
import importlib
from .pybind_cabess import pywrap_Universal, UniversalModel, init_spdlog
from .utilities import check_positive_integer, check_non_negative_integer


class ConvexSparseSolver(BaseEstimator):
    r"""
    Get sparse optimal solution of convex loss function by sparse-Constrained Optimization via Splicing Iteration (SCOPE) algorithm, which also can be used for variables selection.
    Specifically, ConvexSparseSolver aims to tackle this problem: min_{x} f(x) s.t. ||x||_0 <= s, where f(x) is a convex loss function and s is the sparsity level. Each element of x can be seen as a variable, and the nonzero elements of x are the selected variables.

    Parameters
    ----------
    + dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as p.
    + sample_size : int, optional, default=1
        sample size, only used in the selection of support size, denoted as n.
    + sparsity_level : int or array-like, optional, default=None
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as s. If sparsity_level is an array-like, it should be a list of integers. 
        default is `range(min(n, int(n/(log(log(n))log(p)))))`
        Used only when path_type = "seq".
    + aux_para_size : int, optional, default=0
        The total number of auxiliary variables, which means that they need to be considered in optimization but always be selected.
        This is for the convenience of some models, for example, the intercept in linear regression is an auxiliary variable.
    + cv : int, optional, default=1
        The folds number when use the cross-validation method.
        - If `cv`=1, cross-validation would not be used.
        - If `cv`>1, support size will be chosen by CV's test loss,
          instead of information criterion.
    + cv_fold_id: array-like with shape (n,), optional, default=None
        An array indicates different folds in CV.
        Samples in the same fold should be given the same number.
        The number of different masks should be equal to `cv`.
        Used only when `cv` > 1.
    + group : array-like with shape (p,), optional, default=range(p)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: [0,2,1,2] (not incremental), [1,2,3,3] (not start from 0), [0,2,2,3] (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example, "sparsity_level=[3]" means there will be 3 groups of variables selected rather than 3 variables,
        and "always_include=[0,3]" means the 0-th and 3-th groups must be selected.
    + data : custom class, optional, default=None
        Any class which is match to loss function. It can cantain all data that loss should be known, like samples, responses, weights, etc.
    + max_iter : int, optional, default=20
        Maximum number of iterations taken for the
        splicing algorithm to converge.
        The limitation of loss reduction can guarantee the convergence.
        The number of iterations is only to simplify the implementation.
    + max_exchange_num : int, optional, default=5
        Maximum exchange number when splicing.
    + splicing_type : {"halve", "taper"}, optional, default="halve"
        The type of reduce the exchange number in each iteration
        from max_exchange_num.
        "halve" for decreasing by half, "taper" for decresing by one.
    + path_type : {"seq", "gs"}, optional, default="seq"
        The method to be used to select the optimal support size.
        - For path_type = "seq", we solve the problem for all sizes in `sparsity_level` successively.
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
    + regular_coef : float, optional, default=0.0
        L2 regularization coefficient for computational stability.
        Note that if `regular_coef` is not 0 and the length of `sparsity_level` is not 1, algorithm will compute full hessian matrix of loss function, which is time-consuming.
    + always_select : array-like, optional, default=[]
        An array contains the indexes of variables which must be selected.
        Its effect is simillar to see these variables as auxiliary variables and set `aux_para_size`. 
    + screening_size : int, optional, default=-1
        The number of variables remaining after the screening before variables select. Screening is used to reduce the computational cost.
        `screening_size` should be a non-negative number smaller than p,
        but larger than any value in sparsity_level.
        - If screening_size=-1, screening will not be used.
        - If screening_size=0, screening_size will be set as
          `min(p, int(n / (log(log(n))log(p))))`.
    + important_search : int, optional, default=128
        The number of important variables which need be splicing. 
        This is used to reduce the computational cost. If it's too large, it would greatly increase runtime.
    + init_active_set : array-like, optional, default=[]
        The index of the variable in initial active set.
    + is_warm_start : bool, optional, default=True
        When tuning the optimal parameter combination, whether to use the last solution as a warm start to accelerate the iterative convergence of the splicing algorithm.
    + thread : int, optional, default=1
        Max number of multithreads. Only used for cross-validation.
        - If thread = 0, the maximum number of threads supported by
          the device will be used.

    Attributes
    ----------
    coef_ : array-like, shape(p, )
        The sparse optimal solution
    aux_para_ : array-like, shape(`aux_para_size`,)
        The aux_para of the model.
    ic_ : float
        If cv=1, it stores the score under chosen information criterion.
    test_loss_ : float
        If cv>1, it stores the test loss under cross-validation.
    train_loss_ : float
        The loss on training data.
    regularization_: float
        The best L2 regularization coefficient.

    Examples
    --------
        from abess import ConvexSparseSolver, make_glm_data
        import numpy as np
        import jax.numpy as jnp

        n = 30
        p = 5
        k = 3
        family = "gaussian"
        data = make_glm_data(family=family, n=n, p=p, k=k,
            coef_ = np.array([0, 1, 0, 1, -1]),

        model = ConvexSparseSolver(dimensionality=p, sparsity_level=k)

        def loss(para, aux_para, data):
            return jnp.sum(
                jnp.square(data.y - data.x @ para)
            )
        model.set_loss_jax(loss)

        model.fit(data)

        beta = model.get_solution()
        support_set = model.selected_variables()

    References
    ----------
    - Junxian Zhu, Canhong Wen, Jin Zhu, Heping Zhang, and Xueqin Wang.
      A polynomial algorithm for best-subset selection problem.
      Proceedings of the National Academy of Sciences,
      117(52):33117-33123, 2020.
    """

    # attributes
    coef_ = None
    aux_para_ = None
    ic_ = 0
    train_loss_ = 0
    test_loss_ = 0

    def __init__(
        self,
        dimensionality,
        sparsity_level=None,
        aux_para_size=0,
        sample_size=1,
        data = None,
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
        always_select=None,
        screening_size=-1,
        important_search=128,
        group=None,
        init_active_set=None,
        is_warm_start=True,
        thread=1,
    ):
        self.model = UniversalModel()
        self.dimensionality = dimensionality
        self.aux_para_size = aux_para_size
        self.sample_size = sample_size
        self.data = data
        self.max_iter = max_iter
        self.max_exchange_num = max_exchange_num
        self.splicing_type = splicing_type
        self.path_type = path_type
        self.sparsity_level = sparsity_level
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
        self.init_active_set = init_active_set
        self.is_warm_start = is_warm_start
        self.thread = thread

    def solve(self, 
        loss=None, 
        gradient=None,
        hessian=None,
        loss_overload=None,
        data=None,
        console_log_level="off", file_log_level="off", log_file_name="logs/scope.log"):
        r"""
        Set the optimization objective and begin to solve

        Parameters
        ----------
        + loss : function('para': array, ('aux_para': array), ('data': custom class)) ->  float 
            Defined the objective of optimization, must be written in JAX if gradient and hessian are not provided.
        + gradient : function('para': array, 'aux_para': array, 'data': custom class, 'compute_index': array) -> array
            Defined the gradient of loss function, return the gradient of `aux_para` and the parameters in `compute_index`.
        + hessian : function('para': array, 'aux_para': array, 'data': custom class, 'compute_index': array) -> 2D array
            Defined the hessian of loss function, return the hessian matrix of the parameters in `compute_index`.
        + loss_overloaded : a wrap of Cpp overloaded function which defined the objective of optimization, examples can be found in https://github.com/abess-team/scope_example.
        + data : custom class, optional, default=None
            Any class which is match to loss function. It can cantain all data that loss should be known, like samples, responses, weights, etc.
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

        # data
        if data is None:
            data = self.data

        # dimensionality
        p = self.dimensionality
        check_positive_integer(p, "dimensionality")

        # sample_size
        n = self.sample_size
        check_positive_integer(n, "sample_size")

        # aux_para_size
        m = self.aux_para_size
        check_non_negative_integer(m, "aux_para_size")

        # max_iter
        check_non_negative_integer(self.max_iter, "max_iter")

        # max_exchange_num
        check_positive_integer(self.max_exchange_num, "max_exchange_num")

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
        check_positive_integer(self.cv, "cv")
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

        # sparsity_level
        if self.path_type == "gs":
            sparsity_level = np.array([0], dtype="int32")
        else:
            if self.sparsity_level == None:
                if n == 1 or group_num == 1:
                    sparsity_level = np.array([0, 1], dtype="int32")
                else:
                    sparsity_level = np.array(
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
                if isinstance(self.sparsity_level, (int, float)):
                    sparsity_level = np.array([self.sparsity_level], dtype="int32")
                else:
                    sparsity_level = np.array(self.sparsity_level, dtype="int32")
                sparsity_level = np.sort(np.unique(sparsity_level))
                if sparsity_level[0] < 0 or sparsity_level[-1] > group_num:
                    raise ValueError(
                        "All sparsity_level should be between 0 and dimensionality"
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
                    max(sparsity_level[-1], gs_higher_bound),
                    int(n / (np.log(np.log(n)) * np.log(group_num))),
                ),
            )
        else:
            screening_size = self.screening_size
            if screening_size > group_num or screening_size < max(
                sparsity_level[-1], gs_higher_bound
            ):
                raise ValueError(
                    "screening_size should be between max(sparsity_level) and dimensionality."
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
        check_non_negative_integer(self.thread, "thread")

        # splicing_type
        if self.splicing_type == "halve":
            splicing_type = 0
        elif self.splicing_type == "taper":
            splicing_type = 1
        else:
            raise ValueError('splicing_type should be "halve" or "taper".')

        # important_search
        check_non_negative_integer(self.important_search, "important_search")

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

        # init_active_set
        if self.init_active_set is None:
            init_active_set = np.array([], dtype="int32")
        else:
            init_active_set = np.array(self.init_active_set, dtype="int32")
            if init_active_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be " "an 1D array of integers."
                )
            if init_active_set.min() < 0 or init_active_set.max() >= p:
                raise ValueError("init_active_set contains wrong index.")

        # set optimization objective
        if loss_overload is not None:
            self.__set_loss_autodiff(loss_overload)
        elif loss is not None and gradient is not None and hessian is not None:
            self.__set_loss_custom(loss, gradient, hessian)
        elif loss is not None:
            self.__set_loss_jax(loss, self.aux_para_size)    
        else:
            raise ValueError("No loss function is provided!")

        result = pywrap_Universal(
            data,
            self.model,
            p,
            n,
            m,
            self.max_iter,
            self.max_exchange_num,
            path_type,
            self.is_warm_start,
            ic_type,
            self.ic_coef,
            self.cv,
            sparsity_level,
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
            init_active_set,
        )

        self.coef_ = result[0]
        self.aux_para_ = result[1].squeeze()
        self.train_loss_ = result[2]
        self.test_loss_ = result[3]
        self.ic_ = result[4]

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
            
            solver = ConvexSparseSolver(dimensionality=5, cv=10)
            solver.set_split_method(lambda data, index:  Data(data.x[index, :], data.y[index]))
        """
        self.model.set_slice_by_sample(spliter)
        if deleter is not None:
            self.model.set_deleter(deleter)

    def set_init_parameters_method(self, func):
        r"""
        Register a callback function to initialize parameters and auxiliary parameters for each sub-problem of optimization.

        Parameters
        ----------
        + func : function {'para': array-like, 'aux_para': array-like, 'data': custom class, 'active_index': array-like, 'return': tuple of array-like}
            - `para` and `aux_para` are the default initialization of parameters and auxiliary parameters.
            - `data` is the training set of sub-problem.
            - `active_index` is the index of parameters needed initialization, the parameters not in `active_index` must be zeros. 
            - The function should return a tuple of array-like, the first element is the initialization of parameters and the second element is the initialization of auxiliary parameters.
        """
        self.model.set_init_para(func)

    def get_parameters(self):
        r""" 
        Get the solution of optimization, include the parameters and auxiliary parameters (if exists).
        """
        if self.aux_para_size > 0:
            return self.coef_, self.aux_para_
        else:
            return self.coef_
    
    def get_selected_variables(self):
        r""" 
        Get the index of selected variables which is the non-zero parameters.
        """
        return np.nonzero(self.coef_)[0]

    def __loss_decorator(loss, aux_para_size):
        if loss.__code__.co_argcount == 3:
            return loss
        elif loss.__code__.co_argcount == 2 and aux_para_size > 0:
            def __loss(para, aux_para, data):
                    return loss(para, aux_para)
            return __loss
        elif loss.__code__.co_argcount == 2 and aux_para_size == 0:
            def __loss(para, aux_para, data):
                    return loss(para, data)
            return __loss
        elif loss.__code__.co_argcount == 1:
            def __loss(para, aux_para, data):
                return loss(para)
            return __loss
        else:
            raise ValueError("The loss function should have 1, 2 or 3 arguments.")

    def __set_loss_autodiff(self, loss_overloaded):
        r"""
        Register loss function as callback function. This method only can register loss function with Cpp library `autodiff`.

        Parameters
        ----------
        + loss_overloaded : a wrap of Cpp overloaded function which defined the objective of optimization, examples can be found in https://github.com/abess-team/scope_example.
        """
        self.model.set_loss_of_model(loss_overloaded)
        self.model.set_gradient_autodiff(loss_overloaded)
        self.model.set_hessian_autodiff(loss_overloaded)        

    def __set_loss_jax(self, loss, aux_para_size):
        r"""
        Register loss function as callback function. This method only can register loss function with Python package `JAX`.

        Parameters
        ----------
        + loss : function('para': jax.numpy.DeviceArray, 'aux_para': jax.numpy.DeviceArray, 'data': custom class) ->  float or function('para': jax.numpy.DeviceArray, 'data': custom class) -> float
            Defined the objective of optimization, must be written in JAX.

        Examples
        --------
            import jax.numpy as jnp
            from abess import ConvexSparseSolver
            
            class CustomData:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

            def linear_no_intercept(para, data):
                return jnp.sum(jnp.square(data.x @ para - data.y))

            def linear_with_intercept(para, aux_para, data):
                return jnp.sum(jnp.square(data.x @ para + aux_para - data.y))
            
            solver1 = ConvexSparseSolver(10)
            solver1.set_loss_jax(linear_no_intercept)

            solver2 = ConvexSparseSolver(10, aux_para_size=1)
            solver2.set_loss_jax(linear_with_intercept)
        """
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")

        loss = ConvexSparseSolver.__loss_decorator(loss, aux_para_size)

        # the function for differential
        def func_(para_compute, aux_para, para, ind, data):
            para_complete = para.at[ind].set(para_compute)
            return loss(para_complete, aux_para, data)

        def grad_(para, aux_para, data, compute_para_index):
            para_j = jnp.array(para)
            aux_para_j = jnp.array(aux_para)
            para_compute_j = jnp.array(para[compute_para_index])
            return np.array(
                jnp.append(
                    *jax.grad(func_, (1, 0))(
                        para_compute_j, aux_para_j, para_j, compute_para_index, data
                    )
                )
            )

        def hessian_(para, aux_para, data, compute_para_index):
            para_j = jnp.array(para)
            aux_para_j = jnp.array(aux_para)
            para_compute_j = jnp.array(para[compute_para_index])
            return np.array(
                jax.jacfwd(jax.jacrev(func_))(
                    para_compute_j, aux_para_j, para_j, compute_para_index, data
                )
            )

        self.model.set_loss_of_model(loss)
        self.model.set_gradient_user_defined(grad_)
        self.model.set_hessian_user_defined(hessian_)
    
    def __set_loss_custom(self, loss, gradient, hessian):
        r"""
        Register loss function and its gradient and hessian as callback function.

        Parameters
        ----------
        + loss : function {'para': array-like, 'aux_para': array-like, 'data': custom class, 'return': float}
            Defined the objective of optimization.
        + gradient : function {'para': array-like, 'aux_para': array-like, 'data': custom class, 'compute_index': array-like, 'return': array-like}
            Defined the gradient of loss function, return the gradient of `aux_para` and the parameters in `compute_index`.
        + hessian : function {'para': array-like, 'aux_para': array-like, 'data': custom class, 'compute_index': array-like, 'return': 2D array-like}
            Defined the hessian of loss function, return the hessian matrix of the parameters in `compute_index`.

        Examples
        --------
            import numpy as np
            def loss(para, aux_para, data):
                return np.sum(np.square(data.y - data.x @ para))
            def grad(para, aux_para, data, compute_para_index):
                return -2 * data.x[:,compute_para_index].T @ (data.y - data.x @ para)
            def hess(para, aux_para, data, compute_para_index):
                return 2 * data.x[:,compute_para_index].T @ data.x[:,compute_para_index]

            model.set_loss_custom(loss=loss, gradient=grad, hessian=hess)
        """
        self.model.set_loss_of_model(loss)
        # NOTE: Perfect Forwarding of grad and hess is neccessary for func written in Pybind11_Cpp code 
        self.model.set_gradient_user_defined(
            lambda arg1, arg2, arg3, arg4: gradient(arg1, arg2, arg3, arg4)
        )

        self.model.set_hessian_user_defined(
            lambda arg1, arg2, arg3, arg4: hessian(arg1, arg2, arg3, arg4)
        )

        