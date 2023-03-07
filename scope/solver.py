from sklearn.model_selection import KFold
from .base_solver import BaseSolver
from sklearn.base import BaseEstimator
import numpy as np
import nlopt
import jax
from jax import numpy as jnp
from ._scope import pywrap_Universal, UniversalModel, init_spdlog, NloptConfig


class ScopeSolver(BaseEstimator):
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
    + sample_size : int, optional, default=1
        sample size, only used in the selection of support size, denoted as n.
    + always_select : array-like, optional, default=[]
        An array contains the indexes of variables which must be selected.
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
        - For path_type = "gs", we solve the problem with support size ranged between `gs_lower_bound` and `gs_upper_bound`, where the specific support size to be considered is determined by golden section.
    + gs_lower_bound : int, optional, default=0
        The lower bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    + gs_upper_bound : int, optional, default=`min(n, int(n/(log(log(n))log(p))))`
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
    + console_log_level : str, optional, default="off"
        The level of output log to console, which can be "off", "error", "warning", "debug". For example, if it's "warning", only error and warning log will be output to console.
    + file_log_level : str, optional, default="off"
        The level of output log to file, which can be "off", "error", "warning", "debug". For example, if
        it's "off", no log will be output to file.
    + log_file_name : str, optional, default="logs/scope.log"
        The name (relative path) of log file, which is used to store the log information.

    Attributes
    ----------
    params : array-like, shape(p, )
        The sparse optimal solution
    eval_objective : float
        If cv=1, it stores the score under chosen information criterion.
        If cv>1, it stores the test objective under cross-validation.
    train_objective : float
        The objective on training data.
    value_of_objective: float
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

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
        nlopt_solver=nlopt.opt(nlopt.LD_LBFGS, 1),
        max_iter=20,
        ic_type="aic",
        ic_coef=1.0,
        cv=1,
        split_method=None,
        deleter=None,
        cv_fold_id=None,
        group=None,
        init_params_of_sub_optim=None,
        warm_start=True,
        important_search=128,
        screening_size=-1,
        max_exchange_num=5,
        splicing_type="halve",
        path_type="seq",
        gs_lower_bound=None,
        gs_upper_bound=None,
        thread=1,
        jax_platform="cpu",
        random_state=None,
        console_log_level="off",
        file_log_level="off",
        log_file_name="logs/scope.log",
    ):
        self.model = UniversalModel()
        self.dimensionality = dimensionality
        self.sparsity = sparsity
        self.sample_size = sample_size

        self.always_select = always_select
        self.nlopt_solver = nlopt_solver
        self.max_iter = max_iter
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.cv = cv
        self.split_method = split_method
        self.deleter = deleter
        self.cv_fold_id = cv_fold_id
        self.group = group
        self.init_params_of_sub_optim = init_params_of_sub_optim
        self.warm_start = warm_start
        self.important_search = important_search
        self.screening_size = screening_size
        self.max_exchange_num = max_exchange_num
        self.splicing_type = splicing_type
        self.path_type = path_type
        self.gs_lower_bound = gs_lower_bound
        self.gs_upper_bound = gs_upper_bound
        self.thread = thread
        self.jax_platform = jax_platform
        self.random_state = random_state
        self.console_log_level = console_log_level
        self.file_log_level = file_log_level
        self.log_file_name = log_file_name

    def get_config(self, deep=True):
        return super().get_params(deep)

    def set_config(self, **params):
        return super().set_params(**params)

    @staticmethod
    def _set_log_level(console_log_level, file_log_level, log_file_name):
        # log level
        log_level_dict = {
            "off": 6,
            "error": 4,
            "warning": 3,
            "debug": 1,
        }
        console_log_level = console_log_level.lower()
        file_log_level = file_log_level.lower()
        if console_log_level not in log_level_dict or file_log_level not in log_level_dict:
            raise ValueError(
                "console_log_level and file_log_level must be in 'off', 'error', 'warning', 'debug'"
            )
        console_log_level = log_level_dict[console_log_level]
        file_log_level = log_level_dict[file_log_level]
        # log file name
        if not isinstance(log_file_name, str):
            raise ValueError("log_file_name must be a string")

        init_spdlog(console_log_level, file_log_level, log_file_name)

    def solve(
        self,
        objective,
        init_support_set=None,
        init_params=None,
        gradient=None,
        hessian=None,
        cpp=False,
        data=None,
        jit=False,
    ):
        r"""
        Set the optimization objective and begin to solve

        Parameters
        ----------
        + objective : function('params': array, ('data': custom class)) ->  float
            Defined the objective of optimization, must be written in JAX if gradient and hessian are not provided.
            If `cpp` is `True`, `objective` can be a wrap of Cpp overloaded function which defined the objective of optimization with Cpp library `autodiff`, examples can be found in https://github.com/abess-team/scope_example.
        + init_support_set : array-like of int, optional, default=[]
            The index of the variables in initial active set.
        + init_params : array-like of float, optional, default is an all-zero vector
            An initial value of parameters.
        + gradient : function('params': array, 'data': custom class, 'compute_index': array) -> array
            Defined the gradient of objective function, return the gradient of parameters in `compute_index`.
        + hessian : function('params': array, 'data': custom class, 'compute_index': array) -> 2D array
            Defined the hessian of objective function, return the hessian matrix of the parameters in `compute_index`.
        + cpp : bool, optional, default=False
            If `cpp` is `True`, `objective` must be a wrap of Cpp overloaded function which defined the objective of optimization with Cpp library `autodiff`, examples can be found in https://github.com/abess-team/scope_example.
        + data : custom class, optional, default=None
            Any class which is match to objective function. It can cantain all data that objective should be known, like samples, responses, weights, etc.
        """
        ScopeSolver._set_log_level(
            self.console_log_level, self.file_log_level, self.log_file_name
        )

        if self.jax_platform not in ["cpu", "gpu", "tpu"]:
            raise ValueError("jax_platform must be in 'cpu', 'gpu', 'tpu'")
        jax.config.update("jax_platform_name", self.jax_platform)
        if jit and getattr(data, "__hash__") is None:
            raise ValueError("Non-hashable data is not supported by jit.")

        nlopt_config = NloptConfig(
            self.nlopt_solver.get_algorithm(),
            self.nlopt_solver.get_algorithm_name(),
            self.nlopt_solver.get_stopval(),
            self.nlopt_solver.get_ftol_rel(),
            self.nlopt_solver.get_ftol_abs(),
            self.nlopt_solver.get_xtol_rel(),
            self.nlopt_solver.get_maxtime(),
            self.nlopt_solver.get_population(),
            self.nlopt_solver.get_vector_storage(),
        )

        p = self.dimensionality
        BaseSolver._check_positive_integer(p, "dimensionality")

        n = self.sample_size
        BaseSolver._check_positive_integer(n, "sample_size")

        BaseSolver._check_non_negative_integer(self.max_iter, "max_iter")

        # max_exchange_num
        BaseSolver._check_positive_integer(self.max_exchange_num, "max_exchange_num")

        # ic_type
        information_criterion_dict = {
            "aic": 1,
            "bic": 2,
            "gic": 3,
            "ebic": 4,
        }
        if self.ic_type not in information_criterion_dict.keys():
            raise ValueError('ic_type should be "aic", "bic", "ebic" or "gic"')
        ic_type = information_criterion_dict[self.ic_type]

        # cv
        BaseSolver._check_positive_integer(self.cv, "cv")
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
                raise ValueError(
                    "The length of group should be equal to dimensionality."
                )
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

        # always_select
        always_select = np.unique(np.array(self.always_select, dtype="int32"))
        if always_select.size > 0 and (
            always_select[0] < 0 or always_select[-1] >= group_num
        ):
            raise ValueError("always_select should be between 0 and dimensionality.")

        # default sparsity level
        force_min_sparsity = always_select.size
        default_max_sparsity = max(
            force_min_sparsity,
            group_num
            if group_num <= 5
            else int(group_num / np.log(np.log(group_num)) / np.log(group_num)),
        )

        # path_type
        if self.path_type == "seq":
            path_type = 1
            gs_lower_bound, gs_upper_bound = 0, 0
            if self.sparsity is None:
                sparsity = np.arange(
                    force_min_sparsity,
                    default_max_sparsity + 1,
                    dtype="int32",
                )
            else:
                sparsity = np.unique(np.array(self.sparsity, dtype="int32"))
                if sparsity.size == 0:
                    raise ValueError("sparsity should not be empty.")
                if sparsity[0] < force_min_sparsity or sparsity[-1] > group_num:
                    raise ValueError(
                        "All sparsity should be between 0 (when `always_select` is default) and dimensionality (when `group` is default)."
                    )
        elif self.path_type == "gs":
            path_type = 2
            sparsity = np.array([0], dtype="int32")
            if self.gs_lower_bound is None:
                gs_lower_bound = force_min_sparsity
            else:
                BaseSolver._check_non_negative_integer(
                    self.gs_lower_bound, "gs_lower_bound"
                )
                gs_lower_bound = self.gs_lower_bound

            if self.gs_upper_bound is None:
                gs_upper_bound = default_max_sparsity
            else:
                BaseSolver._check_non_negative_integer(
                    self.gs_upper_bound, "gs_upper_bound"
                )
                gs_upper_bound = self.gs_upper_bound

            if gs_lower_bound < force_min_sparsity or gs_upper_bound > group_num:
                raise ValueError(
                    "gs_lower_bound and gs_upper_bound should be between 0 (when `always_select` is default) and dimensionality (when `group` is default)."
                )
            if gs_lower_bound > gs_upper_bound:
                raise ValueError(
                    "gs_upper_bound should be larger than gs_lower_bound."
                )
        else:
            raise ValueError("path_type should be 'seq' or 'gs'")

        # screening_size
        if self.screening_size == -1:
            screening_size = -1
        elif self.screening_size == 0:
            screening_size = max(sparsity[-1], gs_upper_bound, default_max_sparsity)
        else:
            screening_size = self.screening_size
            if screening_size > group_num or screening_size < max(
                sparsity[-1], gs_upper_bound
            ):
                raise ValueError(
                    "screening_size should be between max(sparsity) and dimensionality."
                )

        # thread
        BaseSolver._check_non_negative_integer(self.thread, "thread")

        # splicing_type
        if self.splicing_type == "halve":
            splicing_type = 0
        elif self.splicing_type == "taper":
            splicing_type = 1
        else:
            raise ValueError('splicing_type should be "halve" or "taper".')

        # important_search
        BaseSolver._check_non_negative_integer(
            self.important_search, "important_search"
        )

        # cv_fold_id
        if self.cv > 1:
            if self.cv_fold_id is None:
                kf = KFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state
                ).split(np.zeros(self.sample_size))

                self.cv_fold_id = np.zeros(self.sample_size)
                for i, (_, fold_id) in enumerate(kf):
                    self.cv_fold_id[fold_id] = i
            else:
                self.cv_fold_id = np.array(self.cv_fold_id, dtype="int32")
                if self.cv_fold_id.ndim > 1:
                    raise ValueError("group should be an 1D array of integers.")
                if self.cv_fold_id.size != n:
                    raise ValueError(
                        "The length of group should be equal to X.shape[0]."
                    )
                if len(set(self.cv_fold_id)) != self.cv:
                    raise ValueError(
                        "The number of different masks should be equal to `cv`."
                    )
        else:
            self.cv_fold_id = np.array([], dtype="int32")

        self.__set_split_method()
        self.__set_init_params_of_sub_optim()

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
                raise ValueError(
                    "The length of init_params must match `dimensionality`!"
                )

        # set optimization objective
        if cpp:
            loss_fn = self.__set_objective_cpp(objective, gradient, hessian)
        else:
            loss_fn = self.__set_objective_py(objective, gradient, hessian, jit, init_params, data)

        result = pywrap_Universal(
            data,
            self.model,
            nlopt_config,
            p,
            n,
            0,
            self.max_iter,
            self.max_exchange_num,
            path_type,
            self.warm_start,
            ic_type,
            self.ic_coef,
            self.cv,
            sparsity,
            np.array([0.0]),
            gs_lower_bound,
            gs_upper_bound,
            screening_size,
            group,
            always_select,
            self.thread,
            splicing_type,
            self.important_search,
            self.cv_fold_id,
            init_support_set,
            init_params,
            np.zeros(0),
        )

        self.params = np.array(result[0])
        self.support_set = np.nonzero(self.params)[0]
        self.train_objective = result[2]
        self.eval_objective = result[4] if self.cv == 1 else result[3]
        self.value_of_objective = loss_fn(self.params, data)

        return self.params

    def get_result(self):
        r"""
        Get the solution of optimization, include the parameters ...
        """
        return {
            "params": self.params,
            "support_set": self.support_set,
            "value_of_objective": self.value_of_objective,
            "train_objective": self.train_objective,
            "eval_objective": self.eval_objective,
        }

    def __set_split_method(self):
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
        self.model.set_slice_by_sample(self.split_method)
        self.model.set_deleter(self.deleter)

    def __set_init_params_of_sub_optim(self):
        r"""
        Register a callback function to initialize parameters for each sub-problem of optimization.

        Parameters
        ----------
        + func : function {'params': array-like, 'data': custom class, 'active_index': array-like, 'return': array-like}
            - `params` is the default initialization of parameters and auxiliary parameters.
            - `data` is the training set of sub-problem.
            - `active_index` is the index of parameters needed initialization, the parameters not in `active_index` must be zeros.
            - The function should return the initialization of parameters.
        """
        self.model.set_init_params_of_sub_optim(self.init_params_of_sub_optim)

    def __set_objective_cpp(self, objective, gradient, hessian):
        r"""
        Register objective function as callback function. This method only can register objective function with Cpp library `autodiff`.

        Parameters
        ----------
        + objective_overloaded : a wrap of Cpp overloaded function which defined the objective of optimization, examples can be found in https://github.com/abess-team/scope_example.
        

        Register objective function and its gradient and hessian as callback function.

        Parameters
        ----------
        + objective : function {'params': array-like, 'data': custom class, 'return': float}
            Defined the objective of optimization.
        + gradient : function {'params': array-like, 'data': custom class, 'return': array-like}
            Defined the gradient of objective function, return the gradient of parameters.
        + hessian : function {'params': array-like, 'data': custom class, 'return': 2D array-like}
            Defined the hessian of objective function, return the hessian matrix of the parameters.
        """
        self.model.set_loss_of_model(objective)
        if gradient is None:
            self.model.set_gradient_autodiff(objective)
        else:
            self.model.set_gradient_user_defined(
                lambda params, data: (objective(params, data), gradient(params, data))
            )
        if hessian is None:
            self.model.set_hessian_autodiff(objective)
        else:
            # NOTE: Perfect Forwarding of grad and hess is neccessary for func written in Pybind11_Cpp code
            self.model.set_hessian_user_defined(
                lambda params, data: hessian(params, data)
            )
        return objective

    def __set_objective_py(self, objective, gradient, hessian, jit, test_params, test_data):
        r"""
        Register objective function as callback function. This method only can register objective function with Python package `JAX`.

        Parameters
        ----------
        + objective : function('params': jax.numpy.DeviceArray, 'data': custom class) ->  float or function('params': jax.numpy.DeviceArray, 'data': custom class) -> float
            Defined the objective of optimization, must be written in JAX.


        Register objective function and its gradient and hessian as callback function.

        Parameters
        ----------
        + objective : function {'params': array-like, 'data': custom class, 'return': float}
            Defined the objective of optimization.
        + gradient : function {'params': array-like, 'data': custom class, 'return': array-like}
            Defined the gradient of objective function, return the gradient of parameters.
        + hessian : function {'params': array-like, 'data': custom class, 'return': 2D array-like}
            Defined the hessian of objective function, return the hessian matrix of the parameters.
        """        
        loss_, grad_ = BaseSolver._set_objective(objective, gradient, jit)

        # hess
        if hessian is None:
            hess_ = lambda params, data: jax.hessian(loss_)(params, data)
        elif hessian.__code__.co_argcount == 2:
            hess_ = hessian
        elif hessian.__code__.co_argcount == 1:
            hess_ = lambda params, data: hessian(params)
        else:
            raise ValueError("The hessian function should have 1 or 2 arguments.")
        if jit:
            hess_ = jax.jit(hess_, static_argnums=(1,))


        loss_fn = lambda params, data: loss_(params, data).item()
        def value_and_grad(params, data):
            value, grad = grad_(params, data)
            return value.item(), np.array(grad)
        def hess_fn(params, data):
            return np.array(hess_(jnp.array(params), data))

        self.model.set_loss_of_model(loss_fn)
        self.model.set_gradient_user_defined(value_and_grad)
        self.model.set_hessian_user_defined(hess_fn)

        return loss_fn


class GrahtpSolver(BaseSolver):
    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
        step_size=0.005,
        nlopt_solver=nlopt.opt(nlopt.LD_LBFGS, 1),
        max_iter=100,
        ic_type="aic",
        ic_coef=1.0,
        metric_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        jax_platform="cpu",
        random_state=None,
    ):
        self.fast = False  # fast version of GraHTP is actually IHT
        self.step_size = step_size
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            always_select=always_select,
            nlopt_solver=nlopt_solver,
            max_iter=max_iter,
            ic_type=ic_type,
            ic_coef=ic_coef,
            metric_method=metric_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
            jax_platform=jax_platform,
            random_state=random_state,
        )

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
    ):
        if sparsity <= self.always_select.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
            )
        # init
        params = init_params
        support_old = np.array([], dtype="int32")

        for iter in range(self.max_iter):
            # S1: gradient descent
            params_bias = params - self.step_size * value_and_grad(params, data)[1]
            # S2: Gradient Hard Thresholding
            score = np.abs(params_bias)
            score[self.always_select] = np.inf
            support_new = np.argpartition(score, -sparsity)[-sparsity:]
            # terminating condition
            if np.all(set(support_old) == set(support_new)):
                break
            else:
                support_old = support_new
            # S3: debise
            params = np.zeros(self.dimensionality)
            if self.fast:
                params[support_new] = params_bias[support_new]
            else:
                params[support_new], _ = self._cache_nlopt(
                    loss_fn, value_and_grad, params_bias, support_new, data
                )

        # final optimization for IHT
        if self.fast:
            params[support_new], _ = self._cache_nlopt(
                loss_fn, value_and_grad, params, support_new, data
            )

        return params, support_new


class GraspSolver(BaseSolver):

    ## inherited the constructor of BaseSolver

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
    ):
        if sparsity <= self.always_select.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
            )
        # init
        params = init_params
        support_old = np.array([], dtype="int32")

        for iter in range(self.max_iter):
            # compute local gradient
            grad_values = value_and_grad(params, data)[1]
            score = np.abs(grad_values)
            score[self.always_select] = np.inf
            # identify directions
            if 2 * sparsity < self.dimensionality:
                Omega = [
                    idx
                    for idx in np.argpartition(score, -2 * sparsity)[-2 * sparsity :]
                    if score[idx] != 0.0
                ]  # supp of top 2k largest absolute values of gradient
            else:
                Omega = np.nonzero(score)[0]  # supp(z)

            # merge supports
            support_new = np.unique(np.append(Omega, params.nonzero()[0]))

            # terminating condition
            if iter > 0 and np.all(set(support_old) == set(support_new)):
                break
            else:
                support_old = support_new

            # minimize
            params_bias = np.zeros(self.dimensionality)
            if support_new.size > 0:
                params_bias[support_new], _ = self._cache_nlopt(
                    loss_fn, value_and_grad, params, support_new, data
                )

            # prune estimate
            score = np.abs(params_bias)
            score[self.always_select] = np.inf
            support_set = np.argpartition(score, -sparsity)[-sparsity:]
            params = np.zeros(self.dimensionality)
            params[support_set] = params_bias[support_set]

        return params, support_set


class IHTSolver(GrahtpSolver):
    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
        step_size=0.005,
        nlopt_solver=nlopt.opt(nlopt.LD_LBFGS, 1),
        max_iter=100,
        ic_type="aic",
        ic_coef=1.0,
        metric_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        jax_platform="cpu",
        random_state=None,
    ):
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            always_select=always_select,
            step_size=step_size,
            nlopt_solver=nlopt_solver,
            max_iter=max_iter,
            ic_type=ic_type,
            ic_coef=ic_coef,
            metric_method=metric_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
            jax_platform=jax_platform,
            random_state=random_state,
        )
        self.fast = True  # IHT is actually fast version of GraHTP
