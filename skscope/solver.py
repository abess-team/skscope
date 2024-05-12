#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Zezhi Wang
# Copyright (C) 2023 abess-team
# Licensed under the MIT License.

from sklearn.model_selection import KFold
from .base_solver import BaseSolver
from sklearn.base import BaseEstimator
import numpy as np
import jax
from jax import numpy as jnp
from . import _scope, utilities
from .numeric_solver import convex_solver_LBFGS


class ScopeSolver(BaseEstimator):
    r"""
    Get sparse optimal solution of convex objective function by Sparse-Constrained Optimization via Splicing Iteration (SCOPE) algorithm, which also can be used for variables selection.
    Specifically, ``ScopeSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.

    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        default is ``range(int(p/log(log(p))/log(p)))``.
        Used only when ``path_type`` is "seq".
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``ScopeSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=20
        Maximum number of iterations taken for converging.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method. If ``cv`` = 1, the sparsity level will be chosen by the information criterion. If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method: callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id: array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    important_search : int, default=128
        The number of important variables which need be splicing.
        This is used to reduce the computational cost. If it's too large, it would greatly increase runtime.
    screening_size : int, default=-1
        The number of variables remaining after the screening before variables select. Screening is used to reduce the computational cost.
        ``screening_size`` should be a non-negative number smaller than p, but larger than any value in sparsity. If ``screening_size`` is -1, screening will not be used. If ``screening_size`` is 0, ``screening_size`` will be set as ``int(p/log(log(p))/log(p))``.
    max_exchange_num : int, optional, default=5
        Maximum exchange number when splicing.
    is_dynamic_max_exchange_num : bool, default=True
        If ``is_dynamic_max_exchange_num`` is True, ``max_exchange_num`` will be decreased dynamically according to the number of variables exchanged in the last iteration.
    greedy : bool, default=True,
        If ``greedy`` is True, the first exchange-number which can reduce the objective function value will be selected.
        Otherwise, the exchange-number which can reduce the objective function value most will be selected.
    splicing_type : {"halve", "taper"}, default="halve"
        The type of reduce the exchange number in each iteration from ``max_exchange_num``.
        "halve" for decreasing by half, "taper" for decresing by one.
    path_type : {"seq", "gs"}, default="seq"
        The method to be used to select the optimal sparsity level. For path_type = "seq", we solve the problem for all sizes in `sparsity` successively. For path_type = "gs", we solve the problem with sparsity level ranged between `gs_lower_bound` and `gs_upper_bound`, where the specific sparsity level to be considered is determined by golden section.
    gs_lower_bound : int, default=0
        The lower bound of golden-section-search for sparsity searching.
        Used only when path_type = "gs".
    gs_upper_bound : int, optional
        The higher bound of golden-section-search for sparsity searching.
        Default is ``int(p/log(log(p))/log(p))``.
        Used only when path_type = "gs".
    thread : int, default=1
        Max number of multithreads used for cross-validation. If thread = 0, the maximum number of threads supported by the device will be used.
    random_state : int, optional
        The random seed used for cross-validation.
    console_log_level : str, default="off"
        The level of output log to console, which can be "off", "error", "warning", "debug".
        For example, if ``console_log_level`` is "warning", only error and warning log will be output to console.
    file_log_level : str, default="off"
        The level of output log to file, which can be "off", "error", "warning", "debug".
        For example, if ``file_log_level`` is "off", no log will be output to file.
    log_file_name : str, default="logs/skscope.log"
        The name (relative path) of log file, which is used to store the log information.

    Attributes
    ----------
    params :array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value:float
        The value of objective function on the solution.
    support_set :array of int
        The indices of selected variables, sorted in ascending order.
    cv_test_loss :float
        If cv=1, it stores the score under chosen information criterion. If cv>1, it stores the test objective under cross-validation.
    cv_train_loss :float
        The objective on training data.

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
        preselect=[],
        numeric_solver=convex_solver_LBFGS,
        max_iter=20,
        ic_method=None,
        cv=1,
        split_method=None,
        cv_fold_id=None,
        group=None,
        important_search=128,
        screening_size=-1,
        max_exchange_num=5,
        is_dynamic_max_exchange_num=True,
        greedy=True,
        splicing_type="halve",
        path_type="seq",
        gs_lower_bound=None,
        gs_upper_bound=None,
        thread=1,
        random_state=None,
        console_log_level="off",
        file_log_level="off",
        log_file_name="logs/skscope.log",
    ):
        self.model = _scope.UniversalModel()
        self.dimensionality = dimensionality
        self.sparsity = sparsity
        self.sample_size = sample_size

        self.preselect = preselect
        self.numeric_solver = numeric_solver
        self.max_iter = max_iter
        self.ic_method = ic_method
        self.cv = cv
        self.split_method = split_method
        self.deleter = None
        self.cv_fold_id = cv_fold_id
        self.group = group
        self.warm_start = True
        self.important_search = important_search
        self.screening_size = screening_size
        self.max_exchange_num = max_exchange_num
        self.is_dynamic_max_exchange_num = is_dynamic_max_exchange_num
        self.greedy = greedy
        self.use_hessian = False
        self.splicing_type = splicing_type
        self.path_type = path_type
        self.gs_lower_bound = gs_lower_bound
        self.gs_upper_bound = gs_upper_bound
        self.thread = thread
        self.jax_platform = "cpu"
        self.random_state = random_state
        self.console_log_level = console_log_level
        self.file_log_level = file_log_level
        self.log_file_name = log_file_name
        self.hessian = None
        self.cpp = False

    def get_config(self, deep=True):
        """
        Get parameters for this solver.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this solver and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return super().get_params(deep)

    def set_config(self, **params):
        """Set the parameters of this solver.

        Parameters
        ----------
        **params : dict
            Solver parameters.

        Returns
        -------
        self :
            Solver instance.
        """
        return super().set_params(**params)

    def get_estimated_params(self):
        r"""
        Get the optimal parameters of the objective function.

        Returns
        -------
        parameters : array of shape (dimensionality,)
            The optimal solution of optimization.
        """
        return self.params

    def get_support(self):
        r"""
        Get the support set of the optimal parameters.

        Returns
        -------
        support_set : array of int
            The indices of selected variables, sorted in ascending order.
        """
        return self.support_set

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
        if (
            console_log_level not in log_level_dict
            or file_log_level not in log_level_dict
        ):
            raise ValueError(
                "console_log_level and file_log_level must be in 'off', 'error', 'warning', 'debug'"
            )
        console_log_level = log_level_dict[console_log_level]
        file_log_level = log_level_dict[file_log_level]
        # log file name
        if not isinstance(log_file_name, str):
            raise ValueError("log_file_name must be a string")

        _scope.init_spdlog(console_log_level, file_log_level, log_file_name)

    def solve(
        self,
        objective,
        data=None,
        layers=[],
        init_support_set=None,
        init_params=None,
        gradient=None,
        jit=False,
    ):
        r"""
        Optimize the optimization objective function.

        Parameters
        ----------
        objective : callable
            The objective function to be minimized: ``objective(params, data) -> loss``
            where ``params`` is a 1-D array with shape (dimensionality,) and
            ``data`` is the fixed parameters needed to completely specify the function.
            ``objective`` must be written in ``JAX`` library if ``gradient`` is not provided.
        data : optional
            Extra arguments passed to the objective function and its derivatives (if existed).
        layers : list of ``Layer`` objects, default=[]
            ``Layer`` is a "decorator" of the objective function.
            The parameters will be processed by the ``Layer`` before entering the objective function.
            The different layers can achieve different effects,
            and they can be sequentially concatenated together to form a larger layer,
            enabling the implementation of more complex functionalities.
            The ``Layer`` objects can be found in ``skscope.layers``.
            If ``layers`` is not empty, ``objective`` must be written in ``JAX`` library.
        init_support_set : array of int, default=[]
            The index of the variables in initial active set.
        init_params : array of shape (dimensionality,), optional
            The initial value of parameters, default is an all-zero vector.
        gradient : callable, optional
            A function that returns the gradient vector of parameters: ``gradient(params, data) -> array of shape (dimensionality,)``,
            where ``params`` is a 1-D array with shape (dimensionality,) and ``data`` is the fixed parameters needed to completely specify the function.
            If ``gradient`` is not provided, ``objective`` must be written in ``JAX`` library.
        jit : bool, default=False
            If ``objective`` or ``gradient`` are written in JAX, ``jit`` can be set to True to speed up the optimization.

        Returns
        -------
        parameters : array of shape (dimensionality,)
            The optimal solution of optimization.
        """
        hessian = self.hessian
        cpp = self.cpp
        ScopeSolver._set_log_level(
            self.console_log_level, self.file_log_level, self.log_file_name
        )

        jax.config.update("jax_platform_name", self.jax_platform)

        p = self.dimensionality
        BaseSolver._check_positive_integer(p, "dimensionality")

        n = self.sample_size
        BaseSolver._check_positive_integer(n, "sample_size")

        BaseSolver._check_non_negative_integer(self.max_iter, "max_iter")

        # max_exchange_num
        BaseSolver._check_positive_integer(self.max_exchange_num, "max_exchange_num")

        # group
        if self.group is None:
            group = np.arange(p, dtype="int32")
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

        # preselect
        preselect = np.unique(np.array(self.preselect, dtype="int32"))
        if preselect.size > 0 and (preselect[0] < 0 or preselect[-1] >= group_num):
            raise ValueError("preselect should be between 0 and dimensionality.")

        # default sparsity level
        force_min_sparsity = preselect.size
        default_max_sparsity = max(
            force_min_sparsity,
            (
                group_num
                if group_num <= 5
                else int(group_num / np.log(np.log(group_num)) / np.log(group_num))
            ),
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
                    raise ValueError("Sparsity should not be empty.")
                if sparsity[0] < force_min_sparsity or sparsity[-1] > group_num:
                    raise ValueError("There is an invalid sparsity.")
        elif self.path_type == "gs":
            if len(layers) > 0:
                raise ValueError(
                    "The path_type should be 'seq' when the layers are specified."
                )
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
                    "gs_lower_bound and gs_upper_bound should be between 0 and dimensionality."
                )
            if gs_lower_bound > gs_upper_bound:
                raise ValueError("gs_upper_bound should be larger than gs_lower_bound.")
        else:
            raise ValueError("path_type should be 'seq' or 'gs'")

        # screening_size
        if len(layers) > 0 and self.screening_size != -1:
            raise ValueError(
                "The screening_size should be -1 when the layers are specified."
            )
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
                    "screening_size should be between sparsity and dimensionality."
                )

        # thread
        BaseSolver._check_non_negative_integer(self.thread, "thread")

        # splicing_type
        if self.splicing_type == "halve":
            splicing_type = 0
        elif self.splicing_type == "taper":
            splicing_type = 1
        else:
            raise ValueError("splicing_type should be 'halve' or 'taper'.")

        # important_search
        BaseSolver._check_non_negative_integer(
            self.important_search, "important_search"
        )

        # cv
        BaseSolver._check_positive_integer(self.cv, "cv")
        if self.cv > n:
            raise ValueError("cv should not be greater than sample_size.")
        if self.cv > 1:
            ic_method = utilities.AIC if self.ic_method is None else self.ic_method
            if self.split_method is None:
                raise ValueError("split_method should be provided when cv > 1.")
            self.model.set_slice_by_sample(self.split_method)
            self.model.set_deleter(self.deleter)
            if self.cv_fold_id is None:
                kf = KFold(
                    n_splits=self.cv, shuffle=True, random_state=self.random_state
                ).split(np.zeros(n))

                self.cv_fold_id = np.zeros(n)
                for i, (_, fold_id) in enumerate(kf):
                    self.cv_fold_id[fold_id] = i
            else:
                self.cv_fold_id = np.array(self.cv_fold_id, dtype="int32")
                if self.cv_fold_id.ndim > 1:
                    raise ValueError("cv_fold_id should be an 1D array of integers.")
                if self.cv_fold_id.size != n:
                    raise ValueError(
                        "The length of cv_fold_id should be equal to sample_size."
                    )
                if len(set(self.cv_fold_id)) != self.cv:
                    raise ValueError(
                        "The number of different elements in cv_fold_id should be equal to cv."
                    )
        else:
            self.cv_fold_id = np.array([], dtype="int32")
            if sparsity.size == 1 and self.ic_method is None:
                ic_method = utilities.AIC
            elif sparsity.size > 1 and self.ic_method is None:
                raise ValueError(
                    "ic_method should be provided for choosing sparsity level with information criterion."
                )
            elif self.sample_size <= 1:
                raise ValueError("sample_size should be given when using ic_method.")
            else:
                ic_method = self.ic_method
        if gradient is None and len(layers) > 0:
            if len(layers) == 1:
                assert layers[0].out_features == self.dimensionality
            else:
                for i in range(len(layers) - 1):
                    assert layers[i].out_features == layers[i + 1].in_features
                assert layers[-1].out_features == self.dimensionality
            loss_fn = self.__set_objective_py(objective, None, None, jit, layers)
            p = layers[0].in_features
            for layer in layers[::-1]:
                sparsity = layer.transform_sparsity(sparsity)
                group = layer.transform_group(group)
                preselect = layer.transform_preselect(preselect)
        else:
            # set optimization objective
            if cpp:
                loss_fn = self.__set_objective_cpp(objective, gradient, hessian)
            else:
                loss_fn = self.__set_objective_py(objective, gradient, hessian, jit)

        # init_support_set
        if init_support_set is None:
            init_support_set = np.array([], dtype="int32")
        else:
            init_support_set = np.array(init_support_set, dtype="int32")
            if init_support_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be an 1D array of integers."
                )
            if init_support_set.min() < 0 or init_support_set.max() >= p:
                raise ValueError("init_support_set contains wrong index.")

        # init_params
        if init_params is None:
            random_init = False
            if len(layers) > 0:
                random_init = np.any(
                    np.array([layer.random_initilization for layer in layers])
                )
            if random_init:
                init_params = np.random.RandomState(self.random_state).randn(p)
            else:
                init_params = np.zeros(p, dtype=float)
        else:
            init_params = np.array(init_params, dtype=float)
            if init_params.shape != (p,):
                raise ValueError(
                    "The length of init_params should be equal to dimensionality."
                )

        result = _scope.pywrap_Universal(
            data,
            self.model,
            self.numeric_solver,
            p,
            n,
            0,
            self.max_iter,
            self.max_exchange_num,
            path_type,
            self.greedy,
            self.use_hessian,
            self.is_dynamic_max_exchange_num,
            self.warm_start,
            ic_method,
            self.cv,
            sparsity,
            np.array([0.0]),
            gs_lower_bound,
            gs_upper_bound,
            screening_size,
            group,
            preselect,
            self.thread,
            splicing_type,
            self.important_search,
            self.cv_fold_id,
            init_support_set,
            init_params,
            np.zeros(0),
        )

        self.params = np.array(result[0])
        self.objective_value = loss_fn(self.params, data)
        if len(layers) > 0:
            for layer in layers:
                self.params = layer.transform_params(self.params)

        self.support_set = np.sort(np.nonzero(self.params)[0])
        self.information_criterion = result[3]
        self.cross_validation_loss = result[2] if self.cv > 1 else None

        return self.params

    def get_result(self):
        r"""
        Get the result of optimization.

        Returns
        -------
        results : dict
            The result of optimization, including the following keys:

            + ``params`` : array of shape (dimensionality,)
                The optimal parameters.
            + ``support_set`` : array of int
                The support set of the optimal parameters.
            + ``objective_value`` : float
                The value of objective function at the optimal parameters.
            + ``information_criterion`` : float
                The value of information criterion.
            + ``cross_validation_loss`` : float
                The mean loss of cross-validation.
        """
        return {
            "params": self.params,
            "support_set": self.support_set,
            "objective_value": self.objective_value,
            "information_criterion": self.information_criterion,
            "cross_validation_loss": self.cross_validation_loss,
        }

    def __set_objective_cpp(self, objective, gradient, hessian):
        self.model.set_loss_of_model(lambda params, data: objective(params, data))
        if gradient is None:
            self.model.set_gradient_autodiff(objective)
        else:
            self.model.set_gradient_user_defined(
                lambda params, data: (objective(params, data), gradient(params, data))
            )
        if hessian is None:
            self.model.set_hessian_autodiff(objective)
        else:
            # NOTE: Perfect Forwarding of grad and hess is neccessary for func
            # written in Pybind11_Cpp code
            self.model.set_hessian_user_defined(
                lambda params, data: hessian(params, data)
            )
        return objective

    def __set_objective_py(self, objective, gradient, hessian, jit, layers=[]):
        loss_, grad_ = BaseSolver._set_objective(objective, gradient, jit, layers)

        # hess
        if hessian is None:

            def hess_(params, data):
                return jax.hessian(loss_)(params, data)

        elif hessian.__code__.co_argcount == 1:

            def hess_(params, data):
                return hessian(params)

        else:

            def hess_(params, data):
                return hessian(params, data)

        if jit:
            hess_ = jax.jit(hess_)

        def loss_fn(params, data):
            value = loss_(params, data)
            if not np.isfinite(value):
                raise ValueError("The objective function returned {}.".format(value))
            if isinstance(value, float):
                return value
            return value.item()

        def value_and_grad(params, data):
            value, grad = grad_(params, data)
            if not np.isfinite(value):
                raise ValueError("The objective function returned {}.".format(value))
            if not np.all(np.isfinite(grad)):
                raise ValueError("The gradient returned contains NaN or Inf.")
            if isinstance(value, float):
                return value, np.array(grad)
            return value.item(), np.array(grad)

        def hess_fn(params, data):
            h = np.array(hess_(jnp.array(params), data))
            if not np.all(np.isfinite(h)):
                raise ValueError("The hessian returned contains NaN or Inf.")
            return h

        self.model.set_loss_of_model(loss_fn)
        self.model.set_gradient_user_defined(value_and_grad)
        self.model.set_hessian_user_defined(hess_fn)

        return loss_fn


class HTPSolver(BaseSolver):
    r"""
    Get sparse optimal solution of convex objective function by Gradient Hard Thresholding Pursuit (GraHTP) algorithm.
    Specifically, ``HTPSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.

    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    step_size : float, default=0.005
        Step size of gradient descent.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``HTPSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method.
        - If ``cv`` = 1, the sparsity level will be chosen by the information criterion.
        - If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method : callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id : array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set : array of int
        The indices of selected variables, sorted in ascending order.

    References
    ----------
    Yuan X T, Li P, Zhang T. Gradient Hard Thresholding Pursuit[J]. J. Mach. Learn. Res., 2017, 18(1): 6027-6069.

    """

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        preselect=[],
        step_size=0.005,
        numeric_solver=convex_solver_LBFGS,
        max_iter=100,
        group=None,
        ic_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            preselect=preselect,
            numeric_solver=numeric_solver,
            max_iter=max_iter,
            group=group,
            ic_method=ic_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
            random_state=random_state,
        )
        self.step_size = step_size

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
        preselect,
        group,
    ):
        if sparsity <= preselect.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
                preselect,
                group,
            )
        # init
        params = init_params
        best_suppport_group_tuple = None
        best_loss = np.inf
        results = {}  # key: tuple of ordered support set, value: params
        group_num = len(np.unique(group))
        group_indices = [np.where(group == i)[0] for i in range(group_num)]

        for n_iters in range(self.max_iter):
            # S1: gradient descent
            params_bias = params - self.step_size * value_and_grad(params, data)[1]
            # S2: Gradient Hard Thresholding
            score = np.array(
                [
                    np.sum(np.square(params_bias[group_indices[i]]))
                    for i in range(group_num)
                ]
            )
            score[preselect] = np.inf
            support_new_group = np.argpartition(score, -sparsity)[-sparsity:]
            support_new_group_tuple = tuple(np.sort(support_new_group))
            # terminating condition
            if support_new_group_tuple in results:
                break
            # S3: debise
            params = np.zeros_like(init_params)
            support_new = np.concatenate([group_indices[i] for i in support_new_group])
            params[support_new] = params_bias[support_new]
            loss, params = self._numeric_solver(
                loss_fn, value_and_grad, params, support_new, data
            )
            # update cache
            if loss < best_loss:
                best_loss = loss
                best_suppport_group_tuple = support_new_group_tuple
            results[support_new_group_tuple] = params

        self.n_iters = n_iters
        return results[best_suppport_group_tuple], np.concatenate(
            [group_indices[i] for i in best_suppport_group_tuple]
        )


class IHTSolver(HTPSolver):
    r"""
    Get sparse optimal solution of convex objective function by Iterative Hard Thresholding (IHT) algorithm.
    Specifically, ``IHTSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.

    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    step_size : float, default=0.005
        Step size of gradient descent.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``IHTSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method.
        - If ``cv`` = 1, the sparsity level will be chosen by the information criterion.
        - If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method : callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id : array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set : array of int
        The indices of selected variables, sorted in ascending order.

    References
    ----------
    Yuan X T, Li P, Zhang T. Gradient Hard Thresholding Pursuit[J]. J. Mach. Learn. Res., 2017, 18(1): 6027-6069.

    """

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
        preselect,
        group,
    ):
        if sparsity <= preselect.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
                preselect,
                group,
            )
        # init
        params = init_params
        support_old_group = np.array([], dtype="int32")
        group_num = len(np.unique(group))
        group_indices = [np.where(group == i)[0] for i in range(group_num)]

        for n_iters in range(self.max_iter):
            # S1: gradient descent
            params_bias = params - self.step_size * value_and_grad(params, data)[1]
            # S2: Gradient Hard Thresholding
            score = np.array(
                [
                    np.sum(np.square(params_bias[group_indices[i]]))
                    for i in range(group_num)
                ]
            )
            score[preselect] = np.inf
            support_new_group = np.argpartition(score, -sparsity)[-sparsity:]
            # terminating condition
            if np.all(set(support_old_group) == set(support_new_group)):
                break
            else:
                support_old_group = support_new_group
            # S3: debise
            params = np.zeros_like(init_params)
            support_new = np.concatenate([group_indices[i] for i in support_new_group])
            params[support_new] = params_bias[support_new]

        # final optimization for IHT
        _, params = self._numeric_solver(
            loss_fn, value_and_grad, params, support_new, data
        )
        self.n_iters = n_iters
        return params, support_new


class GraspSolver(BaseSolver):
    r"""
    Get sparse optimal solution of convex objective function by Gradient Support Pursuit (GraSP) algorithm.
    Specifically, ``GraspSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.


    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``GraspSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method.
        - If ``cv`` = 1, the sparsity level will be chosen by the information criterion.
        - If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method : callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id : array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set : array of int
        The indices of selected variables, sorted in ascending order.

    References
    ----------
    Bahmani S, Raj B, Boufounos P T. Greedy sparsity-constrained optimization[J]. The Journal of Machine Learning Research, 2013, 14(1): 807-841.
    """

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
        preselect,
        group,
    ):
        if sparsity <= preselect.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
                preselect,
                group,
            )
        # init
        params = init_params
        best_suppport_tuple = None
        best_loss = np.inf
        results = {}  # key: tuple of ordered support set, value: params
        group_num = len(np.unique(group))
        group_indices = [np.where(group == i)[0] for i in range(group_num)]

        for n_iters in range(self.max_iter):
            # compute local gradient
            grad_values = value_and_grad(params, data)[1]
            score = np.array(
                [
                    np.sum(np.square(grad_values[group_indices[i]]))
                    for i in range(group_num)
                ]
            )
            score[preselect] = np.inf

            # identify directions
            if 2 * sparsity < group_num:
                Omega = [
                    idx
                    for idx in np.argpartition(score, -2 * sparsity)[-2 * sparsity :]
                    if score[idx] != 0.0
                ]  # supp of top 2k largest absolute values of gradient
            else:
                Omega = np.nonzero(score)[0]  # supp(z)

            # merge supports
            support_new = np.concatenate([group_indices[i] for i in Omega])
            support_new = np.unique(np.append(support_new, params.nonzero()[0]))
            suppport_tuple = tuple(np.sort(support_new))

            # terminating condition
            if suppport_tuple in results:
                break

            # minimize
            params_bias = np.zeros_like(init_params)
            params_bias[support_new] = params[support_new]
            _, params_bias = self._numeric_solver(
                loss_fn, value_and_grad, params_bias, support_new, data
            )

            # prune estimate
            score = np.array(
                [
                    np.sum(np.square(params_bias[group_indices[i]]))
                    for i in range(group_num)
                ]
            )
            score[preselect] = np.inf
            support_set_group = np.argpartition(score, -sparsity)[-sparsity:]
            support_set = np.concatenate([group_indices[i] for i in support_set_group])
            params = np.zeros_like(init_params)
            params[support_set] = params_bias[support_set]

            # update cache
            loss = loss_fn(params, data)
            if loss < best_loss:
                best_loss = loss
                best_suppport_tuple = suppport_tuple
            results[suppport_tuple] = params

        self.n_iters = n_iters
        return results[best_suppport_tuple], np.nonzero(results[best_suppport_tuple])[0]


class FobaSolver(BaseSolver):
    r"""
    Get sparse optimal solution of convex objective function by Forward-Backward greedy (FoBa) algorithm.
    Specifically, ``FobaSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.


    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    use_gradient : bool, default=True
        Whether to use gradient information to metric the importance of variables or not.
        Using gradient information will accelerate the algorithm but the solution may be not accurate.
    threshold : float, default=0.0
        The threshold to determine whether a variable is selected or not.
    foba_threshold_ratio : float, default=0.5
        The threshold for determining whether a variable is deleted or not will be set to ``threshold`` * ``foba_threshold_ratio``.
    strict_sparsity : bool, default=True
        Whether to strictly control the sparsity level to be ``sparsity`` or not.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``FobaSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method.
        - If ``cv`` = 1, the sparsity level will be chosen by the information criterion.
        - If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method : callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id : array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set : array of int
        The indices of selected variables, sorted in ascending order.

    References
    ----------
    Liu J, Ye J, Fujimaki R. Forward-backward greedy algorithms for general convex smooth functions over a cardinality constraint[C]//International Conference on Machine Learning. PMLR, 2014: 503-511.
    """

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        use_gradient=True,
        threshold=0.0,
        foba_threshold_ratio=0.5,
        strict_sparsity=True,
        preselect=[],
        numeric_solver=convex_solver_LBFGS,
        max_iter=100,
        group=None,
        ic_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            preselect=preselect,
            numeric_solver=numeric_solver,
            max_iter=max_iter,
            group=group,
            ic_method=ic_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
            random_state=random_state,
        )
        self.threshold = threshold
        self.use_gradient = use_gradient
        self.foba_threshold_ratio = foba_threshold_ratio
        self.strict_sparsity = strict_sparsity

    def _forward_step(
        self, loss_fn, value_and_grad, params, support_set_group, data, group_indices
    ):
        if self.use_gradient:
            # FoBa-gdt algorithm
            value_old, grad = value_and_grad(params, data)
            score = np.array(
                [
                    np.sum(np.square(grad[group_indices[i]]))
                    for i in range(len(group_indices))
                ]
            )
            score[support_set_group] = -np.inf
        else:
            # FoBa-obj algorithm
            value_old = loss_fn(params, data)
            score = np.empty(len(group_indices), dtype=float)
            for idx in range(len(group_indices)):
                if idx in support_set_group:
                    score[idx] = -np.inf
                    continue
                cache_param = params[group_indices[idx]]
                score[idx] = (
                    value_old
                    - self._numeric_solver(
                        loss_fn,
                        value_and_grad,
                        params,
                        group_indices[idx],
                        data=data,
                    )[0]
                )
                params[group_indices[idx]] = cache_param

        direction = np.argmax(score)
        if score[direction] < self.threshold:
            return params, support_set_group, -1.0
        support_set_group = np.append(support_set_group, direction)

        inactive_set = np.ones_like(params, dtype=bool)
        support_set = np.concatenate([group_indices[i] for i in support_set_group])
        inactive_set[support_set] = False
        params[inactive_set] = 0.0
        value_new, params = self._numeric_solver(
            loss_fn,
            value_and_grad,
            params,
            support_set,
            data=data,
        )

        return params, support_set_group, value_old - value_new

    def _backward_step(
        self,
        loss_fn,
        value_and_grad,
        params,
        support_set_group,
        data,
        backward_threshold,
        group_indices,
    ):
        score = np.empty(len(group_indices), dtype=float)
        for idx in range(len(group_indices)):
            if idx not in support_set_group or idx in self.preselect:
                score[idx] = np.inf
                continue
            cache_param = params[group_indices[idx]]
            params[group_indices[idx]] = 0.0
            score[idx] = loss_fn(params, data)
            params[group_indices[idx]] = cache_param
        direction = np.argmin(score)
        if score[direction] >= backward_threshold:
            return params, support_set_group, False

        support_set_group = np.delete(
            support_set_group, np.argwhere(support_set_group == direction)
        )

        inactive_set = np.ones_like(params, dtype=bool)
        support_set = np.concatenate([group_indices[i] for i in support_set_group])
        inactive_set[support_set] = False
        params[inactive_set] = 0.0
        _, params = self._numeric_solver(
            loss_fn,
            value_and_grad,
            params,
            support_set,
            data=data,
        )

        return params, support_set_group, True

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
        preselect,
        group,
    ):
        if sparsity <= preselect.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
                preselect,
                group,
            )
        # init
        params = np.zeros_like(init_params)
        support_set_group = preselect
        threshold = {}
        group_num = len(np.unique(group))
        group_indices = [np.where(group == i)[0] for i in range(group_num)]

        for n_iters in range(self.max_iter):
            if support_set_group.size >= min(2 * sparsity, group_num):
                break
            params, support_set_group, backward_threshold = self._forward_step(
                loss_fn, value_and_grad, params, support_set_group, data, group_indices
            )
            if backward_threshold < 0:
                break
            threshold[support_set_group.size] = backward_threshold

            while support_set_group.size > preselect.size:
                params, support_set_group, success = self._backward_step(
                    loss_fn,
                    value_and_grad,
                    params,
                    support_set_group,
                    data,
                    loss_fn(params, data)
                    + threshold[support_set_group.size] * self.foba_threshold_ratio,
                    group_indices,
                )
                if not success:
                    break

        if self.strict_sparsity:
            while support_set_group.size > sparsity:
                params, support_set_group, _ = self._backward_step(
                    loss_fn,
                    value_and_grad,
                    params,
                    support_set_group,
                    data,
                    np.inf,
                    group_indices,
                )
            while support_set_group.size < sparsity:
                params, support_set_group, _ = self._forward_step(
                    loss_fn,
                    value_and_grad,
                    params,
                    support_set_group,
                    data,
                    group_indices,
                )

        return params, np.concatenate([group_indices[i] for i in support_set_group])


class ForwardSolver(FobaSolver):
    r"""
    Get sparse optimal solution of convex objective function by Forward Selection algorithm.
    Specifically, ``ForwardSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.


    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    threshold : float, default=0.0
        The threshold to determine whether a variable is selected or not.
    strict_sparsity : bool, default=True
        Whether to strictly control the sparsity level to be ``sparsity`` or not.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``ForwardSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method.
        - If ``cv`` = 1, the sparsity level will be chosen by the information criterion.
        - If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method : callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id : array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set : array of int
        The indices of selected variables, sorted in ascending order.

    """

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        threshold=0.0,
        strict_sparsity=True,
        preselect=[],
        numeric_solver=convex_solver_LBFGS,
        max_iter=100,
        group=None,
        ic_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            preselect=preselect,
            use_gradient=False,
            strict_sparsity=strict_sparsity,
            threshold=threshold,
            foba_threshold_ratio=0.5,
            numeric_solver=numeric_solver,
            max_iter=max_iter,
            group=group,
            ic_method=ic_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
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
        preselect,
        group,
    ):
        if sparsity <= preselect.size:
            return super()._solve(
                sparsity,
                loss_fn,
                value_and_grad,
                init_support_set,
                init_params,
                data,
                preselect,
                group,
            )
        # init
        params = np.zeros_like(init_params)
        support_set_group = preselect
        group_num = len(np.unique(group))
        group_indices = [np.where(group == i)[0] for i in range(group_num)]

        for iter in range(sparsity - support_set_group.size):
            params, support_set_group, backward_threshold = self._forward_step(
                loss_fn, value_and_grad, params, support_set_group, data, group_indices
            )
            if backward_threshold < 0.0:
                break

        if self.strict_sparsity:
            while support_set_group.size < sparsity:
                params, support_set_group, _ = self._forward_step(
                    loss_fn,
                    value_and_grad,
                    params,
                    support_set_group,
                    data,
                    group_indices,
                )

        return params, np.concatenate([group_indices[i] for i in support_set_group])


class OMPSolver(ForwardSolver):
    r"""
    Get sparse optimal solution of convex objective function by Orthogonal Matching Pursuit (OMP) algorithm.
    Specifically, ``OMPSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.


    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        Sample size, denoted as :math:`n`.
    threshold : float, default=0.0
        The threshold to determine whether a variable is selected or not.
    strict_sparsity : bool, default=True
        Whether to strictly control the sparsity level to be ``sparsity`` or not.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``OMPSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_LBFGS``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_method : callable, optional
        A function to calculate the information criterion for choosing the sparsity level.
        ``ic(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
        Used only when ``sparsity`` is array and ``cv`` is 1.
        Note that ``sample_size`` must be given when using ``ic_method``.
    cv : int, default=1
        The folds number when use the cross-validation method.
        - If ``cv`` = 1, the sparsity level will be chosen by the information criterion.
        - If ``cv`` > 1, the sparsity level will be chosen by the cross-validation method.
    split_method : callable, optional
        A function to get the part of data used in each fold of cross-validation.
        Its interface should be ``(data, index) -> part_data`` where ``index`` is an array of int.
    cv_fold_id : array of shape (sample_size,), optional
        An array indicates different folds in CV, which samples in the same fold should be given the same number.
        The number of different elements should be equal to ``cv``.
        Used only when `cv` > 1.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set : array of int
        The indices of selected variables, sorted in ascending order.

    References
    ----------
    Shalev-Shwartz S, Srebro N, Zhang T. Trading accuracy for sparsity in optimization problems with sparsity constraints[J]. SIAM Journal on Optimization, 2010, 20(6): 2807-2832.Shalev-Shwartz S, Srebro N, Zhang T. Trading accuracy for sparsity in optimization problems with sparsity constraints[J]. SIAM Journal on Optimization, 2010, 20(6): 2807-2832.
    """

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        threshold=0.0,
        strict_sparsity=True,
        preselect=[],
        numeric_solver=convex_solver_LBFGS,
        max_iter=100,
        group=None,
        ic_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        super().__init__(
            dimensionality=dimensionality,
            sparsity=sparsity,
            sample_size=sample_size,
            threshold=threshold,
            strict_sparsity=strict_sparsity,
            preselect=preselect,
            numeric_solver=numeric_solver,
            max_iter=max_iter,
            group=group,
            ic_method=ic_method,
            cv=cv,
            cv_fold_id=cv_fold_id,
            split_method=split_method,
            random_state=random_state,
        )
        self.use_gradient = True
