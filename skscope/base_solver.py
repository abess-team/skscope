#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Zezhi Wang
# Copyright (C) 2023 abess-team
# Licensed under the MIT License.

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import numpy as np
import jax
from .numeric_solver import convex_solver_nlopt
import math
from . import layer


class BaseSolver(BaseEstimator):
    r"""
    Get sparse optimal solution of convex objective function by searching all possible combinations of variables.
    Specifically, ``BaseSolver`` aims to tackle this problem: :math:`\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s`, where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.


    Parameters
    ----------
    dimensionality : int
        Dimension of the optimization problem, which is also the total number of variables that will be considered to select or not, denoted as :math:`p`.
    sparsity : int or array of int, optional
        The sparsity level, which is the number of nonzero elements of the optimal solution, denoted as :math:`s`.
        Default is ``range(int(p/log(log(p))/log(p)))``.
    sample_size : int, default=1
        sample size, denoted as :math:`n`.
    preselect : array of int, default=[]
        An array contains the indexes of variables which must be selected.
    numeric_solver : callable, optional
        A solver for the convex optimization problem. ``BaseSolver`` will call this function to solve the convex optimization problem in each iteration.
        It should have the same interface as ``skscope.convex_solver_nlopt``.
    max_iter : int, default=100
        Maximum number of iterations taken for converging.
    group : array of shape (dimensionality,), default=range(dimensionality)
        The group index for each variable, and it must be an incremental integer array starting from 0 without gap.
        The variables in the same group must be adjacent, and they will be selected together or not.
        Here are wrong examples: ``[0,2,1,2]`` (not incremental), ``[1,2,3,3]`` (not start from 0), ``[0,2,2,3]`` (there is a gap).
        It's worth mentioning that the concept "a variable" means "a group of variables" in fact. For example,``sparsity=[3]`` means there will be 3 groups of variables selected rather than 3 variables,
        and ``always_include=[0,3]`` means the 0-th and 3-th groups must be selected.
    ic_type : {'aic', 'bic', 'sic', 'ebic'}, default='aic'
        The type of information criterion for choosing the sparsity level.
        Used only when ``sparsity`` is array and ``cv`` is 1.
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
    metric_method : callable, optional
        A function to calculate the information criterion.
        ``metric(loss, p, s, n) -> ic_value``, where ``loss`` is the value of the objective function, ``p`` is the dimensionality, ``s`` is the sparsity level and ``n`` is the sample size.
    random_state : int, optional
        The random seed used for cross-validation.

    Attributes
    ----------
    params : array of shape(dimensionality,)
        The sparse optimal solution.
    objective_value: float
        The value of objective function on the solution.
    support_set :array of int
        The indices of selected variables, sorted in ascending order.

    """

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        preselect=[],
        numeric_solver=convex_solver_nlopt,
        max_iter=100,
        group=None,
        ic_type="aic",
        metric_method=None,
        cv=1,
        cv_fold_id=None,
        split_method=None,
        random_state=None,
    ):
        self.dimensionality = dimensionality
        self.sample_size = sample_size
        self.sparsity = sparsity
        self.preselect = preselect
        self.max_iter = max_iter
        self.group = group
        self.ic_type = ic_type
        self.ic_coef = 1.0
        self.metric_method = metric_method
        self.cv = cv
        self.cv_fold_id = cv_fold_id
        self.split_method = split_method
        self.jax_platform = "cpu"
        self.random_state = random_state
        self.numeric_solver = numeric_solver

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

    def solve(
        self,
        objective,
        data=None,
        layers=[],
        gradient=None,
        init_support_set=None,
        init_params=None,
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
        jax.config.update("jax_platform_name", self.jax_platform)

        BaseSolver._check_positive_integer(self.dimensionality, "dimensionality")
        BaseSolver._check_positive_integer(self.sample_size, "sample_size")
        BaseSolver._check_non_negative_integer(self.max_iter, "max_iter")

        # group
        if self.group is None:
            self.group = np.arange(self.dimensionality, dtype="int32")
            group_num = self.dimensionality  # len(np.unique(group))
        else:
            self.group = np.array(self.group)
            if self.group.ndim > 1:
                raise ValueError("Group should be an 1D array of integers.")
            if self.group.size != self.dimensionality:
                raise ValueError(
                    "The length of group should be equal to dimensionality."
                )
            group_num = len(np.unique(self.group))
            if self.group[0] != 0:
                raise ValueError("Group should start from 0.")
            if any(self.group[1:] - self.group[:-1] < 0):
                raise ValueError("Group should be an incremental integer array.")
            if not group_num == max(self.group) + 1:
                raise ValueError("There is a gap in group.")

        # preselect
        self.preselect = np.unique(np.array(self.preselect, dtype="int32"))
        if self.preselect.size > 0 and (
            self.preselect[0] < 0 or self.preselect[-1] >= group_num
        ):
            raise ValueError("preselect should be between 0 and dimensionality.")

        # default sparsity level
        force_min_sparsity = self.preselect.size
        default_max_sparsity = max(
            force_min_sparsity,
            group_num
            if group_num <= 5
            else int(group_num / np.log(np.log(group_num)) / np.log(group_num)),
        )

        # sparsity
        if self.sparsity is None:
            self.sparsity = np.arange(
                force_min_sparsity,
                default_max_sparsity + 1,
                dtype="int32",
            )
        else:
            self.sparsity = np.unique(np.array(self.sparsity, dtype="int32"))
            if self.sparsity.size == 0:
                raise ValueError("Sparsity should not be empty.")
            if self.sparsity[0] < force_min_sparsity or self.sparsity[-1] > group_num:
                raise ValueError("There is an invalid sparsity.")

        BaseSolver._check_positive_integer(self.cv, "cv")
        if self.cv == 1:
            if self.ic_type not in ["aic", "bic", "sic", "ebic"]:
                raise ValueError(
                    "ic_type should be one of ['aic', 'bic', 'sic','ebic']."
                )
        else:
            if self.cv > self.sample_size:
                raise ValueError("cv should not be greater than sample_size.")
            if data is None and self.split_method is None:
                data = np.arange(self.sample_size)
                self.split_method = lambda data, index: index
            if self.split_method is None:
                raise ValueError("split_method should be provided when cv > 1.")
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
                    raise ValueError("cv_fold_id should be an 1D array of integers.")
                if self.cv_fold_id.size != self.sample_size:
                    raise ValueError(
                        "The length of cv_fold_id should be equal to sample_size."
                    )
                if len(set(self.cv_fold_id)) != self.cv:
                    raise ValueError(
                        "The number of different elements in cv_fold_id should be equal to cv."
                    )

        sparsity = self.sparsity
        group = self.group
        preselect = self.preselect
        if gradient is None and len(layers) > 0:
            if len(layers) == 1:
                assert layers[0].out_features == self.dimensionality
            else:
                for i in range(len(layers) - 1):
                    assert layers[i].out_features == layers[i + 1].in_features
                assert layers[-1].out_features == self.dimensionality
            loss_, grad_ = BaseSolver._set_objective(objective, gradient, jit, layers)
            p = layers[0].in_features
            for layer in layers[::-1]:
                sparsity = layer.transform_sparsity(sparsity)
                group = layer.transform_group(group)
                preselect = layer.transform_preselect(preselect)
        else:
            p = self.dimensionality
            loss_, grad_ = BaseSolver._set_objective(objective, gradient, jit)

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

        if self.cv == 1:
            is_first_loop: bool = True
            for s in sparsity:
                init_params, init_support_set = self._solve(
                    s,
                    loss_fn,
                    value_and_grad,
                    init_support_set,
                    init_params,
                    data,
                    preselect,
                    group,
                )  # warm start: use results of previous sparsity as initial value
                objective_value = loss_fn(init_params, data)
                eval = self._metric(
                    objective_value,
                    self.ic_type,
                    s,
                    self.sample_size,
                )
                if is_first_loop or eval < self.eval_objective:
                    is_first_loop = False
                    self.params = init_params
                    self.support_set = init_support_set
                    self.objective_value = objective_value
                    self.eval_objective = eval

        else:  # self.cv > 1
            cv_eval = {s: 0.0 for s in sparsity}
            cache_init_support_set = {}
            cache_init_params = {}
            for s in sparsity:
                for i in range(self.cv):
                    train_index = np.where(self.cv_fold_id != i)[0]
                    test_index = np.where(self.cv_fold_id == i)[0]
                    init_params, init_support_set = self._solve(
                        s,
                        loss_fn,
                        value_and_grad,
                        init_support_set,
                        init_params,
                        self.split_method(data, train_index),
                        preselect,
                        group,
                    )  # warm start: use results of previous sparsity as initial value
                    cv_eval[s] += loss_fn(
                        init_params, self.split_method(data, test_index)
                    )
                cache_init_support_set[s] = init_support_set
                cache_init_params[s] = init_params
            best_sparsity = min(cv_eval, key=cv_eval.get)
            self.params, self.support_set = self._solve(
                best_sparsity,
                loss_fn,
                value_and_grad,
                cache_init_support_set[best_sparsity],
                cache_init_params[best_sparsity],
                data,
                preselect,
                group,
            )
            self.objective_value = loss_fn(self.params, data)
            self.eval_objective = cv_eval[best_sparsity]

        if len(layers) > 0:
            for layer in layers:
                self.params = layer.transform_params(self.params)

        self.support_set = np.sort(np.nonzero(self.params)[0])
        return self.params

    @staticmethod
    def _set_objective(objective, gradient, jit, layers=[]):
        # objective function
        if objective.__code__.co_argcount == 1:
            if len(layers) == 0:

                def loss_(params, data):
                    return objective(params)

            else:

                def loss_(params, data):
                    for layer in layers:
                        params = layer.transform_params(params)
                    return objective(params)

        else:
            if len(layers) == 0:

                def loss_(params, data):
                    return objective(params, data)

            else:

                def loss_(params, data):
                    for layer in layers:
                        params = layer.transform_params(params)
                    return objective(params, data)

        if jit:
            loss_ = jax.jit(loss_)

        if gradient is None:

            def grad_(params, data):
                return jax.value_and_grad(loss_)(params, data)

        elif gradient.__code__.co_argcount == 1:

            def grad_(params, data):
                return (loss_(params, data), gradient(params))

        else:

            def grad_(params, data):
                return (loss_(params, data), gradient(params, data))

        if jit:
            grad_ = jax.jit(grad_)

        return loss_, grad_

    def _metric(
        self,
        objective_value: float,
        method: str,
        effective_params_num: int,
        train_size: int,
    ) -> float:
        """
        aic: 2L + 2s
        bic: 2L + s * log(n)
        sic: 2L + s * log(log(n)) * log(p)
        ebic: 2L + s * (log(n) + 2 * log(p))
        """
        if self.metric_method is not None:
            return self.metric_method(
                objective_value,
                self.dimensionality,
                effective_params_num,
                train_size,
            )

        if method == "aic":
            return 2 * objective_value + 2 * effective_params_num
        elif method == "bic":
            return (
                objective_value
                if train_size <= 1.0
                else 2 * objective_value
                + self.ic_coef * effective_params_num * np.log(train_size)
            )
        elif method == "sic":
            return (
                objective_value
                if train_size <= 1.0
                else 2 * objective_value
                + self.ic_coef
                * effective_params_num
                * np.log(np.log(train_size))
                * np.log(self.dimensionality)
            )
        elif method == "ebic":
            return 2 * objective_value + self.ic_coef * effective_params_num * (
                np.log(train_size) + 2 * np.log(self.dimensionality)
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
        if sparsity == 0:
            return np.zeros_like(init_params), np.array([], dtype=int)
        if sparsity < preselect.size:
            raise ValueError(
                "The number of always selected variables is larger than the sparsity."
            )
        group_num = len(np.unique(group))
        group_indices = [np.where(group == i)[0] for i in range(group_num)]

        if (
            math.comb(
                group_num - preselect.size,
                sparsity - preselect.size,
            )
            > self.max_iter
        ):
            raise ValueError(
                "The number of subsets is too large, please reduce the sparsity, dimensionality or increase max_iter."
            )

        def all_subsets(p: int, s: int, preselect: np.ndarray = np.zeros(0)):
            universal_set = np.setdiff1d(np.arange(group_num), preselect)
            p = p - preselect.size
            s = s - preselect.size

            def helper(start: int, s: str, curr_selection: np.ndarray):
                if s == 0:
                    yield curr_selection
                else:
                    for i in range(start, p - s + 1):
                        yield from helper(
                            i + 1, s - 1, np.append(curr_selection, universal_set[i])
                        )

            yield from helper(0, s, preselect)

        result = {"params": None, "support_set": None, "objective_value": math.inf}
        params = init_params.copy()
        for support_set_group in all_subsets(group_num, sparsity, preselect):
            support_set = np.concatenate([group_indices[i] for i in support_set_group])
            inactive_set = np.ones_like(init_params, dtype=bool)
            inactive_set[support_set] = False
            params[inactive_set] = 0.0
            params[support_set] = init_params[support_set]
            loss, params = self._numeric_solver(
                loss_fn, value_and_grad, params, support_set, data
            )
            if loss < result["objective_value"]:
                result["params"] = params.copy()
                result["support_set"] = support_set
                result["objective_value"] = loss

        return result["params"], result["support_set"]

    def _numeric_solver(
        self,
        loss_fn,
        value_and_grad,
        params,
        optim_variable_set,
        data,
    ):
        """
        Solve the optimization problem with given support set.

        Parameters
        ----------
        loss_fn: Callable[[Sequence[float], Any], float]
            The loss function.
        value_and_grad: Callable[[Sequence[float], Any], Tuple[float, Sequence[float]]]
            The function to compute the loss and gradient.
        params: Sequence[float]
            The complete initial parameters.
        optim_variable_set: Sequence[int]
            The index of variables to be optimized.
        data: Any
            The data passed to loss_fn and value_and_grad.

        Returns
        -------
        loss: float
            The loss of the optimized parameters, i.e., `loss_fn(params, data)`.
        optimized_params: Sequence[float]
            The optimized parameters.
        """
        if not isinstance(params, np.ndarray) or params.ndim != 1:
            raise ValueError("params should be a 1D np.ndarray.")
        if (
            not isinstance(optim_variable_set, np.ndarray)
            or optim_variable_set.ndim != 1
        ):
            raise ValueError("optim_variable_set should be a 1D np.ndarray.")

        if optim_variable_set.size == 0:
            return loss_fn(params, data)

        return self.numeric_solver(
            loss_fn, value_and_grad, params, optim_variable_set, data
        )

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
        """
        return {
            "params": self.params,
            "support_set": self.support_set,
            "objective_value": self.objective_value,
        }

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
    def _check_positive_integer(var, name: str):
        if not isinstance(var, int) or var <= 0:
            raise ValueError("{} should be an positive integer.".format(name))

    @staticmethod
    def _check_non_negative_integer(var, name: str):
        if not isinstance(var, int) or var < 0:
            raise ValueError("{} should be an non-negative integer.".format(name))
