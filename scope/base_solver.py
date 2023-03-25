from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import numpy as np
import jax
import nlopt
import math


class BaseSolver(BaseEstimator):
    """
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
    """

    def __init__(
        self,
        dimensionality,
        sparsity=None,
        sample_size=1,
        *,
        always_select=[],
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
        self.dimensionality = dimensionality
        self.sample_size = sample_size
        self.sparsity = sparsity
        self.always_select = always_select
        self.max_iter = max_iter
        self.ic_type = ic_type
        self.ic_coef = ic_coef
        self.metric_method = metric_method
        self.cv = cv
        self.cv_fold_id = cv_fold_id
        self.split_method = split_method
        self.jax_platform = jax_platform
        self.random_state = random_state
        self.nlopt_solver = nlopt_solver

    def get_config(self, deep=True):
        return super().get_params(deep)

    def set_config(self, **params):
        return super().set_params(**params)

    def solve(
        self,
        objective,
        gradient=None,
        init_support_set=None,
        init_params=None,
        data=None,
        jit=False,
    ):
        r"""
        Set the optimization objective and begin to solve

        Parameters
        ----------
        + objective : function('params': array of float(, 'data': custom class)) ->  float
            Defined the objective of optimization, must be written in JAX if gradient is not provided.
        + gradient : function('params': array of float(, 'data': custom class)) -> array of float
            Defined the gradient of objective function, return the gradient of parameters.
        + init_support_set : array-like of int, optional, default=[]
            The index of the variables in initial active set.
        + init_params : array-like of float, optional, default is an all-zero vector
            An initial value of parameters.
        + data : custom class, optional, default=None
            The data that objective function should be known, like samples, responses, weights, etc, which is necessary for cross validation. It can be any class which is match to objective function.
        + jit : bool, optional, default=False
            just-in-time compilation with XLA, but it should be a pure function.
        """
        if self.jax_platform not in ["cpu", "gpu", "tpu"]:
            raise ValueError("jax_platform must be in 'cpu', 'gpu', 'tpu'")
        jax.config.update("jax_platform_name", self.jax_platform)

        BaseSolver._check_positive_integer(self.dimensionality, "dimensionality")
        BaseSolver._check_positive_integer(self.sample_size, "sample_size")
        BaseSolver._check_non_negative_integer(self.max_iter, "max_iter")

        # always_select
        self.always_select = np.unique(np.array(self.always_select, dtype="int32"))
        if self.always_select.size > 0 and (
            self.always_select[0] < 0 or self.always_select[-1] >= self.dimensionality
        ):
            raise ValueError("always_select should be between 0 and dimensionality.")

        # default sparsity level
        force_min_sparsity = self.always_select.size
        default_max_sparsity = max(
            force_min_sparsity,
            self.dimensionality
            if self.dimensionality <= 5
            else int(
                self.dimensionality
                / np.log(np.log(self.dimensionality))
                / np.log(self.dimensionality)
            ),
        )

        # sparsity
        if self.sparsity == None:
            self.sparsity = np.arange(
                force_min_sparsity,
                default_max_sparsity + 1,
                dtype="int32",
            )
        else:
            self.sparsity = np.unique(np.array(self.sparsity, dtype="int32"))
            if (
                self.sparsity[0] < force_min_sparsity
                or self.sparsity[-1] > self.dimensionality
            ):
                raise ValueError(
                    "All sparsity should be between 0 (when `always_select` is default) and dimensionality."
                )

        BaseSolver._check_positive_integer(self.cv, "cv")
        if self.cv == 1:
            if self.ic_type not in ["aic", "bic", "gic", "ebic"]:
                raise ValueError(
                    "ic_type should be one of ['aic', 'bic', 'gic','ebic']."
                )
            if self.ic_coef <= 0:
                raise ValueError("ic_coef should be positive.")
        else:
            if self.cv > self.sample_size:
                raise ValueError("cv should not be greater than sample_size")
            if data is None and self.split_method is None:
                data = np.arange(self.sample_size)
                self.split_method = lambda data, index: index
            if data is None:
                raise ValueError("data should be provided when cv > 1")
            if self.split_method is None:
                raise ValueError("split_method should be provided when cv > 1")
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
                if self.cv_fold_id.size != self.sample_size:
                    raise ValueError(
                        "The length of group should be equal to X.shape[0]."
                    )
                if len(set(self.cv_fold_id)) != self.cv:
                    raise ValueError(
                        "The number of different masks should be equal to `cv`."
                    )

        if init_support_set is None:
            init_support_set = np.array([], dtype="int32")
        else:
            init_support_set = np.array(init_support_set, dtype="int32")
            if init_support_set.ndim > 1:
                raise ValueError(
                    "The initial active set should be " "an 1D array of integers."
                )
            if (
                init_support_set.min() < 0
                or init_support_set.max() >= self.dimensionality
            ):
                raise ValueError("init_support_set contains wrong index.")

        if init_params is None:
            init_params = np.zeros(self.dimensionality, dtype=float)
        else:
            init_params = np.array(init_params, dtype=float)
            if init_params.shape != (self.dimensionality,):
                raise ValueError(
                    "The length of init_params must match `dimensionality`!"
                )

        loss_, grad_ = BaseSolver._set_objective(objective, gradient, jit)
        loss_fn = lambda params, data: loss_(params, data).item()

        def value_and_grad(params, data):
            value, grad = grad_(params, data)
            return value.item(), np.array(grad)

        if self.cv == 1:
            is_first_loop: bool = True
            for s in self.sparsity:
                init_params, init_support_set = self._solve(
                    s, loss_fn, value_and_grad, init_support_set, init_params, data
                )  ## warm start: use results of previous sparsity as initial value
                value_of_objective = loss_fn(init_params, data)
                eval = self._metric(
                    value_of_objective,
                    self.ic_type,
                    s,
                    self.sample_size,
                )
                if is_first_loop or eval < self.eval_objective:
                    is_first_loop = False
                    self.params = init_params
                    self.support_set = init_support_set
                    self.value_of_objective = value_of_objective
                    self.eval_objective = eval

        else:  # self.cv > 1
            cv_eval = {s: 0.0 for s in self.sparsity}
            cache_init_support_set = {}
            cache_init_params = {}
            for s in self.sparsity:
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
                    )  ## warm start: use results of previous sparsity as initial value
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
            )
            self.value_of_objective = loss_fn(self.params, data)
            self.eval_objective = cv_eval[best_sparsity]

        return self.params

    @staticmethod
    def _set_objective(objective, gradient, jit):
        # objective function
        if objective.__code__.co_argcount == 2:
            loss_ = objective
        elif objective.__code__.co_argcount == 1:
            loss_ = lambda params, data: objective(params)
        else:
            raise ValueError("The objective function should have 1 or 2 arguments.")
        if jit:
            loss_ = jax.jit(loss_)

        if gradient is None:
            grad_ = lambda params, data: jax.value_and_grad(loss_)(params, data)
        elif gradient.__code__.co_argcount == 2:
            grad_ = lambda params, data: (loss_(params, data), gradient(params, data))
        elif gradient.__code__.co_argcount == 1:
            grad_ = lambda params, data: (loss_(params, data), gradient(params))
        else:
            raise ValueError("The gradient function should have 1 or 2 arguments.")
        if jit:
            grad_ = jax.jit(grad_)

        return loss_, grad_

    def _metric(
        self,
        value_of_objective: float,
        method: str,
        effective_params_num: int,
        train_size: int,
    ) -> float:
        """
        aic: 2L + 2s
        bic: 2L + s * log(n)
        gic: 2L + s * log(log(n)) * log(p)
        ebic: 2L + s * (log(n) + 2 * log(p))
        """
        if self.metric_method is not None:
            return self.metric_method(
                value_of_objective,
                self.dimensionality,
                effective_params_num,
                train_size,
            )

        if method == "aic":
            return 2 * value_of_objective + 2 * effective_params_num
        elif method == "bic":
            return (
                value_of_objective
                if train_size <= 1.0
                else 2 * value_of_objective
                + self.ic_coef * effective_params_num * np.log(train_size)
            )
        elif method == "gic":
            return (
                value_of_objective
                if train_size <= 1.0
                else 2 * value_of_objective
                + self.ic_coef
                * effective_params_num
                * np.log(np.log(train_size))
                * np.log(self.dimensionality)
            )
        elif method == "ebic":
            return 2 * value_of_objective + self.ic_coef * effective_params_num * (
                np.log(train_size) + 2 * np.log(self.dimensionality)
            )
        else:
            return value_of_objective

    def _solve(
        self,
        sparsity,
        loss_fn,
        value_and_grad,
        init_support_set,
        init_params,
        data,
    ):
        """
        Solve the optimization problem with given sparsity. Need to be implemented by corresponding concrete class.

        Parameters
        ----------
        sparsity: int
            The number of non-zero parameters.
        loss_fn: Callable[[Sequence[float], Any], float]
            The loss function.
        value_and_grad: Callable[[Sequence[float], Any], Tuple[float, Sequence[float]]]
            The function to compute the loss and gradient.
        init_params: Sequence[float]
            The complete initial parameters. This is only a suggestion. Whether it is effective depends on the specific algorithm
        init_support_set: Sequence[int]
            The initial support_set. This is only a suggestion. Whether it is effective depends on the specific algorithm
        data: Any
            The data passed to loss_fn and value_and_grad.
        Returns
        -------
        params: Sequence[float]
            The solution of optimization.
        support_set: Sequence[int]
            The index of selected variables which is the non-zero parameters.
        """
        if sparsity == 0:
            return np.zeros(self.dimensionality), np.array([], dtype=int)
        if sparsity < self.always_select.size:
            raise ValueError(
                "The number of always selected variables is larger than the sparsity."
            )

        if (
            math.comb(
                self.dimensionality - self.always_select.size,
                sparsity - self.always_select.size,
            )
            > self.max_iter
        ):
            raise ValueError(
                "The number of subsets is too large, please reduce the sparsity, dimensionality or increase max_iter."
            )

        def all_subsets(p: int, s: int, always_select: np.ndarray = np.zeros(0)):
            universal_set = np.setdiff1d(np.arange(p), always_select)
            p = p - always_select.size
            s = s - always_select.size

            def helper(start: int, s: str, curr_selection: np.ndarray):
                if s == 0:
                    yield curr_selection
                else:
                    for i in range(start, p - s + 1):
                        yield from helper(
                            i + 1, s - 1, np.append(curr_selection, universal_set[i])
                        )

            yield from helper(0, s, always_select)

        result = {"params": None, "support_set": None, "value_of_objective": math.inf}
        for support_set in all_subsets(
            self.dimensionality, sparsity, self.always_select
        ):
            opt_params, loss = self._cache_nlopt(
                loss_fn, value_and_grad, init_params, support_set, data
            )
            if loss < result["value_of_objective"]:
                params = np.zeros(self.dimensionality)
                params[support_set] = opt_params
                result["params"] = params
                result["support_set"] = support_set
                result["value_of_objective"] = loss

        return result["params"], result["support_set"]

    def _cache_nlopt(
        self,
        loss_fn,
        value_and_grad,
        init_params,
        optim_variable_idx,
        data,
        background_params=None,
    ):
        """
        Nlopt often throws RuntimeError even if the optimization is nearly successful.
        This function is used to cache the best result and return it.

        Parameters
        ----------
        loss_fn: Callable[[Sequence[float], Any], float]
            The loss function.
        value_and_grad: Callable[[Sequence[float], Any], Tuple[float, Sequence[float]]]
            The function to compute the loss and gradient.
        init_params: Sequence[float]
            The complete initial parameters.
        optim_variable_idx: Sequence[int]
            The index of variables to be optimized.
        data: Any
            The data passed to loss_fn and value_and_grad.
        background_params: Sequence[float] | None
            The values of parameters that are not optimized, default is all zeros.

        Returns
        -------
        optim_params: Sequence[float]
            The optimized parameters, whose length is the same as optim_variable_idx.
        loss: float
            The loss of the optimized parameters.
        """
        if len(optim_variable_idx) == 0:
            return np.zeros(0), math.inf
        best_loss = math.inf
        best_params = None
        if background_params is None:
            background_params = np.zeros(self.dimensionality)
        else:
            background_params = np.copy(
                background_params
            )  # avoid changing the original background_params

        def cache_opt_fn(x, grad):
            nonlocal best_loss, best_params
            background_params[
                optim_variable_idx
            ] = x  # update the nonlocal variable: background_params
            if grad.size > 0:
                loss, full_grad = value_and_grad(background_params, data)
                grad[:] = full_grad[optim_variable_idx]
            else:
                loss = loss_fn(background_params, data)
            if loss < best_loss:
                best_loss = loss
                best_params = np.copy(x)
            return loss

        nlopt_solver = nlopt.opt(
            self.nlopt_solver.get_algorithm(), optim_variable_idx.size
        )
        if nlopt_solver.get_algorithm_name() != self.nlopt_solver.get_algorithm_name():
            raise ValueError("The algorithm of nlopt_solver is invalid.")
        nlopt_solver.set_stopval(self.nlopt_solver.get_stopval())
        nlopt_solver.set_ftol_rel(self.nlopt_solver.get_ftol_rel())
        nlopt_solver.set_ftol_abs(self.nlopt_solver.get_ftol_abs())
        nlopt_solver.set_xtol_rel(self.nlopt_solver.get_xtol_rel())
        nlopt_solver.set_maxtime(self.nlopt_solver.get_maxtime())
        nlopt_solver.set_population(self.nlopt_solver.get_population())
        nlopt_solver.set_vector_storage(self.nlopt_solver.get_vector_storage())
        nlopt_solver.set_min_objective(cache_opt_fn)

        try:
            opt_params = nlopt_solver.optimize(init_params[optim_variable_idx])
            return opt_params, nlopt_solver.last_optimum_value()
        except RuntimeError:
            return best_params, best_loss

    def get_result(self):
        r"""
        Get the solution of optimization, include the parameters ...
        """
        return {
            "params": self.params,
            "support_set": self.support_set,
            "value_of_objective": self.value_of_objective,
            "eval_objective": self.eval_objective,
        }

    @staticmethod
    def _check_positive_integer(var, name: str):
        if not isinstance(var, int) or var <= 0:
            raise ValueError("{} should be an positive integer.".format(name))

    @staticmethod
    def _check_non_negative_integer(var, name: str):
        if not isinstance(var, int) or var < 0:
            raise ValueError("{} should be an non-negative integer.".format(name))
