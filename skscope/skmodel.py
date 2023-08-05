import numpy as np
import jax.numpy as jnp
from skscope import ScopeSolver
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import (
    check_array,
    check_random_state,
    check_X_y,
    check_is_fitted,
)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._param_validation import Hidden, Interval, StrOptions
from numbers import Integral, Real


class PortfolioSelection(BaseEstimator):
    r"""
    Construct a sparse portfolio using ``skscope`` with ``MinVar`` or ``MeanVar`` measure.

    Parameters
    ------------
    sparsity : int, default=10
        The number of stocks to be chosen, i.e., the sparsity level

    obj : {"MinVar", "MeanVar"}, default="MinVar"
            The objective of the portfolio optimization

    alpha: float, default=0
        The penalty coefficient of the return

    cov_matrix : {"empirical", "lw"}, default="lw"
        Specify the estimator of covariance matrix.
        If ``empirical``, it will be the empirical estimator. If ``lw``, it will be the LedoitWolf estimator.

    random_state : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, default=None
        The seed to initialize the parameter ``init_params`` in ``ScopeSolver``
    """

    def __init__(
        self,
        sparsity=5,
        obj="MinVar",
        alpha=0,
        cov_matrix="lw",
        random_state=None,
    ):
        self.sparsity = sparsity
        self.obj = obj
        self.alpha = alpha
        self.cov_matrix = cov_matrix
        self.random_state = random_state

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left")],
        "obj": [StrOptions({"MinVar", "MeanVar"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "cov_matrix": [StrOptions({"empirical", "lw"})],
        "random_state": ["random_state"],
    }

    # def _more_tags(self):
    #     return {'non_deterministic': True}

    def fit(self, X, y=None, sample_weight=None):
        r"""
        The fit function is used to comupte the weight of the desired sparse portfolio with a certain objective.

        Parameters
        ----------
        X : array-like of shape (n_periods, n_assets)
            Return data of n_assets assets spanning n_periods periods

        y : ignored
            Not used, present here for API consistency by convention.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        --------
        self : object
            Fitted Estimator.
        """

        self.random_state_ = check_random_state(self.random_state)
        X = check_array(X)
        T, N = X.shape

        # rng = np.random.default_rng(self.random_state)
        # init_params = rng.standard_normal(N)
        init_params = self.random_state_.randn(N)

        mu = np.mean(X, axis=0)
        if self.cov_matrix == "empirical":
            Sigma = np.cov(X.T)
        elif self.cov_matrix == "lw":
            Sigma = LedoitWolf().fit(X).covariance_
        else:
            raise ValueError("{} estimator is not supported.".format(self.cov_matrix))

        if self.obj == "MinVar":

            def custom_objective(params):
                params = params / jnp.sum(params)
                var = params @ Sigma @ params
                return var

        elif self.obj == "MeanVar":

            def custom_objective(params):
                params = params / jnp.sum(params)
                var = params @ Sigma @ params - self.alpha * mu @ params
                return var

        else:
            raise ValueError("{} objective is not supported.".format(self.obj))

        solver = ScopeSolver(N, self.sparsity)
        params = solver.solve(custom_objective, init_params=init_params, jit=True)
        self.weight = params / params.sum()
        self.coef_ = self.weight

        return self

    def score(self, X, y=None, sample_weight=None, measure="Sharpe"):
        r"""
        Give data, and it return the Sharpe ratio of the portfolio constructed with the weight ``self.coef_``

        Parameters
        -----------
        X : array-like of shape (n_periods, n_assets)
            Return data of n_assets assets spanning n_periods periods

        y : ignored
            Not used, present here for API consistency by convention.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        measure : {"Sharpe"}, default="Sharpe"
            The measure of the performance of a portfolio.

        Returns
        --------
        score : float
            The Sharpe ratio of the constructed portfolio.
        """
        X = check_array(X)
        return_ = X @ self.coef_
        if measure == "Sharpe":
            score = np.mean(return_) / np.std(return_)
        else:
            raise ValueError("{} measure is not supported.".format(measure))
        return score


class NonlinearSelection(BaseEstimator):
    r"""
    Select relevant features which may have nonlinear dependence on the target.

    Parameters
    -----------
    sparsity : int, default=5
        The number of features to be selected, i.e., the sparsity level.

    gamma_x : float, default=0.7
        The width parameter of Gaussian kernel for X.

    gamma_y : float, default=0.7
        The width parameter of Gaussian kernel for y.
    """

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left")],
        "gamma_x": [Interval(Real, 0, None, closed="neither")],
        "gamma_y": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        sparsity=5,
        gamma_x=0.7,
        gamma_y=0.7,
    ):
        self.sparsity = sparsity
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y

    def fit(
        self,
        X,
        y,
        sample_weight=None,
    ):
        r"""
        The fit function is used to comupte the coeffifient vector ``coef_`` and
        those features corresponding to larger coefficients are considered having
        stronger dependence on the target.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        --------
        self : object
            Fitted Estimator.
        """
        X, y = check_X_y(X, y)
        if sample_weight is None:
            pass
        else:
            sample_weight = np.array(sample_weight).reshape(-1)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("Dimension mismatch.")
        n, p = X.shape
        if p < self.sparsity:
            raise ValueError("invalid sparsity.")

        Gamma = np.eye(n) - np.ones((n, 1)) @ np.ones((1, n)) / n
        L = rbf_kernel(y.reshape(-1, 1), gamma=self.gamma_y)
        L_bar = Gamma @ L @ Gamma
        response = L_bar.reshape(-1)
        K_bar = np.zeros((n**2, p))
        for k in range(p):
            x = X[:, k]
            tmp = rbf_kernel(x.reshape(-1, 1), gamma=self.gamma_x)
            K_bar[:, k] = (Gamma @ tmp @ Gamma).reshape(-1)
        covariate = K_bar

        def custom_objective(alpha):
            loss = jnp.mean((response - covariate @ jnp.abs(alpha)) ** 2)
            return loss

        solver = ScopeSolver(p, sparsity=self.sparsity)
        alpha = solver.solve(custom_objective)
        self.coef_ = np.abs(alpha)
        return self

    def score(self, X, y, sample_weight=None):
        r"""
        Give test data, and it return the test score of this fitted model.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            Feature matrix.

        y : array-like, shape(n_samples,)
            Target values.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            The negative loss on the given data.
        """
        X, y = check_X_y(X, y)
        n, p = X.shape
        Gamma = np.eye(n) - np.ones((n, 1)) @ np.ones((1, n)) / n
        L = rbf_kernel(y.reshape(-1, 1), gamma=self.gamma_y)
        L_bar = Gamma @ L @ Gamma
        response = L_bar.reshape(-1)
        K_bar = np.zeros((n**2, p))
        for k in range(p):
            x = X[:, k]
            tmp = rbf_kernel(x.reshape(-1, 1), gamma=self.gamma_x)
            K_bar[:, k] = (Gamma @ tmp @ Gamma).reshape(-1)
        covariate = K_bar
        score = -np.mean((response - covariate @ self.coef_) ** 2)
        return score


class RobustRegression(BaseEstimator):
    r"""
    A robust regression procedure via sparsity constrained exponential loss minimization.
    Specifically, ``RobustRegression`` solves the following problem:
    :math:`\min_{\beta}-\sum_{i=1}^n\exp\{-(y_i-x_i^{\top}\beta)^2/\gamma\} \text{ s.t. } \|\beta\|_0 \leq s`
    where :math:`\gamma` is a hyperparameter controlling the degree of robustness and
    :math:`s` is a hyperparameter controlling the sparsity level of :math:`\beta`.

    Parameters
    -----------
    sparsity : int, default=5
        The number of features to be selected, i.e., the sparsity level.

    gamma : float, default=1
        The parameter controlling the degree of robustness for the estimator.
    """

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left")],
        "gamma": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, sparsity=5, gamma=1):
        self.sparsity = sparsity
        self.gamma = gamma

    def fit(self, X, y, sample_weight=None):
        r"""
        The fit function is used to comupte the coeffifient vector ``coef_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        --------
        self : object
            Fitted Estimator.
        """
        X, y = check_X_y(X, y)
        n, p = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = np.array(sample_weight)

        def custom_objective(params):
            err = -jnp.exp(-jnp.square(y - X @ params) / self.gamma)
            loss = jnp.average(err, weights=sample_weight)
            return loss

        solver = ScopeSolver(p, self.sparsity)
        self.coef_ = solver.solve(custom_objective, jit=True)

        return self

    def score(self, X, y, sample_weight=None):
        r"""
        Give test data, and it return the test score of this fitted model.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            Feature matrix.

        y : array-like, shape(n_samples,)
            Target values.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            The weighted exponential loss of the given data.
        """
        X, y = check_X_y(X, y)
        n, p = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = np.array(sample_weight)

        err = -np.exp(-np.square(y - X @ self.coef_) / self.gamma)
        loss = np.average(err, weights=sample_weight)
        score = -loss
        return score
