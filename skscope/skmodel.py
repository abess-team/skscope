import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from skscope import ScopeSolver
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import (
    check_array,
    check_random_state,
    check_X_y,
    check_is_fitted,
    check_consistent_length,
)
from sklearn.utils._param_validation import Hidden, Interval, StrOptions
from numbers import Integral, Real
from scipy import interpolate


def check_data(X, y=None, sample_weight=None):
    if y is None:
        X = check_array(X)
    else:
        X, y = check_X_y(X, y, dtype="numeric", y_numeric=True)
    n, p = X.shape

    if sample_weight is None:
        sample_weight = np.ones(n)
    else:
        sample_weight = np.array(sample_weight, dtype="float")
        if sample_weight.ndim > 1:
            raise ValueError("sample_weight should be a 1-D array.")
        if sample_weight.size != n:
            raise ValueError("X.shape[0] should be equal to sample_weight.size")
        if sample_weight.sum() == 0:
            raise ValueError("Weights sum to zero, can't be normalized.")

        useful_index = list()
        for i, w in enumerate(sample_weight):
            if w > 0:
                useful_index.append(i)
        if len(useful_index) < n:
            X = X[useful_index, :]
            if not (y is None):
                y = y[useful_index, :] if len(y.shape) > 1 else y[useful_index]
            sample_weight = sample_weight[useful_index]
            n = len(useful_index)

    return X, y, sample_weight


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
        sparsity=1,
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
        self._validate_params()
        X, y, sample_weight = check_data(X, y, sample_weight)
        T, N = X.shape
        self.n_features_in_ = N

        self.random_state_ = check_random_state(self.random_state)
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
        check_is_fitted(self)
        X, y, sample_weight = check_data(X, y, sample_weight)
        n, p = X.shape
        if p != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self.n_features_in_))

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
        sparsity=1,
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
        self._validate_params()
        X, y, sample_weight = check_data(X, y, sample_weight)
        n, p = X.shape
        self.n_features_in_ = p

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
        alpha = solver.solve(custom_objective, jit=True)
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
        check_is_fitted(self)
        X, y, sample_weight = check_data(X, y, sample_weight)
        n, p = X.shape
        if p != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self.n_features_in_))

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

    Note: When :math:`\gamma` is large, the exponential loss is approximately equivalent to :math:`|y_i-x_i^{\top}\beta|^2/\gamma` and thus similar to
    the least square estimator. When :math:`\gamma` is small, the sample :math:`i` with large error :math:`|y_i-x_i^{\top}\beta|`
    will obtain small impact on the estimation of :math:`\beta` and thus limiting the impact of outlier (i.e., improve the robustness
    but reduce the efficiency).
    Therefore, :math:`\gamma` need to be selected carefully with prior knowledge of the data or via some data-dirven methods (e.g. cross validation) to
    achieve a appropriate trade-off between robustness and efficiency of the resulting estimator.

    Parameters
    -----------
    sparsity : int, default=1
        The number of features to be selected, i.e., the sparsity level.

    gamma : float, default=1
        The parameter controlling the degree of robustness for the estimator.
    """

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left")],
        "gamma": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, sparsity=1, gamma=1):
        self.sparsity = sparsity
        self.gamma = gamma

    def _more_tags(self):
        return {"requires_y": False}

    def fit(self, X, y=None, sample_weight=None):
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
        self._validate_params()
        X, y, sample_weight = check_data(X, y, sample_weight)
        n, p = X.shape
        self.n_features_in_ = p

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
        check_is_fitted(self)
        X, y, sample_weight = check_data(X, y, sample_weight)
        n, p = X.shape
        if p != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self.n_features_in_))

        err = -np.exp(-np.square(y - X @ self.coef_) / self.gamma)
        loss = np.average(err, weights=sample_weight)
        score = -loss
        return score


class MultivariateFailure(BaseEstimator):
    r"""
    Multivariate failure time model.

    Parameters
    ----------
    sparsity : int, default=5
        The number of features to be selected, i.e., the sparsity level.
    """

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, sparsity=5):
        self.sparsity = sparsity

    def fit(self, X, y, delta, sample_weight=None):
        r"""
        Minimize negative partial log-likelihood with sparsity constraint for provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : array-like, shape = (n_samples, n_events)
            Observed time of multiple events.

        delta : array-like, shape = (n_samples, n_events)
            Indicator matrix of censoring.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        --------
        self : object
            Fitted Estimator.
        """
        self._validate_params()
        n, p = X.shape
        K = delta.shape[1]
        self.n_features_in_ = p
        self.n_events = K

        def multivariate_failure_objective(params):
            Xbeta_expanded = jnp.matmul(X, params)[:, None]
            sum_exp_Xbeta = logsumexp(
                Xbeta_expanded + jnp.log(y >= y[:, None, :]), axis=1
            )
            loss = -jnp.mean((Xbeta_expanded - sum_exp_Xbeta) * delta)
            return loss

        solver = ScopeSolver(p, self.sparsity)
        self.coef_ = solver.solve(multivariate_failure_objective, jit=True)
        return self

    def predict(self, X):
        r"""
        Given the features, predict the hazard function up to some constant independent of the sample.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            Feature matrix.

        Returns
        --------
        hazard : array, shape = (n_samples,)
            the quantity :math:`e^{\beta^{\top}X_i}` proportional to the harzard function up to
            some constant independent of the sample index :math:`i` such that
            :math:`\lambda_k(t;X_{i})=\lambda_{0k}(t)e^{\beta^{\top}X_i}`.
        """
        check_is_fitted(self)
        X, _, _ = check_data(X)
        Xbeta = X @ self.coef_
        return np.exp(Xbeta)

    def score(self, X, y, delta, sample_weight=None):
        r"""
        Give test data, and it return the test score of this fitted model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : array-like, shape = (n_samples, n_events)
            Observed time of multiple events.

        delta : array-like, shape = (n_samples, n_events)
            Indicator matrix of censoring.

        sample_weight : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            The log likelihood of the given data.
        """
        check_is_fitted(self)
        n, p = X.shape
        K = delta.shape[1]
        if p != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self.n_features_in_))
        if K != self.n_events:
            raise ValueError(
                "y.shape[1] and delta.shape[1] should be " + str(self.events)
            )

        Xbeta = np.matmul(X, self.coef_)
        tmp = np.ones((n, K))
        for k in range(K):
            tmp[:, k] = X @ self.coef_ - np.log(
                np.matmul(
                    y[:, k].reshape(1, -1) >= y[:, k].reshape(-1, 1), np.exp(Xbeta)
                )
            )
        score = np.mean(tmp * delta)
        return score

class IsotonicRegression(BaseEstimator):
    r"""
    Isotonic regression. 

    Parameters
    -----------
    sparsity : int, default=5
        The number of features to be selected, i.e., the sparsity level.
    """

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self, 
        sparsity=5,
    ):
        self.sparsity = sparsity

    def _check_input_data_shape(self, X):
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = (
                "Isotonic regression input X should be a 1d array or "
                "2d array with 1 feature"
            )
            raise ValueError(msg)

    def fit(
        self, 
        X, 
        y,
        sample_weight=None,
    ):  
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        """
        self._validate_params()
        # check_params = dict(accept_sparse=False, ensure_2d=False)
        # X = check_array(
        #     X, input_name="X", dtype=[np.float64, np.float32], **check_params
        # )
        # y = check_array(y, input_name="y", dtype=X.dtype, **check_params)
        # check_consistent_length(X, y, sample_weight)

        X = check_array(X, ensure_2d=False)
        X = X.reshape(-1)
        n = len(y)

        def isotonic_loss(params):
            return jnp.sum(jnp.square(y - jnp.cumsum(jnp.abs(params))))
        solver = ScopeSolver(n, sparsity=self.sparsity)
        self.params = solver.solve(isotonic_loss)
        y_pred = np.cumsum(np.abs(self.params))
        self.f_ = interpolate.interp1d(
                X, y_pred, kind="linear"
            )
        return self


    def transform(
        self, 
        X,
    ):
        """Transform new data by linear interpolation.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The transformed data.
        """
        X = check_array(X, ensure_2d=False)
        self._check_input_data_shape(X)
        X = X.reshape(-1)
        y_pred = self.f_(X)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.
        """
        check_is_fitted(self)
        X, y, sample_weight = check_data(X, y, sample_weight)
        n, p = X.shape
        if p != self.n_features_in_:
            raise ValueError("X.shape[1] should be " + str(self.n_features_in_))
        
        y_pred = self.predict(X)
        score = r2_score(y, y_pred, sample_weight=sample_weight)
        return score

    def predict(self, X):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        y_pred = self.transform(X)
        return y_pred