import numpy as np
import jax.numpy as jnp
from skscope import ScopeSolver
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import rbf_kernel


class PortfolioSelection(BaseEstimator):
    r"""
    Construct a sparse portfolio using ``skscope`` with ``MinVar`` or ``MeanVar`` measure.

    Parameters
    ------------
    s : int, default=10
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
        s=10,
        obj="MinVar",
        alpha=0,
        cov_matrix="lw",
        random_state=None,
    ):
        self.s = s
        self.obj = obj
        self.alpha = alpha
        self.cov_matrix = cov_matrix
        self.random_state = random_state

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
        X = np.array(X)
        T, N = X.shape

        rng = np.random.default_rng(self.random_state)
        init_params = rng.standard_normal(N)

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

        solver = ScopeSolver(N, self.s)
        params = solver.solve(custom_objective, init_params=init_params, jit=True)
        self.weight = params / params.sum()
        self.coef_ = self.weight

        return self

    def score(self, X, y=None, sample_weight=None, measure="Sharpe"):
        r"""
        Give date, and it return the Sharpe ratio of the portfolio constructed with the weight ``self.coef_``

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
        # X, y, sample_weight = new_data_check(self, X, y, sample_weight)
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
    ----------
    sparsity : int, default=None
        The number of features to be selected, i.e., the sparsity level
    gamma_x : float, default=None
        The width parameter of Gaussian kernel for X. If None, defaults to 1.0 / n_features.
    gamma_y : float, default=None
        The width parameter of Gaussian kernel for y.
    """

    def __init__(self, sparsity=None, gamma_x=0.7, gamma_y=0.7):
        self.sparsity = sparsity
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y

    def fit(self, X, y, sample_weight=None):
        r"""
        The fit function is used to comupte the weight of the desired sparse portfolio with a certain objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : ignored
            Not used, present here for API consistency by convention.
        """
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

        def custom_objective(alpha):
            loss = jnp.mean((response - covariate @ jnp.abs(alpha)) ** 2)
            return loss

        solver = ScopeSolver(p, sparsity=self.sparsity)
        alpha = solver.solve(custom_objective, jit=True)
        self.coef_ = alpha
        return self

    def score(self, X, y=None, sample_weight=None):
        """
        ignore.
        ``score`` is not used in ``NonlinearSelection``.
        """
        print( "score is not used in NonlinearSelection.")
