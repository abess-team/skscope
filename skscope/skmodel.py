import numpy as np
import jax.numpy as jnp
from skscope import ScopeSolver
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf


class PortfolioSelection(BaseEstimator):
    r"""
    Construct a sparse portfolio using ``skscope`` with ``MinVar`` or ``MeanVar`` measure (see ``fit`` later)

    Parameters
    ------------
    k : int, default=None
        The number of stocks to be chosen, i.e., the sparsity level
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, default=None
        The seed to initialize the parameter ``init_params`` in ``ScopeSolver``
    """
    def __init__(self, k=None, seed=None):
        self.k = k
        self.seed = seed

    def fit(self, X, y=None, sample_weight=None, obj="MinVar", lambda_=None, hist=False):
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

        obj : {"MinVar", "MeanVar"}, default="MinVar"
            The objective of the portfolio optimization
        
        lambda_ : float
            The penalty coefficient of the return

        hist : bool, default=False
            Whether use the sample estimator,
            if false, the covariance matrix will be the LedoitWolf estimator    
        
        Returns
        --------
        self : object
            Fitted Estimator.
        """
        X = np.array(X)
        T, N = X.shape
        
        rng = np.random.default_rng(self.seed)
        init_params = rng.standard_normal(N)

        mu = np.mean(X, axis=0)
        if hist:
            Sigma = np.cov(X.T)
        else:
            Sigma = LedoitWolf().fit(X).covariance_

        if obj == "MinVar":
            def custom_objective(params):
                params = params / jnp.sum(params)
                var = params @ Sigma @ params
                return var
        elif obj == "MeanVar":
            def custom_objective(params):
                params = params / jnp.sum(params)
                var = params @ Sigma @ params - lambda_ * mu @ params
                return var
        else:
            raise ValueError("{} objective is not supported.".format(obj))

        solver = ScopeSolver(N, self.k)
        params = solver.solve(custom_objective, init_params=init_params)
        self.weight =  params / params.sum()
        self.coef_ = self.weight

        return self

    def predict(self):
        pass

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
        return_ = X @ self.coef_
        if measure == "Sharpe":
            score = np.mean(return_) / np.std(return_)
        else:
            raise ValueError("{} measure is not supported.".format(measure))
        return score