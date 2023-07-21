import numpy as np
import jax.numpy as jnp
from skscope import ScopeSolver
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf

class Portfolio(BaseEstimator):
    def __init__(self, k=None, seed=None):
        r"""
        k : int, default=None
            the number of stocks to be chosen, i.e., the sparsity level
        """
        self.k = k
        self.seed = seed
        

    def fit(self, X, y=None, sample_weight=None, obj="MinVar", lambda_=None, hist=False):
        r"""

        Parameters
        ----------
        X : array-like of shape (n_periods, n_assets)
            return data of n_assets assets spanning n_periods periods

        y : Ignored
            Not used, present here for API consistency by convention.
        
        sample_weight : Ignored

        obj : {"MinVar", "MeanVar"}, default="MinVar"
            the objective of the portfolio optimization
        
        lambda_ : float, 

        hist : bool, default=False
            whether use the sample estimator,
            if false, the covariance matrix will be the LedoitWolf estimator    
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
        return_ = X @ self.coef_
        if measure == "Sharpe":
            score = np.mean(return_) / np.std(return_)
        else:
            raise ValueError("{} measure is not supported.".format(measure))
        return score



