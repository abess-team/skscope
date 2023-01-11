from sklearn.base import BaseEstimator
import numpy as np

class BaseSolver(BaseEstimator):
    # attributes
    params = None
    value_objective = None

    def get_config(self, deep=True):
        return self.get_params(deep)

    def set_config(self, **params):
        return self.set_params(**params)

    def get_params(self):
        r""" 
        Get the solution of optimization, include the parameters and auxiliary parameters (if exists).
        """
        if self.aux_params_size is not None and self.aux_params_size > 0:
            return self.params, self.aux_params
        else:
            return self.params
    
    def get_support_set(self):
        r""" 
        Get the index of selected variables which is the non-zero parameters.
        """
        return np.nonzero(self.params)[0]

    def get_value_objective(self):
        r"""
        Get the value of the objective function on the solution.
        """
        return self.value_objective
