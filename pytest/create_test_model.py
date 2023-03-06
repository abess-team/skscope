from sklearn.datasets import make_regression
from jax import numpy as jnp
import numpy as np

class CreateTestModel:
    def __init__(self, N=100, P=5, K=3, seed=1):
        self.N = N
        self.P = P
        self.K = K
        self.seed = seed

    def create_linear_model(self):
        X, Y, true_params = make_regression(
            self.N, self.P, n_informative=self.K, coef=True, random_state=self.seed
        )

        def linear_model(params):
            return jnp.sum(jnp.square(Y - jnp.matmul(X, params)))
        def linear_model_numpy(params):
            return np.sum(np.square(Y - np.matmul(X, params)))
        def grad_linear_model(params):
            return -2 * np.matmul(X.T, (Y - np.matmul(X, params)))
        def hess_linear_model(params):
            return 2 * np.matmul(X.T, X)
        
        data = {"X" : X, "Y" : Y}

        def linear_model_data(params, data):
            return jnp.sum(jnp.square(data["Y"] - data["X"] @ params))

        def split_method(data, index):
            return {"X" : data["X"][index,], "Y" : data["Y"][index]}

        return {
            "n_samples": self.N,
            "n_features": self.P,
            "n_informative": self.K,
            "params": true_params,
            "loss": linear_model,
            "loss_data": linear_model_data,
            "data": data,
            "split_method": split_method,
            "loss_numpy": linear_model_numpy,
            "grad": grad_linear_model,
            "hess": hess_linear_model,
        }
