from sklearn.datasets import make_regression
from jax import numpy as jnp

class CreateTestModel:
    def __init__(self, N=20, P=5, K=3, seed=1):
        self.N = N
        self.P = P
        self.K = K
        self.seed = seed

    def create_linear_model(self):
        X, Y, true_params = make_regression(
            self.N, self.P, n_informative=self.K, coef=True, random_state=self.seed
        )

        def linear_model(params):
            return jnp.sum(jnp.square(Y - X @ params))
        
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
            "loss_jit": None,
            "grad_jit": None,
            "hess_jit": None,
        }
