from sklearn.datasets import make_regression
from jax import numpy as jnp
import numpy as np
import scope


class CreateTestModel:
    def __init__(self, N=100, P=5, K=2, seed=1):
        self.N = N
        self.P = P
        self.K = K
        self.seed = seed

    def create_linear_model(self):
        X, Y, true_params = make_regression(
            self.N, self.P, n_informative=self.K, coef=True, random_state=self.seed
        )

        def linear_model(params):
            return jnp.sum(jnp.square(Y - jnp.matmul(X, params))) / 2

        def linear_model_numpy(params):
            return np.sum(np.square(Y - np.matmul(X, params))) / 2

        def grad_linear_model(params):
            return -np.matmul(X.T, (Y - np.matmul(X, params)))

        def hess_linear_model(params):
            return np.matmul(X.T, X)
        
        X_jnp = jnp.array(X)
        Y_jnp = jnp.array(Y)
        def linear_model_data(params, data_indices):
            return jnp.sum(jnp.square(Y_jnp[data_indices] - X_jnp[data_indices,] @ params))

        return {
            "n_samples": self.N,
            "n_features": self.P,
            "n_informative": self.K,
            "params": true_params,
            "support_set": np.nonzero(true_params)[0],
            "loss": linear_model,
            "loss_data": linear_model_data,
            "loss_numpy": linear_model_numpy,
            "grad": grad_linear_model,
            "hess": hess_linear_model,
            "data": (X, Y),
        }

