from sklearn.datasets import make_regression
from jax import numpy as jnp
import numpy as np
from scope import ScopeSolver


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

        data = {"X": X, "Y": Y}

        def linear_model_data(params, data):
            return jnp.sum(jnp.square(data["Y"] - data["X"] @ params))

        def split_method(data, index):
            return {
                "X": data["X"][
                    index,
                ],
                "Y": data["Y"][index],
            }

        return {
            "n_samples": self.N,
            "n_features": self.P,
            "n_informative": self.K,
            "params": true_params,
            "support_set": np.nonzero(true_params)[0],
            "loss": linear_model,
            "loss_data": linear_model_data,
            "data": data,
            "split_method": split_method,
            "loss_numpy": linear_model_numpy,
            "grad": grad_linear_model,
            "hess": hess_linear_model,
            "cpp_model": ScopeSolver.quadratic_objective(
                np.matmul(X.T, X), -np.matmul(X.T, Y)
            ),
        }
    
    def create_easy_linear_model(self):
        X_part, Y, true_params_part = make_regression(
            self.N, self.K, n_informative=self.K, coef=True, random_state=self.seed
        )
        # create a matrix with the first K columns of X_part and the rest of the columns are 0
        X = np.concatenate((X_part, np.zeros((self.N, self.P - self.K))), axis=1)
        true_params = np.concatenate((true_params_part, np.zeros(self.P - self.K)))

        def linear_model(params):
            return jnp.sum(jnp.square(Y - jnp.matmul(X, params))) / 2

        def linear_model_numpy(params):
            return np.sum(np.square(Y - np.matmul(X, params))) / 2

        def grad_linear_model(params):
            return -np.matmul(X.T, (Y - np.matmul(X, params)))

        def hess_linear_model(params):
            return np.matmul(X.T, X)

        data = {"X": X, "Y": Y}

        def linear_model_data(params, data):
            return jnp.sum(jnp.square(data["Y"] - data["X"] @ params))

        def split_method(data, index):
            return {
                "X": data["X"][
                    index,
                ],
                "Y": data["Y"][index],
            }

        return {
            "n_samples": self.N,
            "n_features": self.P,
            "n_informative": self.K,
            "params": true_params,
            "support_set": np.nonzero(true_params)[0],
            "loss": linear_model,
            "loss_data": linear_model_data,
            "data": data,
            "split_method": split_method,
            "loss_numpy": linear_model_numpy,
            "grad": grad_linear_model,
            "hess": hess_linear_model,
            "cpp_model": ScopeSolver.quadratic_objective(
                np.matmul(X.T, X), -np.matmul(X.T, Y)
            ),
        }

