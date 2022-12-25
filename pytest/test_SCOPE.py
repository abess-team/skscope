from abess import ConvexSparseSolver, make_multivariate_glm_data, make_glm_data
import pytest
import numpy as np
from utilities import assert_fit, assert_value
import importlib



class TestConvexSparseSolver:
    """
    Test for ConvexSparseSolver
    """

    @staticmethod
    def test_linear_model_jax():
        jnp = importlib.import_module("jax.numpy")
        np.random.seed(1)
        n = 30
        p = 5
        k = 3
        family = "gaussian"

        data = make_glm_data(family=family, n=n, p=p, k=k)

        model = ConvexSparseSolver(model_size=p, support_size=k)

        def loss(para, aux_para, data):
            return jnp.sum(
                jnp.square(data.y - data.x @ para)
            )

        model.set_loss_jax(loss)

        model.fit(data)

        assert_value(model.coef_, data.coef_)
    
    @staticmethod
    def test_linear_model_custom():
        np.random.seed(1)
        n = 30
        p = 5
        k = 3
        family = "gaussian"

        data = make_glm_data(family=family, n=n, p=p, k=k)

        model = ConvexSparseSolver(model_size=p, support_size=k)

        def loss(para, aux_para, data):
            return np.sum(
                np.square(data.y - data.x @ para)
            )
        def grad(para, aux_para, data, compute_para_index):
            return -2 * data.x[:,compute_para_index].T @ (data.y - data.x @ para)
        def hess(para, aux_para, data, compute_para_index):
            return 2 * data.x[:,compute_para_index].T @ data.x[:,compute_para_index]

        model.set_loss_custom(loss=loss, gradient=grad, hessian=hess)

        model.fit(data)

        assert_value(model.coef_, data.coef_)
    
    @staticmethod
    def test_linear_model_cv():
        jnp = importlib.import_module("jax.numpy")
        np.random.seed(1)
        n = 30
        p = 5
        k = 3
        family = "gaussian"

        data = make_glm_data(family=family, n=n, p=p, k=k)
        class Data:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        model = ConvexSparseSolver(model_size=p, sample_size=n, cv=3)


        def loss(para, aux_para, data):
            return jnp.sum(
                jnp.square(data.y - data.x @ para)
            )
        model.set_loss_jax(loss)

        model.set_split_method(lambda data, index:  Data(data.x[index, :], data.y[index]))

        model.fit(Data(data.x, data.y))

        assert_value(model.coef_, data.coef_)
    
    @staticmethod
    def test_linear_model_multi():
        jnp = importlib.import_module("jax.numpy")
        np.random.seed(1)
        n = 30
        p = 5
        k = 3
        family = "multigaussian"
        M = 3
        data = make_multivariate_glm_data(family=family, n=n, p=p, k=k, M=M)
        group = [i for i in range(p) for j in range(M)]

        model = ConvexSparseSolver(
            model_size=p * M, sample_size=n, aux_para_size=M, group=group
        )

        def f(para, aux_para, data):
            m = jnp.size(aux_para)
            p = data[0].shape[1]
            return jnp.sum(
                jnp.square(data[1] - data[0] @ para.reshape(p, m) - aux_para)
            )
        model.set_loss_jax(f)

        model.fit((jnp.array(data.x), jnp.array(data.y)))
        
        assert_fit(model.coef_, [c for v in data.coef_ for c in v])
