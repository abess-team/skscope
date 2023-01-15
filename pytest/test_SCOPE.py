from scope import ScopeSolver
import pytest
from sklearn.datasets import make_regression
from jax import numpy as jnp
import nlopt

N, P, K = 10, 5, 3
X, Y, TRUE_PARAMS = make_regression(N, P, n_informative = K,coef=True, random_state=1)

def linear_model(params):
    return jnp.sum(
        jnp.square(Y - X @ params)
    )

class TestScopeSolver:
    """
    Test for ScopeSolver
    """
    @staticmethod
    def test_jax():
        solver = ScopeSolver(P, K)
        solver.solve(linear_model)
        params = solver.get_params() 
        
        assert TRUE_PARAMS == pytest.approx(params, rel=0.01, abs=0.01)
    
    @staticmethod
    def test_jit():
        pass

    @staticmethod
    def test_custom_py():
        pass

    @staticmethod
    def test_custom_cpp():
        pass

    @staticmethod
    def test_autodiff():
        pass

    @staticmethod
    def test_init_params():
        solver = ScopeSolver(P, K)
        solver.solve(linear_model, init_params = TRUE_PARAMS *2)
        params = solver.get_params() 
        
        assert TRUE_PARAMS == pytest.approx(params, rel=0.01, abs=0.01)
    
    @staticmethod
    def test_nlopt_solver():
        solver = ScopeSolver(P, K, nlopt_solver=nlopt.opt(nlopt.LD_SLSQP, 10))
        solver.solve(linear_model)
        params = solver.get_params() 
        
        assert TRUE_PARAMS == pytest.approx(params, rel=0.01, abs=0.01)
