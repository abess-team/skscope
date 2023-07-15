from sklearn.datasets import make_regression
## generate data
n, p, s = 100, 10, 3
X, y, true_coefs = make_regression(n_samples=n, n_features=p, n_informative=s, coef=True, random_state=0) 
print("X shape:", X.shape)
print("Y shape:", y.shape)
import jax.numpy as jnp
def objective_function(coefs):
    return jnp.linalg.norm(y - X @ coefs)
from skscope import ScopeSolver
scope_solver = ScopeSolver(
    dimensionality=p,  ## there are p parameters
    sparsity=s,        ## we want to select s variables
)
scope_solver.solve(objective_function)    
import numpy as np
est_support_set = scope_solver.get_support()
print("Estimated effective predictors:", est_support_set)
print("True effective predictors:", np.nonzero(true_coefs)[0])    
est_coefs = scope_solver.get_estimated_params()
print("Estimated coefficient:", np.around(est_coefs, 2))
print("True coefficient:", np.around(true_coefs, 2))