:parenttoc: True

PredefindModel
======================

Quadraic objective function
---------------------------

Quadratic objective function is :math:`L(x) = <x, Qx> / 2 + <p, x>`, which is often used in linear regression.

Here is an example of using quadratic objective function in linear regression.

.. code-block:: python
    
    import numpy as np
    from scope import ScopeSolver, GraspSolver
    from scope.model import quadratic_objective
    from sklearn.datasets import make_regression

    ## generate data
    n, p, k= 10, 5, 3
    X, y, true_params = make_regression(n_samples=n, n_features=p, n_informative=k, coef=True)

    # scope solver need hessian information
    solver1 = ScopeSolver(dimensionality=5)
    solver1.solve(**quadratic_objective(
                    np.matmul(X.T, X) / n,
                    -np.matmul(X.T, y) / n,
                    hessian=True
                ))
    print(solver1.get_result())

    # grasp solver does not need hessian information
    solver2 = GraspSolver(dimensionality=5)
    solver2.solve(**quadratic_objective(
                    np.matmul(X.T, X) / n,
                    -np.matmul(X.T, y) / n,
                    hessian=False
                ))
    print(solver2.get_result())