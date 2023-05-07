from . import _scope
import numpy as np


def quadratic_objective(Q, p, autodiff=False):
    """
    Create a model of quadratic objective function which is $L(x) = <x, Qx> / 2 + <p, x>$.

    Parameters
    ----------
    + Q : array-like, shape (n_features, n_features)
        The matrix of quadratic term.
    + p : array-like, shape (n_features,)
        The vector of linear term.
    + autodiff : bool, default False
        Whether to return a overloaded function of objective function for cpp library `autodiff`.

    Returns
    -------
    A dict of quadratic model for `solve()`, which may contain part of the following keys:
    + objective : function('params': array, 'data': Any) ->  float
        The objective function.
    + gradient : function('params': array, 'data': Any) -> 1-D array
        The gradient of objective function.
    + hessian : function('params': array, 'data': Any) -> 2-D array
        The hessian of objective function.
    + data : Any
        The data for above functions.

    Examples
    --------
        ```
        import numpy as np
        from scope import ScopeSolver, GraspSolver
        from scope.model import quadratic_objective

        model = quadratic_objective(np.eye(5), np.ones(5))
        solver1 = ScopeSolver(dimensionality=5)
        solver1.solve(
            model["objective"],
            model["data"],
            gradient=model["gradient"],
            hessian=model["hessian"],
        )
        print(solver1.get_result())

        solver2 = GraspSolver(dimensionality=5)
        solver2.solve(
            model['objective'],
            model['data'],
            gradient = model['gradient'],
        )
        print(solver2.get_result())
        ```
    """
    Q = np.array(Q, dtype=float)
    p = np.array(p, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if p.ndim != 1 or p.shape[0] != Q.shape[0]:
        raise ValueError("p must be a vector with length of Q.shape[0].")

    if autodiff:
        return {
            "objective": _scope.quadratic_loss,
            "data": _scope.QuadraticData(Q, p),
        }
    else:
        return {
            "objective": lambda x, d: np.array(_scope.quadratic_loss(x, d)),
            "gradient": lambda x, d: np.array(_scope.quadratic_grad(x, d)),
            "hessian": lambda x, d: np.array(_scope.quadratic_hess(x, d)),
            "data": _scope.QuadraticData(Q, p),
        }

