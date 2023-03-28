from . import _scope
import numpy as np

def quadratic_objective(Q, p, hessian=False, autodiff=False):
    """
    Create a model of quadratic objective function which is $L(x) = <x, Qx> / 2 + <p, x>$.

    Parameters
    ----------
    + Q : array-like, shape (n_features, n_features)
        The matrix of quadratic term.
    + p : array-like, shape (n_features,)
        The vector of linear term.
    + hessian : bool, default False
        Whether to return the hessian of objective function.
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
        from scope import ScopeSolver, GraspSolver, quadratic_objective

        solver = ScopeSolver(dimensionality=5)
        solver.solve(**quadratic_objective(np.eye(5), np.ones(5), hessian=True)) 
        print(solver.get_result())

        solver = GraspSolver(dimensionality=5)
        solver.solve(**quadratic_objective(np.eye(5), np.ones(5), hessian=False))
        print(solver.get_result())
        ```
    """
    Q = np.array(Q, dtype=float)
    p = np.array(p, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if p.ndim != 1 or p.shape[0] != Q.shape[0]:
        raise ValueError("p must be a vector with length of Q.shape[0].")

    if hessian and not autodiff:
        return {
            "objective": lambda x, d: np.array(_scope.quadratic_loss(x, d)),
            "gradient": lambda x, d: np.array(_scope.quadratic_grad(x, d)),
            "hessian": lambda x, d: np.array(_scope.quadratic_hess(x, d)),
            "data": _scope.QuadraticData(Q, p),
        }
    elif not hessian and not autodiff:
        return {
            "objective": lambda x, d: np.array(_scope.quadratic_loss(x, d)),
            "gradient": lambda x, d: np.array(_scope.quadratic_grad(x, d)),
            "data": _scope.QuadraticData(Q, p),
        }
    elif not hessian and autodiff:
        return {
            "objective": _scope.quadratic_loss,
            "data": _scope.QuadraticData(Q, p),
            "cpp": True
        }
    else:
        raise ValueError("hessian and autodiff cannot be both True.")