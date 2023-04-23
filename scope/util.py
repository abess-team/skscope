from . import _scope
import numpy as np
import math
import nlopt


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
            "cpp": True,
        }
    else:
        raise ValueError("hessian and autodiff cannot be both True.")


def convex_solver_nlopt(
    loss_fn,
    value_and_grad,
    init_params,
    optim_variable_set,
    data,
):
    """
    A wrapper of nlopt solver for convex optimization.

    Nlopt often throws RuntimeError even if the optimization is nearly successful.
    This function is used to cache the best result and return it.

    Parameters
    ----------
    loss_fn: Callable[[Sequence[float], Any], float]
        The loss function.
    value_and_grad: Callable[[Sequence[float], Any], Tuple[float, Sequence[float]]]
        The function to compute the loss and gradient.
    init_params: Sequence[float]
        The complete initial parameters.
    optim_variable_set: Sequence[int]
        The index of variables to be optimized.
    data: Any
        The data passed to loss_fn and value_and_grad.

    Returns
    -------
    optim_params: Sequence[float]
        The optimized parameters, which is same as init_params except for the optimized variables.
    loss: float
        The loss of the optimized parameters, i.e., `loss_fn(optim_params, data)`.
    """
    best_loss = math.inf
    best_params = None
    params = np.copy(init_params)

    def cache_opt_fn(x, grad):
        nonlocal best_loss, best_params
        params[optim_variable_set] = x  # update the nonlocal variable: params
        if grad.size > 0:
            loss, full_grad = value_and_grad(params, data)
            grad[:] = full_grad[optim_variable_set]
        else:
            loss = loss_fn(params, data)
        if loss < best_loss:
            best_loss = loss
            best_params = np.copy(x)
        return loss

    nlopt_solver = nlopt.opt(nlopt.LD_LBFGS, optim_variable_set.size)
    nlopt_solver.set_min_objective(cache_opt_fn)

    try:
        params[optim_variable_set] = nlopt_solver.optimize(params[optim_variable_set])
        return params, nlopt_solver.last_optimum_value()
    except RuntimeError:
        params[optim_variable_set] = best_params
        return params, best_loss
