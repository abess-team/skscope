import numpy as np
import math
import nlopt


def convex_solver_nlopt(
    loss_fn,
    value_and_grad,
    params,
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
    params: Sequence[float]
        The complete initial parameters.
    optim_variable_set: Sequence[int]
        The index of variables to be optimized.
    data: Any
        The data passed to loss_fn and value_and_grad.

    Returns
    -------
    loss: float
        The loss of the optimized parameters, i.e., `loss_fn(params, data)`.
    optimized_params: Sequence[float]
        The optimized parameters.
    """
    best_loss = math.inf
    best_params = None

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
        return nlopt_solver.last_optimum_value(), params
    except RuntimeError:
        params[optim_variable_set] = best_params
        return best_loss, params
