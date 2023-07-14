import numpy as np
import math
import nlopt


def convex_solver_nlopt(
    loss_fn,
    value_and_grad,
    init_params,
    optim_variable_set,
    data,
):
    """
    A wrapper of ``nlopt`` solver for convex optimization.

    Parameters
    ----------
    loss_fn: callable
        The objective function.
        ``loss_fn(params, data) -> loss``, where ``params`` is a 1-D array with shape (dimensionality,).
    value_and_grad: callable
        The function to compute the loss and gradient.
        ``value_and_grad(params, data) -> (loss, grad)``, where ``params`` is a 1-D array with shape (dimensionality,).
    init_params: array of shape (dimensionality,)
        The initial value of the parameters to be optimized.
    optim_variable_set: array of int
        The index of variables to be optimized, others are fixed to the initial value.
    data: 
        The data passed to loss_fn and value_and_grad.

    Returns
    -------
    loss: float
        The loss of the optimized parameters, i.e., `loss_fn(params, data)`.
    optimized_params: array of shape (dimensionality,)
        The optimized parameters.
    """
    best_loss = math.inf
    best_params = None

    def cache_opt_fn(x, grad):
        nonlocal best_loss, best_params
        init_params[optim_variable_set] = x  # update the nonlocal variable: params
        if grad.size > 0:
            loss, full_grad = value_and_grad(init_params, data)
            grad[:] = full_grad[optim_variable_set]
        else:
            loss = loss_fn(init_params, data)
        if loss < best_loss:
            best_loss = loss
            best_params = np.copy(x)
        return loss

    nlopt_solver = nlopt.opt(nlopt.LD_LBFGS, optim_variable_set.size)
    nlopt_solver.set_min_objective(cache_opt_fn)

    try:
        init_params[optim_variable_set] = nlopt_solver.optimize(init_params[optim_variable_set])
        return nlopt_solver.last_optimum_value(), init_params
    except RuntimeError:
        init_params[optim_variable_set] = best_params
        return best_loss, init_params
