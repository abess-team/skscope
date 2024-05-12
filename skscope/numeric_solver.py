#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Zezhi Wang
# Copyright (C) 2023 abess-team
# Licensed under the MIT License.

import numpy as np
import math
from scipy.optimize import minimize


def convex_solver_BFGS(
    objective_func,
    value_and_grad,
    init_params,
    optim_variable_set,
    data,
):
    def fun(x):
        init_params[optim_variable_set] = x
        return objective_func(init_params, data)

    def jac(x):
        init_params[optim_variable_set] = x
        _, grad = value_and_grad(init_params, data)
        return grad[optim_variable_set]

    res = minimize(fun, init_params[optim_variable_set], method="BFGS", jac=jac)
    init_params[optim_variable_set] = res.x
    return res.fun, init_params


def convex_solver_LBFGS(
    objective_func,
    value_and_grad,
    init_params,
    optim_variable_set,
    data,
):
    def fun(x):
        init_params[optim_variable_set] = x
        return objective_func(init_params, data)

    def jac(x):
        init_params[optim_variable_set] = x
        _, grad = value_and_grad(init_params, data)
        return grad[optim_variable_set]

    res = minimize(fun, init_params[optim_variable_set], method="L-BFGS-B", jac=jac)
    init_params[optim_variable_set] = res.x
    return res.fun, init_params
