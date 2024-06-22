#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Zezhi Wang
# Copyright (C) 2023 abess-team
# Licensed under the MIT License.

import numpy as np
import pytest

from create_test_model import CreateTestModel
from skscope import (
    utilities,
    ScopeSolver,
    BaseSolver,
    HTPSolver,
    GraspSolver,
    IHTSolver,
)
import skscope._scope as _scope

model_creator = CreateTestModel()
linear = model_creator.create_linear_model()

models = (linear,)
models_ids = ("linear",)
solvers = (ScopeSolver, BaseSolver)  # , HTPSolver, GraspSolver, IHTSolver)
solvers_ids = ("scope", "Base")  # , "GraHTP", "GraSP", "IHT")


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_numeric_solver(model, solver_creator):
    from skscope.numeric_solver import convex_solver_BFGS

    solver = solver_creator(
        model["n_features"], model["n_informative"], numeric_solver=convex_solver_BFGS
    )
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.get_support())


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_init_support_set(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.solve(model["loss"], init_support_set=[0, 1, 2], jit=True)

    assert set(model["support_set"]) == set(solver.get_support())


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_init_params(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.solve(model["loss"], init_params=np.ones(model["n_features"]), jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_ic(model, solver_creator):
    solver = solver_creator(
        model["n_features"],
        [0, model["n_informative"]],
        sample_size=model["n_samples"],
        ic_method=utilities.LinearSIC,
    )
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_cv_random_split(model, solver_creator):
    solver = solver_creator(
        model["n_features"],
        [0, model["n_informative"]],
        model["n_samples"],
        cv=2,
        split_method=lambda data, indeices: (data[0][indeices], data[1][indeices]),
    )
    solver.solve(model["loss_data"], data=model["data"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_cv_given_split(model, solver_creator):
    n_fold = 2
    cv_fold_id = [i for i in range(n_fold)] * (model["n_samples"] // n_fold) + [
        i for i in range(model["n_samples"] % n_fold)
    ]
    solver = solver_creator(
        model["n_features"],
        [0, model["n_informative"]],
        model["n_samples"],
        cv=n_fold,
        cv_fold_id=cv_fold_id,
        split_method=lambda data, indeices: (data[0][indeices], data[1][indeices]),
    )
    solver.solve(model["loss_data"], data=model["data"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_no_autodiff(model, solver_creator):
    """
    Test that the user can provide the gradient and hessian
    """
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.use_hessian = True
    solver.hessian = model["hess"]
    solver.solve(model["loss_numpy"], gradient=model["grad"])

    assert set(model["support_set"]) == set(solver.support_set)


def test_scope_cpp():
    solver = ScopeSolver(linear["n_features"], linear["n_informative"])
    X, Y = linear["data"]
    solver.use_hessian = True
    solver.hessian = lambda x, d: np.array(_scope.quadratic_hess(x, d))
    solver.solve(
        lambda x, d: np.array(_scope.quadratic_loss(x, d)),
        data=_scope.QuadraticData(np.matmul(X.T, X), -np.matmul(X.T, Y)),
        gradient=lambda x, d: np.array(_scope.quadratic_grad(x, d)),
    )

    assert set(solver.support_set) == set(linear["support_set"])


def test_scope_autodiff():
    solver = ScopeSolver(linear["n_features"], linear["n_informative"])
    X, Y = linear["data"]
    solver.use_hessian = True
    solver.cpp = True
    solver.solve(
        _scope.quadratic_loss,
        _scope.QuadraticData(np.matmul(X.T, X), -np.matmul(X.T, Y)),
    )
    assert set(solver.support_set) == set(linear["support_set"])
    solver.use_hessian = False
    solver.solve(
        _scope.quadratic_loss,
        _scope.QuadraticData(np.matmul(X.T, X), -np.matmul(X.T, Y)),
    )
    assert set(solver.support_set) == set(linear["support_set"])


def test_scope_greed():
    solver = ScopeSolver(linear["n_features"], linear["n_informative"], greedy=False)
    solver.solve(linear["loss"], jit=True)

    assert set(linear["support_set"]) == set(solver.support_set)


def test_scope_hessian():
    solver = ScopeSolver(linear["n_features"], linear["n_informative"])
    solver.use_hessian = True
    solver.solve(linear["loss"], jit=True)

    assert set(linear["support_set"]) == set(solver.support_set)


def test_scope_dynamic_max_exchange_num():
    solver = ScopeSolver(
        linear["n_features"], linear["n_informative"], is_dynamic_max_exchange_num=False
    )
    solver.solve(linear["loss"], jit=True)

    assert set(linear["support_set"]) == set(solver.support_set)


def test_scope_args():
    solver = ScopeSolver(
        linear["n_features"],
        gs_lower_bound=linear["n_informative"] - 1,
        gs_upper_bound=linear["n_informative"] + 1,
        group=[i for i in range(linear["n_features"])],
        screening_size=linear["n_features"],
        splicing_type="taper",
        path_type="gs",
        important_search=1,
        preselect=[linear["support_set"][0]],
    )
    solver.solve(linear["loss"], jit=True)

    solver = ScopeSolver(
        linear["n_features"],
        group=[0] + [i for i in range(linear["n_features"] - 1)],
        screening_size=0,
        path_type="gs",
        sample_size=linear["n_samples"],
        cv=2,
        split_method=lambda data, indeices: (data[0][indeices], data[1][indeices]),
    )
    solver.solve(linear["loss_data"], data=linear["data"])
