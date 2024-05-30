#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Zezhi Wang
# Copyright (C) 2023 abess-team
# Licensed under the MIT License.

from skscope import (
    ScopeSolver,
    BaseSolver,
    HTPSolver,
    GraspSolver,
    IHTSolver,
    FobaSolver,
    ForwardSolver,
    OMPSolver,
    PDASSolver,
)
import pytest
from create_test_model import CreateTestModel


model_creator = CreateTestModel()
linear = model_creator.create_linear_model()

models = (linear,)
models_ids = ("linear",)

foba_gdt_solver = lambda *args, **kwargs: FobaSolver(*args, **kwargs, use_gradient=True)
solvers = (
    ScopeSolver,
    BaseSolver,
    HTPSolver,
    GraspSolver,
    IHTSolver,
    FobaSolver,
    foba_gdt_solver,
    ForwardSolver,
    OMPSolver,
    PDASSolver,
)
solvers_ids = (
    "scope",
    "Base",
    "GraHTP",
    "GraSP",
    "IHT",
    "FOBA",
    "FOBA_gdt",
    "Forward",
    "OMP",
    "PDASSolver",
)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_config(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(**solver.get_config())
    solver.solve(model["loss"], jit=True)
    res = solver.get_result()
    assert set(model["support_set"]) == set(res["support_set"])


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_always_select(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], preselect=[0, 1]
    )
    solver.solve(model["loss"], jit=True)

    assert 0 in solver.support_set
    assert 1 in solver.support_set


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_group(model, solver_creator):
    solver = solver_creator(model["n_features"], 1, group=[0, 0, 1, 1, 1])
    solver.solve(model["loss"], jit=True)
    support_set = set(solver.get_support())
    if len(support_set) == 2:
        assert support_set == {0, 1}
    else:
        assert support_set == {2, 3, 4}
