#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Zezhi Wang
# Copyright (C) 2023 abess-team
# Licensed under the MIT License.

import pytest
import numpy as np

from skscope import (
    ScopeSolver,
    BaseSolver,
    HTPSolver,
    GraspSolver,
    IHTSolver,
    FobaSolver,
    ForwardSolver,
    OMPSolver,
    layer,
)

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
)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_linear_constraint(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], random_state=42
    )
    solver.solve(
        model["loss"],
        jit=True,
        layers=[
            layer.Identity(model["n_features"]),
            layer.LinearConstraint(model["n_features"]),
        ],
    )

    assert np.sum(solver.get_estimated_params()) == pytest.approx(1)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_non_negative(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], random_state=42
    )
    solver.solve(
        model["loss"],
        jit=True,
        layers=[
            layer.Identity(model["n_features"]),
            layer.NonNegative(model["n_features"]),
        ],
    )

    assert np.all(solver.get_estimated_params() >= 0.0)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_simplex_constraint(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], random_state=42
    )
    solver.solve(
        model["loss"],
        jit=True,
        layers=[
            layer.Identity(model["n_features"]),
            layer.SimplexConstraint(model["n_features"]),
        ],
    )

    assert np.all(solver.get_estimated_params() >= 0.0)
    assert np.sum(solver.get_estimated_params()) == pytest.approx(1)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_box_constraint(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], random_state=42
    )
    solver.solve(
        model["loss"],
        jit=True,
        layers=[
            layer.Identity(model["n_features"]),
            layer.BoxConstraint(model["n_features"], -1.0, 1.0),
        ],
    )

    assert np.all(solver.get_estimated_params() >= -1.0)
    assert np.all(solver.get_estimated_params() <= 1.0)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_offset_sparse(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], random_state=42
    )
    solver.solve(
        model["loss"],
        jit=True,
        layers=[
            layer.Identity(model["n_features"]),
            layer.OffsetSparse(model["n_features"], -1.0),
        ],
    )

    assert (
        np.sum(solver.get_estimated_params() == -1.0)
        == model["n_features"] - model["n_informative"]
    )
