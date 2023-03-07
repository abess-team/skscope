import numpy as np
import pytest
import nlopt

from create_test_model import CreateTestModel
from scope import ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver


model_creator = CreateTestModel()
linear = model_creator.create_linear_model()

models = (linear,)
models_ids = ("linear",)
solvers = (ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver)
solvers_ids = ("scope", "Base", "GraHTP", "GraSP", "IHT")


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_nlopt_solver(model, solver_creator):
    """
    Custom nlopt solver
    """
    nlopt_solver = nlopt.opt(nlopt.LD_SLSQP, 1)
    nlopt_solver.set_ftol_rel(0.001)

    solver = solver_creator(
        model["n_features"], model["n_informative"], nlopt_solver=nlopt_solver
    )
    params = solver.solve(model["loss"], jit=True)

    assert model["params"] == pytest.approx(params, rel=0.01, abs=0.01)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_always_select(model, solver_creator):
    for i in range(model["n_features"]):
        if model["params"][i] != 0:
            continue
        solver = solver_creator(
            model["n_features"], model["n_informative"], always_select=[i]
        )
        solver.solve(model["loss"], jit=True)

        assert i in solver.get_result()["support_set"]


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_init_support_set(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.solve(model["loss"], init_support_set=[0, 1, 2], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_init_params(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.solve(model["loss"], init_params=np.ones(model["n_features"]), jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


def test_add_coverage():
    solver = ScopeSolver(
        linear["n_features"],
        gs_lower_bound=linear["n_informative"] - 1,
        gs_upper_bound=linear["n_informative"] + 1,
        group=[i for i in range(linear["n_features"])],
        screening_size=linear["n_features"],
        splicing_type="taper",
        path_type="gs",
    )
    solver.solve(linear["loss"], jit=True)

    assert set(linear["support_set"]) == set(solver.support_set)
