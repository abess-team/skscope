from scope import ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver, FobaSolver, FobagdtSolver
import pytest
from create_test_model import CreateTestModel


model_creator = CreateTestModel()
linear = model_creator.create_linear_model()

models = (linear,)
models_ids = ("linear",)
solvers = (ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver, FobaSolver, FobagdtSolver)
solvers_ids = ("scope", "Base", "GraHTP", "GraSP", "IHT", "FoBa-obj", "FoBa-gdt")


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_config(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(**solver.get_config())
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_always_select(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], always_select=[0, 1]
    )
    solver.solve(model["loss"], jit=True)

    assert 0 in solver.support_set
    assert 1 in solver.support_set



