from scope import FobaSolver, ForwardSolver
import pytest
from create_test_model import CreateTestModel

model_creator = CreateTestModel()
linear = model_creator.create_easy_linear_model()

models = (linear,)
models_ids = ("linear",)
solvers = (FobaSolver, ForwardSolver)
solvers_ids = ("FoBa", "Forward")

@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_objective(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)

@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_gradient(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"], use_gradient=True)
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)  