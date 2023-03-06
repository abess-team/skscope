from scope import ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver
import pytest
from create_test_model import CreateTestModel


model_creator = CreateTestModel()
models = (model_creator.create_linear_model(),)
models_ids = ("linear",)
solvers = (ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver)
solvers_ids = ("scope", "Base", "GraHTP", "GraSP", "IHT")


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_base(model, solver_creator):
    """
    Basic cases:
        - only one sparisty level
        - use jax for automatic differentiation
        - without jit

    """
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.solve(model["loss"])

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_config(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(**solver.get_config())
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
@pytest.mark.parametrize("ic_type", ["aic", "bic", "gic", "ebic"])
def test_ic(model, solver_creator, ic_type):
    solver = solver_creator(
        model["n_features"], [0, model["n_informative"]], model["n_samples"], ic_type=ic_type
    )
    solver.solve(model["loss"], jit=True)

    assert set(model["support_set"]) == set(solver.support_set)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_cv(model, solver_creator):
    solver = solver_creator(
        model["n_features"],
        sparsity = [0, model["n_informative"]],
        sample_size=model["n_samples"],
        cv=5,
        split_method=model["split_method"],
    )
    solver.solve(model["loss_data"], data=model["data"])

    assert set(model["support_set"]) == set(solver.support_set)

@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_no_autodiff(model, solver_creator):
    """
    Test that the user can provide the gradient and hessian
    """
    solver = solver_creator(model["n_features"], model["n_informative"])
    if str(solver)[:5] == "Scope":
        solver.solve(model["loss_numpy"], gradient=model["grad"], hessian=model["hess"])
    else:
        solver.solve(model["loss_numpy"], gradient=model["grad"])

    assert set(model["support_set"]) == set(solver.support_set)
