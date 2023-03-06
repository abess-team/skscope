from scope import ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver 
import pytest
import nlopt
from create_test_model import CreateTestModel


model_creator = CreateTestModel()
models = (model_creator.create_linear_model(),)
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

    solver = solver_creator(model["n_features"], model["n_informative"], nlopt_solver=nlopt_solver)
    params = solver.solve(model["loss"], jit=True)
    
    assert model["params"] == pytest.approx(params, rel=0.01, abs=0.01)

@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_always_select(model, solver_creator):
    for i in range(model["n_features"]):
        if model["params"][i] != 0:
            continue
        solver = solver_creator(model["n_features"], model["n_informative"], always_select = [i])
        solver.solve(model["loss"], jit=True)
        
        assert i in solver.get_result()["support_set"]