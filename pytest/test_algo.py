from scope import ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver 
import pytest
import nlopt
from create_test_model import CreateTestModel
import jax


create_test_model = CreateTestModel()
linear = create_test_model.create_linear_model()
models = (linear,)
solvers = (ScopeSolver, BaseSolver, GrahtpSolver, GraspSolver, IHTSolver)
solvers_no_scope = (BaseSolver, GrahtpSolver, GraspSolver, IHTSolver)


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("solver_creator", solvers)
def test_base(model, solver_creator):
    """
    Basic cases: 
        - only one sparisty level
        - use jax for automatic differentiation
        - without jit
    
    """
    solver = solver_creator(model["n_features"], model["n_informative"])
    params = solver.solve(model["loss"])
    
    assert model["params"] == pytest.approx(params, rel=0.01, abs=0.01)

@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("solver_creator", solvers)
def test_nlopt_solver(model, solver_creator):
    """
    Custom nlopt solver
    """
    nlopt_solver = nlopt.opt(nlopt.LD_SLSQP, 1)
    nlopt_solver.set_ftol_rel(0.001)

    solver = solver_creator(model["n_features"], model["n_informative"], nlopt_solver=nlopt_solver)
    params = solver.solve(model["loss"], jit=True)
    
    assert model["params"] == pytest.approx(params, rel=0.01, abs=0.01)

@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("solver_creator", solvers)
def test_config(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(**solver.get_config())
    params = solver.solve(model["loss"], jit=True)
    
    assert model["params"] == pytest.approx(params, rel=0.01, abs=0.01)

@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("solver_creator", solvers)
def test_aic(model, solver_creator):
    solver = solver_creator(model["n_features"])
    params = solver.solve(model["loss"], jit=True)
    
    assert params.size == model["n_features"]

@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("solver_creator", solvers)
def test_cv(model, solver_creator):
    solver = solver_creator(model["n_features"], sample_size = model["n_samples"], cv = 5, split_method = model["split_method"])
    params = solver.solve(model["loss_data"], data = model["data"])

    assert model["params"] == pytest.approx(params, rel=0.01, abs=0.01)

def test_grad():
    pass

def test_grad_hess():
    pass

@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("solver_creator", solvers)
def test_always_select(model, solver_creator):
    for i in range(model["n_features"]):
        if model["params"][i] != 0:
            continue
        solver = solver_creator(model["n_features"], model["n_informative"], always_select = [i])
        solver.solve(model["loss"], jit=True)
        
        assert i in solver.get_result()["support_set"]
