from scope import BaseSolver, GrahtpSolver, GraspSolver, IHTSolver, FobaSolver, ForwardSolver, OmpSolver
import scope
import numpy as np
import pytest
from create_test_model import CreateTestModel


linear = CreateTestModel().create_linear_model()

foba_gdt_solver = lambda *args, **kwargs: FobaSolver(*args, **kwargs, use_gradient=True)
solvers = (BaseSolver, GrahtpSolver, GraspSolver, IHTSolver, FobaSolver, foba_gdt_solver, ForwardSolver, OmpSolver)
solvers_ids = ("Base", "GraHTP", "GraSP", "IHT", "FOBA", "FOBA_gdt", "Forward", "OMP")

@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_quadratic_objective(solver_creator):
    solver = solver_creator(linear["n_features"], linear["n_informative"])
    X, Y = linear["data"]
    model = scope.quadratic_objective(np.matmul(X.T, X), -np.matmul(X.T, Y))
    solver.solve(**model)
    assert set(linear["support_set"]) == set(solver.support_set)

    