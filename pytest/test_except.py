import numpy as np
import pytest
import re

from create_test_model import CreateTestModel
from skscope import ScopeSolver, BaseSolver
import skscope._scope as _scope

model_creator = CreateTestModel()
linear = model_creator.create_linear_model()

models = (linear,)
models_ids = ("linear",)
solvers = (ScopeSolver, BaseSolver)
solvers_ids = ("scope", "Base")


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_group(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(group=np.zeros((1, model["n_features"])))
    with pytest.raises(
        ValueError, match=re.escape("Group should be an 1D array of integers.")
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(group=np.zeros(model["n_features"] + 1))
    with pytest.raises(
        ValueError,
        match=re.escape("The length of group should be equal to dimensionality."),
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(group=np.ones(model["n_features"]))
    with pytest.raises(ValueError, match=re.escape("Group should start from 0.")):
        solver.solve(model["loss"], jit=True)
    solver.set_config(group=-np.arange(model["n_features"]))
    with pytest.raises(
        ValueError, match=re.escape("Group should be an incremental integer array.")
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(group=np.arange(0, 2 * model["n_features"], 2))
    with pytest.raises(ValueError, match=re.escape("There is a gap in group.")):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_preselect(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(preselect=[-1])
    with pytest.raises(
        ValueError, match=re.escape("preselect should be between 0 and dimensionality.")
    ):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_sparsity(model, solver_creator):
    solver = solver_creator(model["n_features"], model["n_informative"])
    solver.set_config(sparsity=[])
    with pytest.raises(ValueError, match=re.escape("Sparsity should not be empty.")):
        solver.solve(model["loss"], jit=True)
    solver.set_config(sparsity=[-1])
    with pytest.raises(ValueError, match=re.escape("There is an invalid sparsity.")):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_ic(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], model["n_samples"]
    )
    solver.set_config(ic_type="ic")
    with pytest.raises(
        ValueError,
        match=re.escape("ic_type should be one of ['aic', 'bic', 'sic','ebic']."),
    ):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_cv(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], model["n_samples"]
    )
    solver.set_config(cv=1 + model["n_samples"])
    with pytest.raises(
        ValueError, match=re.escape("cv should not be greater than sample_size.")
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(cv=model["n_samples"])
    with pytest.raises(
        ValueError, match=re.escape("split_method should be provided when cv > 1.")
    ):
        solver.solve(model["loss"], data=(), jit=True)
    solver.set_config(cv_fold_id=np.zeros((1, model["n_samples"])))
    with pytest.raises(
        ValueError, match=re.escape("cv_fold_id should be an 1D array of integers.")
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(cv_fold_id=np.zeros(1 + model["n_samples"]))
    with pytest.raises(
        ValueError,
        match=re.escape("The length of cv_fold_id should be equal to sample_size."),
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(cv_fold_id=np.zeros(model["n_samples"]))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of different elements in cv_fold_id should be equal to cv."
        ),
    ):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
@pytest.mark.parametrize("solver_creator", solvers, ids=solvers_ids)
def test_init(model, solver_creator):
    solver = solver_creator(
        model["n_features"], model["n_informative"], model["n_samples"]
    )
    with pytest.raises(
        ValueError,
        match=re.escape("The initial active set should be an 1D array of integers."),
    ):
        solver.solve(model["loss"], init_support_set=np.zeros((1, model["n_features"])))
    with pytest.raises(
        ValueError, match=re.escape("init_support_set contains wrong index.")
    ):
        solver.solve(model["loss"], init_support_set=[-1])
    with pytest.raises(
        ValueError,
        match=re.escape("The length of init_params should be equal to dimensionality."),
    ):
        solver.solve(model["loss"], init_params=np.zeros((1, model["n_features"])))


@pytest.mark.parametrize("model", models, ids=models_ids)
def test_log(model):
    solver = solvers[0](model["n_features"], model["n_informative"], model["n_samples"])
    solver.set_config(file_log_level="info")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "console_log_level and file_log_level must be in 'off', 'error', 'warning', 'debug'"
        ),
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(file_log_level="error", log_file_name=123)
    with pytest.raises(ValueError, match=re.escape("log_file_name must be a string")):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
def test_gs(model):
    solver = solvers[0](model["n_features"])
    solver.set_config(path_type="123")
    with pytest.raises(
        ValueError, match=re.escape("path_type should be 'seq' or 'gs'")
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(gs_upper_bound=model["n_features"] + 1, path_type="gs")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "gs_lower_bound and gs_upper_bound should be between 0 and dimensionality."
        ),
    ):
        solver.solve(model["loss"], jit=True)
    solver.set_config(
        gs_upper_bound=model["n_features"], gs_lower_bound=model["n_features"] + 1
    )
    with pytest.raises(
        ValueError,
        match=re.escape("gs_upper_bound should be larger than gs_lower_bound."),
    ):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
def test_screening_size(model):
    solver = solvers[0](model["n_features"], model["n_informative"])
    solver.set_config(screening_size=model["n_informative"] - 1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "screening_size should be between sparsity and dimensionality."
        ),
    ):
        solver.solve(model["loss"], jit=True)


@pytest.mark.parametrize("model", models, ids=models_ids)
def test_splicing_type(model):
    solver = solvers[0](model["n_features"], model["n_informative"])
    solver.set_config(splicing_type="123")
    with pytest.raises(
        ValueError, match=re.escape("splicing_type should be 'halve' or 'taper'.")
    ):
        solver.solve(model["loss"], jit=True)
