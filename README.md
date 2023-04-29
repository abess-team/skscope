[![codecov](https://codecov.io/gh/abess-team/scope/branch/master/graphs/sunburst.svg)](https://codecov.io/gh/abess-team/scope)

## What is `scope`?

Sparsity-Constraint OPtimization via itErative (scope) are algorithms for getting sparse optimal solution of convex objective function, which also can be used for variables selection. Its characteristic is that the optimization objective can be a user-defined function so that the algorithm can be applied to various problems.

Specifically, SCOPE aims to tackle this problem: 
$$\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s,$$
where $f(x)$ is a convex objective function and $s$ is the sparsity level. Each element of $x$ can be seen as a variable, and the nonzero elements of $x$ are the selected variables.

Anothoer mean of `scope` is a specific algorithm, named `Sparse-Constrained Optimization via sPlicing itEration`, which is a general version of `abess`.

Now, this library has supported these algorithm: `scope`, `GraHTP`, `GraSP`, `IHT`.

## Building the package from source

### Prerequisites
+ A compiler with C++17 support 
+ Pip 10+ or CMake >= 3.14

### Install

```bash
git clone git@github.com:abess-team/scope.git --recurse-submodules
pip install ./scope
```

### Building the documentation

```bash
cd scope/docs
pip install -r requirements.txt
conda install -c conda-forge pandoc
make html
```

See the document for more installation details.

## Test

### pytest

```bash
pip install pytest
pytest scope/pytest
```


### First sample

```python
from scope import ScopeSolver, BaseSolver, HTPSolver, GraspSolver, IHTSolver 
import jax.numpy as jnp
from sklearn.datasets import make_regression

## generate data
n, p, k= 10, 5, 3
x, y, true_params = make_regression(n_samples=n, n_features=p, n_informative=k, coef=True)

## first step: define objective function
def custom_objective(params):
    return jnp.sum(
        jnp.square(y - x @ params)
    )

## second step: initialize the solver
solver = ScopeSolver(p, k) # there are p optimization parameters, k of which are non-zero
# solver = BaseSolver(p, k) ## HTPSolver, GraspSolver, IHTSolver are the same

## third step: solve and get the result 
params = solver.solve(custom_objective) # set the optimization objective and begin to solve
optim_result = solver.get_result()

print("Estimated parameter:\n", params)
print("True parameter:\n", true_params)
print("Optim result:\n", optim_result)
```

## Compare with SOTA methods

![](/docs/img/compare.png)
