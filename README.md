[![codecov](https://codecov.io/gh/abess-team/scope/branch/master/graphs/sunburst.svg)](https://codecov.io/gh/abess-team/scope)

# ``skscope``: Fast Sparse-Constraint Optimization

## What is `skscope`?

``skscope`` aims to make sparsity-constrained optimization (SCO) is accessible to everyone because SCO holds immense potential across various domains, including machine learning, statistics, and signal processing. By providing a user-friendly interface, ``skscope`` empowers individuals from diverse backgrounds to harness the power of SCO and unlock its broad range of applications (see examples exhibited below).

![](docs/source/first_page.png)

## Installation

The recommended option for most of users:
  
```bash
pip install skscope
```

If you want to work with the latest development version, the further [installation instructions](skscope.readthedocs.io/userguide/install.html) help you install from source.

## Quick examples

Here's a quick example showcasing how you can use three simple steps to perform feature selection via the ``skscope``:

```python
from skscope import ScopeSolver
from sklearn.datasets import make_regression
import jax.numpy as jnp

## generate data
x, y, coef = make_regression(n_features=10, n_informative=3, coef=True)

## 1. define loss function
def ols_loss(para):
    return jnp.sum(jnp.square(y - x @ para))

## 2. initialize the solver where 10 parameters in total and three of which are sparse
solver = ScopeSolver(10, 3) 

## 3. use the solver to optimized the objective
params = solver.solve(ols_loss) 
```

Below's another example illustrates that you can modify objective function to address another totally different problem. 

```python
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from skscope import ScopeSolver

## generate data
np.random.seed(2023)
x = np.cumsum(np.random.randn(500)) # random walk with normal increment

## 1. define loss function
def tf_objective(params):
    return jnp.sum(jnp.square(x - jnp.cumsum(params)))  

## 2. initialize the solver where 10 parameters in total and three of which are sparse
solver = ScopeSolver(len(x), 10)

## 3. use the solver to optimized the objective
params = solver.solve(tf_objective)

tf_x = jnp.cumsum(params)
plt.plot(x, label='observation', linewidth=0.8)
plt.plot(tf_x, label='filtering trend')
plt.legend(); plt.show()
```

<p align="center">
<img src="docs/source/userguide/figure/tf.png" width="300"/>
</p>

The above Figure shows that the solution of ``ScopeSolver`` now capture the main trend of the observed random work. Again, 4 lines of code help us attain the solution. 

## Example gallery

Since ``skscope`` can easily be applied to diverse objective functions, we can definitely leverage it to develop various machine learning methods that is driven by SCO. In our example gallery, we supply 25 comprehensive statistical/machine learning examples to illustrate the versatility of ``skscope``. 

## Why ``skscope`` is versatile?

The high versatility of ``skscope`` in effectively addressing SCO problems are derived from two key factors: theoretical concepts and computational implementation. In terms of theoretical concepts, there have been remarkable advancements on SCO in recent years, offering a range of efficient iterative methods for solving SCO. Some of these algorithms exhibit elegance by only relying on the current parameters and gradients for the iteration process. On the other hand, significant progress has been made in automatic differentiation, a fundamental component of deep learning algorithms that plays a vital role in computing gradients. By ingeniously combining these two important advancements, ``skscope`` emerges as the pioneering tool capable of handling diverse sparse optimization tasks.

With ``skscope``, the creation of new machine learning methods becomes effortless, leading to the advancement of the "sparsity idea" in machine learning. This, in turn, facilitates the availability of a broader spectrum of machine learning algorithms for tackling real-world problems.

## Software features

- Support multiple state-of-the-art SCO solvers. Now, ``skscope`` has supported these algorithm: SCOPE, HTP, Grasp, IHT, OMP, FoBa. 

- User-friendly API
  
  - zero-knowledge of SCO solvers: the state-of-the-art solvers in ``skscope`` has intuitive and highly unified APIs. 
  
  - extensive documentation: ``skscope`` is fully documented and accompanied with example gallery and reproduction scripts.

- Solving SCO and its generalization: 
  
  - SCO: $\arg\min\limits_{\theta \in R^p} f(\theta) \text{ s.t. } ||\theta||_0 \leq s$; 
  
  - SCO for group-structure parameters: $\arg\min\limits_{\theta \in R^p} f(\theta) \text{ s.t. } I(||\theta_{{G}_i}||_2 \neq 0) \leq s$ where $\{{G}_i\}_{i=1}^q$ is a non-overlapping partition for $\{1, \ldots, p\}$;
  
  - SCO when pre-selecting parameters in set $\mathcal{P}$: $\arg\min\limits_{\theta \in R^p} f(\theta) \text{ s.t. } ||\theta_{\mathcal{P}^c}||_0 \leq s$. 

- Data science toolkit
  
  - Information criterion and cross validation for selecting $s$
  
  - Portable interface for developing new machine learning methods

- Just-in-time-compilation compatibility


