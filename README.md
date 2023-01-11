# Quick Start

## What is SCOPE?

Sparse-Constrained Optimization via Splicing Iteration (SCOPE) is an algorithm for getting sparse optimal solution of convex objective function, which also can be used for variables selection. Its characteristic is that the optimization objective can be a user-defined function so that the algorithm can be applied to various problems.

Specifically, SCOPE aims to tackle this problem: 
$$\min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s,$$
where $f(x)$ is a convex objective function and $s$ is the sparsity level. Each element of $x$ can be seen as a variable, and the nonzero elements of $x$ are the selected variables.

## Install

    ```bash
    git clone git@github.com:abess-team/scope.git
    cd scope
    pip install .
    ```

## First sample

```python
    from scope import ScopeSolver
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

    ## third step: solve and get the result 
    solver.solve(custom_objective) # set the optimization objective and begin to solve
    params = solver.get_params() 
    support_set = solver.get_support_set()

    print("Estimated parameter:\n", params)
    print("True parameter:\n", true_params)
```



# Basic Guide

Here, we will take linear regression as an example, introduce the basic usage of SCOPE to solve a variables selection problem.

Suppose we collect $n$ independent observations for a response variable and $p$ explanatory variables, say $y \in R^n$ and $X \in R^{n\times p}$. Let $\epsilon_1, \ldots, \epsilon_n$ be $\textit{i.i.d.}$ zero-mean random noises and $\epsilon = (\epsilon_1, \ldots, \epsilon_n)$, the linear model has a form:

$$y=X \beta^{*} +\epsilon.$$


First, we need to define a objective function as the optimization objective. 

### 1. find the objective function in math form

Formalize the problem as a sparse convex optimization problem:
$$	arg\min_{\beta \in R^p}L(\beta) := ||y-X \beta||^{2} s.t. ||\beta||_0 \leq s,$$
where objective function $L(\beta)$ is our optimization objective. 

### 2. decide parameters

Parameters, denoted as `params`, are vector which SCOPE needs to optimize and is imposed by sparse-constrain. The number of `params` is actually the dimension of the optimization problem so denoted as `dimensionality`.

From the perspective of variable selection, each parameter corresponds to a variable, and the nonzero parameters correspond to the selected variables. The number of nonzero parameters is the sparsity level and denote as `sparsity`.

In the above example, $\beta$ is the parameters, so `dimensionality` is $p$ and `sparsity` is $s$.

### 3. make custom data

We can pass any additional data through to objective function. The data may be a user-defined class containing information your objective function needs and this objective is the only function that can access the custom type, so it can be anything you want.

In the above example, the objective function need explanatory variables $X$ and response variable $y$, so we can define a class `CustomData` to store them.

```python
    class CustomData:
        def __init__(self, X, y):
            self.X = X
            self.y = y
    
    train_data = CustomData(data_set.x, data_set.y)
```

### 4. use `JAX` package

`JAX` is Autograd version of `Numpy` for high-performance machine learning research. It is a Python library that provides a NumPy-compatible multidimensional array API and automatic differentiation. As the usage of `JAX` is similar to `Numpy`, we will not introduce it here. For more information, please refer to [JAX](https://jax.readthedocs.io/en/latest/index.html).

In the above example, we can define the objective function $L(\beta)$ as `custom_objective(params)`:

```python
    import jax.numpy as jnp

    def custom_objective(params):
        return jnp.sum(
            jnp.square(train_data.y - train_data.X @ params)
        )
```

Then we can initialize the solver and set the objective function. 

### 5. initialize `ScopeSolver`

Those concepts are introduced in the previous section. 

+ `dimensionality` is the number of parameters and must be offered.
+ `sparsity` is the sparsity level, int or list of int. If it is an int, the solver will use the given sparsity level, otherwise, the solver will search the best sparsity level from the given list which will be introduced in the next section.

```python
    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=k ## we want to select k variables
    )
```

Finally, we can solve the problem and get the result.

### 7. begin to solve 

`solve` is the main function of SCOPE, it takes the objective function as optimization objective and commands the algorithm to begin the optimization process. 

```python
    solver.solve(custom_objective)
```

### 8. get results

+ `get_params` returns the optimized parameters.
+ `get_selected_variables` returns the index of selected variables (nonzero parameters).

```python
    beta = solver.get_params()
    support_set = solver.get_selected_variables()
```

# Advanced Guide

Here we will introduce some advanced features of SCOPE.

## group variables selection

In the group variable selection, variables are divided into several non-overlapping groups and each group is treated as a whole, i.e., selected at the same time or not selected at the same time. If we want to select $s$ groups of variables among $q$ groups, the group variable selection can be formulated as this:

Let $G=\{g_1, \dots, g_q\}$ be a partition of the index set $\{1, \dots, p\}$, where $g_i \subset \{1, \dots, p\}$ for $i=1, \dots, q$, $g_i \cap g_j = \emptyset$ for $i \neq j$ and $\bigcup_{i=1}^q g_i = \{1, \dots, p\}$. Then optimize the objective function with constraints:

$$
	\min_{x \in R^p} L(x),\operatorname{ s.t. } \sum_{i=1}^q I({\|x}_{g_i}\|\neq 0) \leq s,
$$
where $s$ is the group sparsity level, i.e., the number of groups to be selected. If $q=p$, then $g_i = \{i\}$ for $i=1, \dots, q$, so the group variable selection is equivalent to the original variable selection. 

In this case, we need to offer group information by `group` parameter. `group` is an incremental integer array of length `dimensionality` starting from 0 without gap, which means the variables in the same group must be adjacent, and they will be selected together or not.

Here are wrong examples: [0,2,1,2] (not incremental), [1,2,3,3] (not start from 0), [0,2,2,3] (there is a gap).

Note that `params`(parameters) must be a vector not matrix and `sparsity` represents the number of groups to be selected here.

```python
    p, s, m = 3, 2, 2
    params_true = [1, 10, 0, 0, -1, 5]
    ScopeSolver(
            dimensionality=p * m, 
            sparsity=s,
            group=[0,0,1,1,2,2]
        )
```

## auxiliary parameters

Sometimes, there are some parameters that need to be optimized but are not imposed by sparse-constrain (i.e., the always selected variables). This kind of parameters is called auxiliary parameters or `aux_params`, which is a useful concept for the convenience. The number of `aux_params` is denoted as `aux_params_size`.

Specifically, denote `params` as $x$ and `aux_params` as $\theta$, then the optimization problem is:
$$\min_{x\in R^p} f(x) := \min_{\theta}l(x,\theta) s.t.  ||x||_0 \leq s,$$
where $f(x)$ is the actual objective function but we can set $l(x,\theta)$ as objective function of `ScopeSolver`.

In this case, the following are needed:

1. `aux_params_size` needs to be offered to `ScopeSolver`.
   ```python
       solver = ScopeSolver(
           dimensionality=p, ## there are p parameters
           sparsity=k, ## we want to select k variables
           aux_params_size=1 ## there is one auxiliary parameter
       )
   ```
2. `aux_params` need to be offered to objective function too.
   ```python
       def custom_objective(params, aux_params):
           return jnp.sum(
               jnp.square(train_data.y - train_data.X @ params - aux_params)
           )
        solver.set_objective_jax(custom_objective)
   ```

In the above example, the `aux_params` is a scalar representing the intercept of linear model.


## Search support size

In the previous section, we have introduced how to set the sparsity level. However, sometimes we do not know the sparsity level and need to search it. In this case, we can set `sparsity` as a list of int, and the solver will search the best sparsity level from the given list.

Note that `sample_size` must be offered to `ScopeSolver` when `sparsity` is a list.

```python
    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n ## the number of samples
    )
```

There are two ways to evaluate sparsity levels:

### 1. cross validation

For cross validation, there are some requirements:
1. When initializing `ScopeSolver`, `sample_size` and `cv` must be offered. If `cv` is not None, the solver will use cross validation to evaluate the sparsity level. `cv` is the number of folds.
   ```python
    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n, ## the number of samples
        cv=10 ## use cross validation
    )
    ```
2. The objective function must take data as input.
    ```python
        def custom_objective(params, data):
            return jnp.sum(
                jnp.square(data.y - data.X @ params)
            )
    ```
    If there are auxiliary parameters, the data must be the last argument.
    ```python
        def custom_objective(params, aux_params, data):
            return jnp.sum(
                jnp.square(data.y - data.X @ params - aux_params)
            )
    ```
3. The data needs to be split into training and validation set. We can use `set_split_method` to set the split method. The split method must be a function that takes two arguments: `data` and `index`, and returns a new data object. The `index` is the index of training set.
    ```python
        def split_method(data, index):
            return CustomData(data.x[index, :], data.y[index])
        solver.set_split_method(split_method)
    ```


### 2. information criterion

There is another way to evaluate sparsity levels, which is information criterion. The larger the information criterion, the better the model. There are four types of information criterion can be used in SCOPE: 'aic', 'bic', 'gic', 'ebic'. If sparsity is list and `cv` is `None`, the solver will use cross validation to evaluate the sparsity level. We can use `ic` to choose information criterions, default is 'gic'.

Here is an example:
```python
    solver = ScopeSolver(
        dimensionality=p, ## there are p parameters
        sparsity=[1, 2, 3, 4, 5] ## we want to select 1-5 variables
        sample_size=n, ## the number of samples
        ic='gic' ## use default way gic to evaluate sparsity levels
    )
```

The way of defining objective function is the same as common way.

# More Examples

Here are some examples of using SCOPE.

## Sparse logistic regression

Logistic regression is a important model to solve classification problem, which is expressed specifically as:

$$
	P(y = 1 | x) = \frac{1}{1+\exp(-x^T \beta)},
$$  
$$
	P(y = 0 | x) = \frac{1}{1+\exp(x^T \beta)},
$$
where $\beta$ is an unknown parameter vector that to be estimated. Since we expect only a few explanatory variables contribute for predicting $y$, we assume $\beta$ is sparse vector with sparsity level $s$.

With $n$ independent data of the explanatory variables $x$ and the response variable $y$, we can estimate $\beta$ by minimizing the negative log-likelihood function under sparsity constraint:

$$
	arg\min_{\beta \in R^p}L(\beta) := -\frac{1}{n}\sum_{i=1}^{n}\{y_{i} x_{i}^{T} \beta-\log (1+\exp(x_{i}^{T} \beta))\}, s.t.  || \beta ||_0 \leq s.
$$

Here is Python code for solving sparse logistic regression problem:

```python
    from scope import ScopeSolver, make_glm_data
    import jax.numpy as jnp

    n, p, k = 10, 5, 3
    data = make_glm_data(n=n, p=p, k=k, family="binomial", standardize=True)
    def logistic_objective(params):
        Xbeta = jnp.matmul(data.x, params)
        # avoid overflow
        return sum(
            [x if x > 100 else 0.0 if x < -100 else log(1 + exp(x)) for x in Xbeta]
        ) - jnp.dot(data.y, Xbeta)
    
    solver = ScopeSolver(p, k)
    solver.solve(logistic_objective)

    print("Estimated parameter: ", solver.get_params(), "True parameter: ", data.coef_)
    print("Estimated sparsity level: ", solver.get_support_set(), "True sparsity level: ", k)
```

## Sparse multiple linear regression

Multiple linear regression is also known as multi output linear regression, multivariate linear regression, or multidimensional linear regression, which is used to predict multiple dependent variables using a linear combination of multiple independent variables. This is different from simple linear regression, which is used to predict a single dependent variable using a single independent variable.

The model is expressed as:
$$
    y = B^* x + \epsilon,
$$
where $y$ is an $m$-dimensional response variable, $x$ is $p$-dimensional predictors, $B \in R^{m \times p}$ is the sparse coefficient matrix, $\epsilon$ is an $m$-dimensional random noise variable with zero mean.

With $n$ independent data of the explanatory variables $X$ and the response variable $Y$, we can estimate $B^* $ by minimizing the objective function under sparsity constraint:
$$ arg\min_{B}L(B) := ||Y-B X||^2, s.t.  || B ||_ {0,2} \leq s, $$
where $|| B ||_ {0, 2}$ is the number of non-zero rows of $B$.

Here is Python code for solving sparse multiple linear regression problem:

```python
    from scope import ScopeSolver, make_multivariate_glm_data
    import jax.numpy as jnp

    n, p, k, m = 10, 5, 3, 2
    data = make_glm_data(n=n, p=p, k=k, M=m, family="multigaussian", standardize=True)
    def multi_linear_objective(params):
        params.reshape((p, m))
        return jnp.sum(jnp.square(data.y - jnp.matmul(data.x, params)))

    solver = ScopeSolver(p * m, k, group=[i for i in range(p) for j in range(m)])
    solver.solve(multi_linear_objective)

    print("Estimated parameter: ", solver.get_params(), "True parameter: ", data.coef_)
    print("Estimated sparsity level: ", solver.get_support_set(), "True sparsity level: ", k)
```
