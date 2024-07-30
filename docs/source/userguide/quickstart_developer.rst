:parenttoc: True

Quick Start for Developer
============================

Here, we explain how machine learning developers can leverage :ref:`skscope <skscope_package>` to facilitate the development of new machine learning methods.

Simplified Implementation 
---------------------------------

:ref:`skscope <skscope_package>` implements multiple powerful algorithms for sparsity-constraint optimization, which have been widely used in the machine learning community and continue to bring new insights into machine learning problems. When developers aim to implement a new learner, :ref:`skscope <skscope_package>` allows them to minimize the effort required for mathematical derivation and programming. For some problems, implementing a new method may require just a few lines of code. This convenience enables developers to focus on formulating the problems they encounter and quickly verify the validity of their formulations using :ref:`skscope <skscope_package>`. Here is an example.

Consider the following paper published in 2023:

- Wen, Canhong, Xueqin Wang, and Aijun Zhang. L0 trend filtering. INFORMS Journal on Computing 35.6 (2023): 1491-1510.

The L0 trend filtering considered in this paper is a concrete sparsity-constraint optimization problem, which is formulated as:

.. math::

   \arg\min_{\boldsymbol{\theta} \in \mathbb{R}^n} \| \mathbf{y} - \mathbf{D}\boldsymbol{\theta} \|_2^2 \; \textup{ subject to: } \| \boldsymbol{\theta} \|_0 \leq s,

where :math:`\mathbf{y} \in \mathbb{R}^n` and 

.. math::

   \mathbf{D} = 
   \begin{bmatrix}
       1,& 0,& 0,& \cdots, &0 \\
       1,& 1,& 0,& \cdots, &0 \\
       \cdots \\
       \cdots \\
       1,& 1,& 1,& \cdots, &1 \\
   \end{bmatrix}.

From the open-source repository provided by authors, solving this optimization problem needs more than 2400 lines of code to implement the core algorithm (see `https://github.com/INFORMSJoC/2021.0313/blob/master/scripts/AMIAS/src/AMIAS.f90 <https://github.com/INFORMSJoC/2021.0313/blob/master/scripts/AMIAS/src/AMIAS.f90>`_). However, with :ref:`skscope <skscope_package>`, solving the problem becomes pretty easy; actually, as can be seen later, no more than ten lines of code can solve the same problem.

We exemplify with a dataset that includes 500 realizations of random walk with normal increment.

.. code-block:: python

   import numpy as np
   np.random.seed(2023)
   x = np.cumsum(np.random.randn(500))

The code on solving the problem can be implemented in the following 6 lines of code:

.. code-block:: python

   import jax.numpy as jnp
   from skscope import PDASSolver
   def tf_objective(params):
       return jnp.linalg.norm(x - jnp.cumsum(params))
   solver = PDASSolver(len(x), 15)
   params = solver.solve(tf_objective)

Notice that, in our implementation, we utilized the fact that :math:`\mathbf{D}\boldsymbol{\theta}` is equivalent to the cumulative summation of :math:`\boldsymbol{\theta}` for further simplification.



High Computational Efficiency
---------------------------------

The high computational efficiency of :ref:`skscope <skscope_package>` frees users from the need to highly customize and optimize their particular methods. To demonstrate the computational power of :ref:`skscope <skscope_package>`, we provide the computational performance along with statistical performance for a dataset with 10000 samples and 10000 dimensions on two regression tasks. The numerical results of `scopesolver` and `ompsolver` are presented in the table below. 

.. list-table:: The numerical experiment results on two specific big scale SCO problems. Accuracy is equal to :math:`\frac{|\operatorname{supp}(\boldsymbol{\theta}^*) \cap \operatorname{supp}(\boldsymbol{\theta})|}{|\operatorname{supp}(\boldsymbol{\theta}^*)|}` and the runtime is measured by seconds. The results are the average of 100 replications, and the parentheses record standard deviation.
   :name: big-scale-benchmark
   :widths: auto
   :header-rows: 1

   * - **Method**
     - **Linear regression Accuracy**
     - **Linear regression Runtime**
     - **Logistic regression Accuracy**
     - **Logistic regression Runtime**
   * - ``FobaSolver``
     - 1.00(0.00)
     - 118.07(21.00)
     - 1.00(0.00)
     - 108.77(26.25)
   * - ``GraspSolver``
     - 1.00(0.00)
     - 31.66(6.68)
     - 1.00(0.00)
     - 46.16(15.10)
   * - ``HTPSolver``
     - 1.00(0.00)
     - 38.97(8.67)
     - 1.00(0.00)
     - 38.97(10.85)
   * - ``IHTSolver``
     - 1.00(0.00)
     - 39.61(10.50)
     - 1.00(0.00)
     - 39.61(11.45)
   * - ``OMPSolver``
     - 1.00(0.00)
     - 80.43(15.35)
     - 1.00(0.00)
     - 78.69(23.49)
   * - ``ScopeSolver``
     - 1.00(0.00)
     - 35.24(7.55)
     - 1.00(0.00)
     - 33.16(9.42)








For most tasks, :ref:`skscope <skscope_package>` can return results in less than two minutes on a personal computer. With this high computational efficiency, machine learning developers can focus on problem formulation rather than optimizing their implementations.


Benchmarked solution
---------------------------------

Having said that, even in cases where optimizing developers' particular methods is necessary, :ref:`skscope <skscope_package>` can assist developers. By using :ref:`skscope <skscope_package>` as a benchmark implementation, developers can compare their highly optimized implementations with the results from :ref:`skscope <skscope_package>`, ensuring that their implementations are correct.


