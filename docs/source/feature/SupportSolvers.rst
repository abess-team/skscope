:parenttoc: True

Support Solvers
==================


There are many solvers ``skscope``. These solvers has similar interface and can be used for solving the same problem. Here we list them.

- ``GraspSolver``: implements the Gradient Support Pursuit (GraSP) algorithm that generalizes the Compressive Sampling Matching Pursuit (CoSaMP) algorithm `[1]`_.

- ``HTPSolver``: implements the hard thresholding pursuit (HTP) algorithm `[2]`_ `[3]`_. 

- ``IHTSolver``: implements the iterative hard thresholding (IHT) algorithm `[4]`_ `[5]`_. 

- ``FobaSolver``: implements Forward-Backward greedy algorithm `[6]`_.

- ``OMPSolver``: implements Orthogonal Matching Pursuit (OMP) algorithm `[7]`_ `[8]`_. 

- ``ForwardSolver``: implements forward stepwise algorithm in statistics literature `[9]`_. Although it is equivalent to ``OMPSolver``, it helps borden the implementation on statistical community. 


Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- _`[1]`: Bahmani, S., Raj, B., & Boufounos, P. T. (2013). Greedy sparsity-constrained optimization. The Journal of Machine Learning Research, 14(1), 807-841.


- _`[2]` Foucart, S. (2011). Hard thresholding pursuit: an algorithm for compressive sensing. SIAM Journal on numerical analysis, 49(6), 2543-2563.

- _`[3]` Yuan, X. T., Li, P., & Zhang, T. (2017). Gradient Hard Thresholding Pursuit. J. Mach. Learn. Res., 18(1), 6027-6069.

- _`[4]` Blumensath, T., & Davies, M. E. (2009). Iterative hard thresholding for compressed sensing. Applied and computational harmonic analysis, 27(3), 265-274.

- _`[5]` Jain, P., Tewari, A., & Kar, P. (2014). On iterative hard thresholding methods for high-dimensional m-estimation. Advances in neural information processing systems, 27.

- _`[6]` Liu, J., Ye, J., & Fujimaki, R. (2014). Forward-backward greedy algorithms for general convex smooth functions over a cardinality constraint. In International Conference on Machine Learning (pp. 503-511). PMLR.

- _`[7]` Wang, J., Kwon, S., & Shim, B. (2012). Generalized orthogonal matching pursuit. IEEE Transactions on signal processing, 60(12), 6202-6216.

- _`[8]` Tropp, J. A., & Gilbert, A. C. (2007). Signal recovery from random measurements via orthogonal matching pursuit. IEEE Transactions on information theory, 53(12), 4655-4666.

- _`[9]` Hastie, T., Tibshirani, R., Friedman, J. H., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2, pp. 1-758). New York: springer.