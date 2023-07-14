Contribute Python Code
========================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are a experienced programmer, you might want to help new features
development or bug fixing for the ``skscope`` library. Here, we going to a general workflow to ease your contribution. 

Before Contribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before contributing, we strongly recommend you conduct self-check for the points listed below.

- You has always open an `issue <https://github.com/abess-team/scope/issues>`__ and make sure someone from the `abess-team <https://github.com/abess-team>`__ agrees that it is really contributive. We don't want you to spend a bunch of time on something that we are working on or we don't think is a good idea.

- Fork the `master repository <https://github.com/abess-team/skscope>`__ by clicking on the “Fork” button on the top right of the page, which would create a copy to your own GitHub account.

- If you have forked ``skscope`` repository, enter it and click "Fetch upsteam", followed by "Fetch and merge" to ensure the code is the latest one

- Clone ``skscope`` into the local computer:

   .. code:: bash

      git clone https://github.com/your_account_name/skscope.git
      cd skscope

- Install ``skscope`` via the code in github by following `installation instruction <../userguide/install.html>`__;

General Workflow 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is the preferred workflow.

1. Developing python code that is used for supplying new features or addressing the reported bugs. 

2. Additionally, you shall add related test code into the ``pytest`` directory and conduct the automated testing:

   .. code:: bash

      cd pytest      
      pytest .

to ensure new features work properly or the bugs are well addressed.

3. Repeat steps 1-2 until ``pytest .`` does not raise any error. Once step 3 is completed, it means the python code development is finished.

To complete the workflow, you shall first push them onto your repository:

   .. code:: bash

      git add changed_files
      git commit -m "YourContributionSummary"
      git push

Next, look back to GitHub, merge your code with the up-to-date codes in `skscope <https://github.com/abess-team/skscope>`__ by clicking the “Contribute” button on your fork to open pull request. Then, we will receive your contribution.

Develop A New Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this part, we are particularly concerned the new features about developing a fast sparsity-constrained optimization (SCO) solver.
We will illustrate a recommended procedure for developing a SCO solver such that your implementation works fluently under the ``skscope``' framework. The potential solvers may include:

- Newton hard threshold thresholding :ref:`[1] <_ref_1>`,

- fast iterative shrinkage-thresholding algorithm (FISTA) :ref:`[2] <_ref_2>`, 

and so on. In the following, we will use FISTA as an example to exhibit the procedure for developing a new solver. 

Implement a New Solver
---------------------------

First, create a new python file in ``skscope`` such as ``skscope/fista.py`` that includes the implementation of this method. In this python file, the sketch of the code is given below.

.. code:: python

   # new algorithms should inherit the base class `BaseSolver`
   from .base_solver import BaseSolver

   class NewSolver(BaseSolver): 
      """
      Introduction about the new algorithm
      """
      def __init__(self, ...):
         super(NewSolver, self).__init__(
            step_size=0.0005, 
            # other init
         )
   
      def _solve(
         self,
         sparsity,
         loss_fn,
         value_and_grad,
         init_support_set,
         init_params,
         data,
      ):
      # Implement the core iterative procedure of the new algorithm

The ``BaseSolver`` implements some generic functions, which plays a role on
checking input and extracting compute results. 
After implementation, don't forget to import the new algorithm in
``skscope/__init__.py``.

Now run ``pip install -e .`` again and this time the installation would be finished quickly. Congratulation! Your work can now be used by:

.. code:: python

   from skscope import NewSolver


Test the Solver
---------------------------

After programming the code, it is necessary to verify the contributed
solver can return a reasonable result. Here, we share our experience
for it. 

1. Test the solver for the compress sensing problem.

Document the Solver
----------------------------

The contribution is almost done. The remaining thing is add a document for this solver. A new solver need a brief introduction and some examples. Also note that the style of Python document is similar to `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

The development of Python API's documentation mainly relies on
`Sphinx <https://pypi.org/project/Sphinx/>`__, `sphinx-gallery <https://pypi.org/project/sphinx-gallery/>`__ (support markdown for Sphinx), `sphinx-rtd-theme <https://pypi.org/project/sphinx-rtd-theme/>`__
(support “Read the Docs” theme for Sphinx) and so on. Please make sure all packages in :code:`docs/requirements.txt` have been installed by:

   .. code:: bash

      pip install -r docs/requirements.txt


Helpful Links 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``pytest``: `a quick start guide <https://www.packtpub.com/product/pytest-quick-start-guide/9781789347562>`__

- Architecture of ``skscope``: `a graphical illustration <AppendixArchitecture.html>`__

- Advanced topics for writing documentation: `Sphinx <https://www.sphinx-doc.org/en/master/>`__.


Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _ref_1:

- [1] Zhou, S., Xiu, N., & Qi, H. D. (2021). Global and quadratic convergence of Newton hard-thresholding pursuit. The Journal of Machine Learning Research, 22(1), 599-643.

.. _ref_2:

- [2] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), 183-202.