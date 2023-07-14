Example Contribution
========================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're a more experienced with the ``skscope`` and are looking forward to
improve your open source development skills, the next step up is to
contribute examples to ``skscope``. 

An document in example gallery is a long-form guide to shows how to use  ``skscope`` to solve a sparsity-constrained optimization (SCO) problem. Typically, in this document, we will:

-  describes the problem and related practical background;

-  show readers the mathematics formulation for this problem and corresponding implementation based on ``skscope``;

-  use either a synthetic dataset or a real-world dataset to validate the implementation. We may also compared with the benchmarked method to illustrate the advantages of SCO and ``skscope``.

The contribution can be either:

1. improving current examples, e.g., 
  
   - correcting typos

   - clarifying concepts

   - reasonably generalizing the current method to handle practical problems

   - applying it on datasets to gain new insights

or 

2. contributing a novel example that is helpful for solving meaningful problems in practice. 

.. _general development procedure:

General Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most of case, the workflow is based on git. But if you are not familiar with command line and git, we suggest you install the `github desktop <https://desktop.github.com/>`__ that provide a user-friendly interaction interface for simplifying documentation contribution. The instrument of "github desktop" is `here <https://docs.github.com/en/desktop/>`__.

1. Fork the `master repository <https://github.com/abess-team/skscope>`__ by clicking on the “Fork” button on the top right of the page, which would create a copy to your own GitHub account;

2. Clone your fork of ``skscope`` to the local computer by
   `Git <https://git-scm.com/>`__;

   .. code:: bash

      git clone https://github.com/YourAccount/skscope.git
      cd skscope

.. _step 3:

3. Work on the local computer for the contribution of examples. 

4. Commit your improvements/contribution on examples:

   .. code:: bash

      git add changed_files
      git commit -m "YourContributionSummary"
      git push

5.  Submit a pull request via Github and explain your contribution for documentation.

Next, we will give more details about `step 3`_ to facilitate your contribution on examples.

Example Contribution Procedure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before contributing example, we presume that you have already complete the steps 1-2 described in `general development procedure`_, and you have installed necessary packages by conducting the following command.

   .. code:: bash

      pip install -r docs/requirements.txt

There are four basic steps to contribute examples:

1. If you going to contribute a new example, please create a new ``.ipynb`` file in the path ``docs/source/userguide/examples``. Or, if you would like to improve a current existing example, please find the corresponding ``.ipynb`` file in the same path. 

2. Write related contents on the ``.ipynb`` files your would like to contribute. 

3. Go to the ``docs`` directory (e.g., via ``cd docs``), 
   and running the following command in the terminal:
   
   .. code:: bash

      make html
   
This command will build web pages in your local computer. Note that, these web pages are built upon your most recent content of ``.ipynb`` files. You can preview by opening/refreshing the ``index.html`` files in ``docs/build/html`` directory.

4. Repeat step 2 and step 3 until you are satisfied with the documentation. 

After steps 1-4, you almost completely the example contribution. To ensure it also works well on the online website, please add the dependence packages in PyPI into ``docs/requirements.txt``. 


Helpful Links
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- IPython notebook files (.ipynb): `an introduction <https://realpython.com/jupyter-notebook-introduction/>`__

- `Sphinx <https://pypi.org/project/Sphinx/>`__
