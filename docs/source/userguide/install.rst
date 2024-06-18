:parenttoc: True

Installation
===================

There are two ways to install the python package :ref:`skscope <skscope_package>` depending on your main purpose.

Install official release
-------------------------------------------------------------------------

This is the recommended approach for most users. Simply install the latest official release via ``pip``:

.. code-block:: Bash

    pip install skscope

For Linux or Mac users, an alternative is

.. code-block:: Bash

    conda install skscope


Install library from source
----------------------------------------

This is recommended if you want to work with the latest development version or if you wish to contribute to :ref:`skscope <skscope_package>`. 


1. First of all, make sure that you have a compiler with C++17 support installed. 
:ref:`skscope <skscope_package>` is a python package with C++ extension, so a C++ Toolchain is required for building it.
For Windows users, the easiest way to configure the C++ compile environment is to install the latest version of 
`Visual Studio Community Edition <https://visualstudio.microsoft.com/downloads/>`_ and choose the "Desktop development with C++" workload. 
`Here <https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation>`_ is a detailed tutorial. 


2. Then, clone the latest source code from GitHub and enter the directory of ``setup.py``:

.. code-block:: Bash

    git clone git@github.com:abess-team/skscope.git --recurse-submodules
    cd skscope

Note that ``--recurse-submodules`` is required since there are some submodules in the project. 


3. Finally, build the package from source using pip (in the editable mode):

.. code-block:: Bash

    pip install -e .

Thanks to the editable mode with the flag ``-e``, we needn't re-build the package :ref:`skscope <skscope_package>` when the source python code changes. 




Dependencies
----------------------------------------

The current minimum dependencies to run ``skscope`` are:

- ``numpy``
- ``scikit-learn>=1.2.2``
- ``jax[cpu]``