:parenttoc: True

Installing SCOPE
===================

There are two different ways to install the python package :ref:`scope <scope_package>`:

- Install the latest official release via pip or conda. This is the recommended approach for most users.
- Building the package from source. This is recommended if you want to work with the latest development version or if you wish to contribute to scope.

Install the latest official release via pip or conda
----------------------------------------

Building the package from source
----------------------------------------

This is recommended if you want to work with the latest development version or if you wish to contribute to scope. In this case, there are some prerequisites:

- A compiler with C++11 support (see https://github.com/pybind/pybind11#supported-compilers)
- CMake >= 3.12

First download the latest source code from GitHub via

.. code-block:: Bash

    $ git clone git@github.com:abess-team/scope.git

Then build the package from source using pip in the editable mode.
The advantage of building the package with the flag ``--editable`` (or ``-e``) is that changes of the source code will immediately be
re-interpreted when the python interpreter restarts without having to re-build the package
:ref:`scope <scope_package>`.

.. code-block:: Bash

    $ pip install --editable ./scope
