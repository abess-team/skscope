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

This is recommended if you want to work with the latest development version or if you wish to contribute to scope.

First download the latest source code from GitHub via

.. code-block:: Bash

    $ git clone git@github.com:abess-team/scope.git
    $ cd scope

Then build the package from source using pip in the editable mode.
The advantage of building the package with the flag ``--editable`` is that changes of the source code will immediately be
re-interpreted when the python interpreter restarts without having to re-build the package
:ref:`DoubleML <doubleml_package>`.

.. code-block:: Bash

    $ pip install --editable .

An alternative to pip with the ``--editable`` flag is the ``develope`` mode of setuptools. To use it call

.. code-block:: Bash

    $ python setup.py develop
