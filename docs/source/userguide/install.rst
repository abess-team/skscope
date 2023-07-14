:parenttoc: True

Installation
===================

There are two ways to install the python package ``skscope`` depending on your main purpose.

Install official release
-------------------------------------------------------------------------

This is the recommended approach for most users. Simply install the latest official release via ``pip``:

.. code-block:: Bash

    pip install skscope


Install library from source
----------------------------------------

This is recommended if you want to work with the latest development version or if you wish to contribute to ``skscope``. 

In this case, there are some prerequisites:

- A compiler with C++17 support
- pip 10+

- clone the latest source code from GitHub via

.. code-block:: Bash

    git clone git@github.com:abess-team/skscope.git --recurse-submodules

Note that ``--recurse-submodules`` is required since there are some submodules in the project. If there are any problem about submodules, you can search the solution in the `guide <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_ or simply raise a `github issue <https://github.com/abess-team/skscope/issues>`_.

Install on Linux or Mac
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build the package from source using pip in the editable mode:

.. code-block:: Bash

    pip install -e .

Thanks to the editable mode with the flag ``-e``, we needn't re-build the package :ref:`skscope <scope_package>` when the source python code changes. 

.. However, if the C++ code changes, we have re-build it by ``pip install -e .`` again.

If the dependence packages has been installed, we can build the package faster by  

.. code-block:: Bash

    python setup.py develop

where the function of the flag ``develop`` is similar with ``-e`` of command ``pip``.

This command will not check or prepare the required environment, so it can save a lot of time. Thus, we can use ``pip`` with first building and ``python`` with re-building.


Install on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have troubles on Windows platform, here are some helpful tips.

- The easiest way to configure the C++ compile environment on the Windows platform is to download and install the latest version of Visual Studio Community Edition. 

- There are no official binary releases of ``jaxlib`` which is necessary for ``jax`` on Windows platform. However, there are some initial community-driven native Windows supports. More details about the installation of ``jax`` can be found `here <https://github.com/google/jax#installation>`__ and a community supported Windows build for jax can be found `here <https://github.com/cloudhan/jax-windows-builder>`__ .
