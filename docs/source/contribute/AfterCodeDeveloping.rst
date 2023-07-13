After Code Developing
=====================

CodeFactor
----------

We check the Python format by
`CodeFactor <https://www.codefactor.io/repository/github/abess-team/abess>`__.
More specifically, the formatters and rules are:

- C++: `CppLint <https://github.com/google/styleguide/tree/gh-pages/cpplint>`__ with `CPPLINT.cfg <https://github.com/abess-team/abess/blob/master/CPPLINT.cfg>`__

- Python: `Pylint <https://www.pylint.org/>`__ with `.pylintrc <https://github.com/abess-team/abess/blob/master/.pylintrc>`__

Each pull request will be checked, and some recommendations will be given if not passed. But don't be worry about those complex rules, most of them can be formatted automatically by some tools.

   Note that there may be few problems that the auto-fix tools can NOT
   deal with. In that situation, please update your pull request
   following the suggestions by CodeFactor.

Auto-format
-----------

Python
~~~~~~

`Autopep8 <https://pypi.org/project/autopep8/>`__ can be used in
formatting Python code. You can easily install it by
``$ pip install autopep8``.

.. _with-vs-code-1:

with VS Code
^^^^^^^^^^^^

Visual Studio Code can deal with Python auto-fix too, with
`Python <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__
extension.

There is no more steps to do. Right-click on an Python file and then
click “Format Document”. That will be done.

.. _with-command-line-1:

with command line
^^^^^^^^^^^^^^^^^

As we memtioned above, the default setting of Autopep8 is enough for us.
Hence run ``$ autopep8 some_code.py > some_code_formatted.py`` and the
formatted code is stored in ``some_code_formatted.py`` now.
