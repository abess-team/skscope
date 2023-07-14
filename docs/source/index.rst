.. _scope_package:

SCOPE
=================================

The Python package **SCOPE** is for getting solution of sparsity-constrained optimization, which also can be used for variables selection. Its characteristic is that the optimization objective can be a user-defined function so that the algorithm can be applied to various problems.

Specifically, **SCOPE** aims to tackle this problem: 

.. math:: 
   \min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s,

where :math:`f(x)` is a differential objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.

.. grid:: 1 2 2 3
   :gutter: 1

   .. grid-item-card::
      :link: userguide/install.html

      :fas:`download;pst-color-primary` **What is skscope**
      ^^^
      What is skscope ?

   .. grid-item-card::
      :link: userguide/install.html

      :fas:`bolt;pst-color-primary` **Quick Start**
      ^^^
      Quick Start

   .. grid-item-card::
      :link: userguide/examples/index.html

      :fas:`list;pst-color-primary` **Examples Gallery**
      ^^^
      Examples Gallery

   .. grid-item-card::
      :link: feature/index.html

      :fas:`palette;pst-color-primary` **Software Features**
      ^^^
      Software Features

   .. grid-item-card::
      :link: autoapi/index.html

      :fas:`file;pst-color-primary` **API Reference**
      ^^^
      API Reference

   .. grid-item-card::
      :link: contribute/index.html

      :fas:`terminal;pst-color-primary` **Contributor's Guide**
      ^^^
      Contributor's Guide

User Guide
--------------------

Information about how to use the software.

.. toctree::
   :maxdepth: 3

   userguide/index


Software Features
-------------------------------

Some special features to notice.

.. toctree::
   :maxdepth: 3

   feature/index

API Reference
--------------------------

Refer to API.

.. toctree::
   :maxdepth: 3

   autoapi/index


Contributor's Guide
----------------------------

.. toctree::
   :maxdepth: 3

   contribute/index