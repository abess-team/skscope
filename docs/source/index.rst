.. _scope_package:

SCOPE
=================================

Sparse-Constrained Optimization via Splicing Iteration (SCOPE) is an algorithm for getting sparse optimal solution of convex objective function, which also can be used for variables selection. Its characteristic is that the optimization objective can be a user-defined function so that the algorithm can be applied to various problems.

Specifically, SCOPE aims to tackle this problem: 

.. math:: 
   \min_{x \in R^p} f(x) \text{ s.t. } ||x||_0 \leq s,

where :math:`f(x)` is a convex objective function and :math:`s` is the sparsity level. Each element of :math:`x` can be seen as a variable, and the nonzero elements of :math:`x` are the selected variables.

.. raw:: html

    <script type="text/javascript">
        // Change icons to light or dark mode icons when theme-switch occures
        var panel_icons = document.getElementsByClassName('panel-icon');
        var observer = new MutationObserver(function(mutations) {
            const dark = document.documentElement.dataset.theme == 'dark';
            for (let i = 0; i < panel_icons.length; i++) {
                var filename = panel_icons[i].src.split('/').slice(-1)[0].split('_')[1];
                panel_icons[i].src = dark ? '_static/dark_' + filename : '_static/light_' + filename;
            }
        });
    observer.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  </script>

.. panels::
    :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2
    :card: text-center
    :img-top-cls: pl-5 pr-5 pt-5 pb-5 panel-icon
    :header: font-weight-bold border-0 h4
    :footer: border-0

    ---
    :img-top: _static/light_quickStart.png

    Quick start
    ^^^^^^^^^^^^^^^

    Here is quick start document.

    +++

    .. link-button:: intro/quickStart
        :type: ref
        :text: To the quick start 
        :classes: btn-block btn-dark stretched-link btn-sm

    ---
    :img-top: _static/light_userguide.png

    User guide
    ^^^^^^^^^^

    Here is user guide document.

    +++

    .. link-button:: guide/guide
        :type: ref
        :text: To the user guide
        :classes: btn-block btn-dark stretched-link btn-sm


    ---
    :img-top: _static/light_api.png

    Features
    ^^^^^^^^^^

    Here is a page for features.

    +++

    .. link-button:: guide/features
        :type: ref
        :text: To the user guide
        :classes: btn-block btn-dark stretched-link btn-sm


    ---
    :img-top: _static/light_api.png

    API
    ^^^^^^^^^^

    Here is API document.

    +++

    .. link-button:: autoapi/scope/index
        :type: ref
        :text: To the API
        :classes: btn-block btn-dark stretched-link btn-sm


.. toctree::
   :hidden:

   self

.. toctree::
   :hidden:

   Install <intro/install>
   Quick Start <intro/quickStart>
   Examples <examples/guide>
   Features <guide/features>
   API <autoapi/scope/index>
   Reference <reference/reference>
   Release notes <releaseNote/releaseNote>

Video Guide
---------------------------------

.. raw:: html

    <p align="center">
        <iframe width="500" height="400" src="https://www.bilibili.com/video/BV13g41187rQ" title="The Twenty" frameborder="0" allowfullscreen></iframe>
    </p>


Main Features
-------------
* **User-defined objective function**: The objective function can be a user-defined function, which can be applied to various problems.
* **Fast**: The algorithm is fast and can be applied to large-scale problems.
* **Easy to use**: The algorithm is easy to use and can be applied to various problems.

Citation
--------

If you use the scope package a citation is highly appreciated:

Acknowledgements
----------------
