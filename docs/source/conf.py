#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'scope'
copyright = '2022, abess-team'
author = 'abess-team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'matplotlib.sphinxext.plot_directive',
    'nbsphinx',
    'jupyter_sphinx',
    "sphinx_copybutton",
    'sphinx_panels',
    'autoapi.extension',
]

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'shared/*']

master_doc = 'index'

autoclass_content = 'class'
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# conda install pydata-sphinx-theme --channel conda-forge
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'github_url': 'https://github.com/abess-team/scope',
    'navigation_with_keys': False,
}

# html_logo = '../img/logo.png'
html_extra_path = ['../img/logo.png']
html_favicon = '../img/favicon.ico'

html_sidebars = {'**': ['logo.html',
                        'search-field.html',
                        'sidebar-nav-bs.html']}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
#exclude traditional Python prompts from your copied code
copybutton_prompt_text = r'>>> |\.\.\. |$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: | {2,5}\.\.\.\.:'
copybutton_prompt_is_regexp = True


# -- remove source code link --------------------------------------------------
html_show_sourcelink = False

# -- autoapi configuration ----------------------------------------------------
autoapi_dirs = ['../../scope']
#autoapi_ignore = ['test_*', 'util*']
autoapi_add_toctree_entry = False
autoapi_options = [ 'members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'imported-members', ]
autoapi_template_dir = "_templates/autoapi"
