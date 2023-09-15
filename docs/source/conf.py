# Configuration file for the Sphinx documentation builder.
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
from pathlib import Path
from typing import Any, Dict

import pydata_sphinx_theme
from sphinx.application import Sphinx
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../skscope'))
sys.path.append(str(Path(".").resolve()))


# -- Project information -----------------------------------------------------

project = 'skscope'
copyright = '2023, abess-team'
author = 'abess-team'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    'sphinx.ext.coverage',
    'sphinx.ext.graphviz',
    "sphinx.ext.viewcode",
    #"sphinxext.rediraffe",
    "sphinx.ext.mathjax",
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
    "sphinx_design",
    "sphinx_copybutton",
    #"_extension.gallery_directive",
    # For extension examples and demos
    "nbsphinx",
    #"ablog",
    "jupyter_sphinx",
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    "matplotlib.sphinxext.plot_directive",
    #"myst_nb",
    #"sphinxcontrib.youtube",
    "numpydoc",
    "sphinx_togglebutton",
    #"jupyterlite_sphinx",
    "sphinx_favicon",
    "autoapi.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path  =  ['_templates']

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
# panels_add_bootstrap_css = False


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]

    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

# -- MyST options ------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
# myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
# myst_heading_anchors = 2
# myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/skscope-light.png"
html_favicon = "_static/logo-short.svg"
html_sourcelink_suffix = ""

# Define the json_url for our version switcher.
json_url = "https://skscope.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    release = "dev"
    if "dev" in release or "rc" in release:
        version_match = "dev"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = "v" + release

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/abess-team/skscope",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/skscope/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Conda",
            "url": "https://anaconda.org/",
            "icon": "fa-solid fa-box",
        },
    ],
    # alternative way to set twitter and github header icons
    # "github_url": "https://github.com/pydata/pydata-sphinx-theme",
    # "twitter_url": "https://twitter.com/PyData",
    "logo": {
        #"text": "skscope",
        "image_dark": "_static/skscope-dark.png",
        "alt_text": "skscope",
    },
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["version-switcher", "navbar-nav"],
    # "announcement": """<div class="sidebar-message">
    # skscope is a optimization tool for Python. If you'd like to contribute, <a href="https://github.com/abess-team/skscope">check out our GitHub repository</a>
    # Your contributions are welcome!
    # </div>""",
    # "show_nav_level": 2,
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "article_footer_items": ["test.html", "test.html"],
    # "content_footer_items": ["test.html", "test.html"],
    # "footer_start": ["test.html", "test.html"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    # "search_bar_position": "navbar",  # TODO: Deprecated - remove in future version
}

html_context = {
    "github_user": "abess-team",
    "github_repo": "skscope",
    "github_version": "master",
    "doc_path": "docs",
}

#rediraffe_redirects = {
#    "contributing.rst": "community/index.rst",
#}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom-icon.js"]
todo_include_todos = True

# see https://sphinx-favicon.readthedocs.io for more information about the
# sphinx-favicon extension

html_sidebars = {
    "userguide/**": [
        "search-field",
        "sidebar-nav-bs",
    ],  # This ensures we test for custom sidebars
    "feature/**": [
        "search-field",
        "sidebar-nav-bs",
    ], 
    "autoapi/**": [
        "search-field",
        "sidebar-nav-bs",
    ], 
    "contribute/**": [
        "search-field",
        "sidebar-nav-bs",
    ], 
    #"examples/no-sidebar": [],  # Test what page looks like with no sidebar items
    #"examples/persistent-search-field": ["search-field"],
    # Blog sidebars
    # ref: https://ablog.readthedocs.io/manual/ablog-configuration-options/#blog-sidebars
    #"examples/blog/*": [
    #    "ablog/postcard.html",
    #    "ablog/recentposts.html",
    #    "ablog/tagcloud.html",
    #    "ablog/categories.html",
    #    "ablog/authors.html",
    #    "ablog/languages.html",
    #    "ablog/locations.html",
    #    "ablog/archives.html",
    #],
}

# nb_execution_mode = "off"
# nb_render_markdown_format = "myst"


#nbsphinx_execute = 'never'  # 每次构建时都执行 Jupyter Notebook
#nbsphinx_allow_errors = True  # 允许出现错误时继续构建
#nbsphinx_requirejs_path = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.1/require.min.js'  # 指定 require.js 的路径

#myst_url_schemes = ("http", "https", "mailto", "#")

# nbsphinx_assume_equations = True

autoapi_dirs = ['../../']
#autoapi_ignore = ['test_*', 'util*']
#autoapi_add_toctree_entry = False
#autoapi_options = [ 'members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'imported-members', ]
autoapi_template_dir = "_templates/autoapi/"
autoapi_generate_api_docs = False
#autoapi_keep_files = True
#autoapi_root = 'api'

def setup_to_main(
    app: Sphinx, pagename: str, templatename: str, context, doctree
) -> None:
    """Add a function that jinja can access for returning an "edit this page" link pointing to `main`."""

    def to_main(link: str) -> str:
        """Transform "edit on github" links and make sure they always point to the main branch.

        Args:
            link: the link to the github edit interface

        Returns:
            the link to the tip of the main branch for the same file
        """
        links = link.split("/")
        idx = links.index("docs")
        return "/".join(links[: idx + 1]) + "/source/" + "/".join(links[idx + 1 :])

    context["to_main"] = to_main


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application
    Returns:
        the 2 parralel parameters set to ``True``.
    """
    app.connect("html-page-context", setup_to_main)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }