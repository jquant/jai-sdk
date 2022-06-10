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
import sphinx_book_theme

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "jai-sdk"
copyright = "2022, JQuant"
author = "JQuant"

from jai import __version__ as version

release = version
# The full version, including alpha/beta/rc tags
# release = 'v0.1.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "sphinx_copybutton", "sphinx.ext.autosectionlabel"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "source/jai.*.rst",
    "source/modules.rst",
    "setup.py",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = "source/images/jai_logo.png"
html_theme = "sphinx_book_theme"
html_title = f"JAI-SDK {release}"
html_theme_options = {
    # New theme
    "repository_url": "https://github.com/jquant/jai-sdk",
    "use_repository_button": True,
    "repository_branch": "main",
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "show_navbar_depth": 0,
}
# html_sidebars = {"**": ["sidebar-logo.html"]}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

add_module_names = False

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = True

autosectionlabel_prefix_document = True
