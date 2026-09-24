from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, os.fspath(SRC))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent / "_ext"))

project = "ddmo"
author = "Ahmed H. Bayoumy"
copyright = "2026, Ahmed H. Bayoumy"

from ddmo import __version__

release = __version__
version = __version__

extensions = [
    "algorithm",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Number algorithms ("Algorithm 1", "Algorithm 2", ...) continuously across pages.
numfig = True
numfig_secnum_depth = 0
numfig_format = {"algorithm": "Algorithm %s"}

pygments_style = "friendly"
pygments_dark_style = "monokai"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = ["pandas"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
]

html_theme = "furo"
html_title = "ddmo"
html_short_title = "ddmo"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700"
    "&family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&display=swap",
    "custom.css",
]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view"],
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0e7490",
        "color-brand-visited": "#0e7490",
        "color-api-name": "#0f172a",
        "color-api-pre-name": "#0f766e",
        "color-admonition-title--note": "#0e7490",
        "color-admonition-title-background--note": "rgba(14, 116, 144, 0.1)",
        "font-stack": "IBM Plex Sans, Segoe UI, system-ui, sans-serif",
        "font-stack--monospace": "IBM Plex Mono, Consolas, monospace",
        "font-stack--headings": "IBM Plex Sans, Segoe UI, system-ui, sans-serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#67e8f9",
        "color-brand-visited": "#67e8f9",
        "color-api-name": "#e2e8f0",
        "color-api-pre-name": "#5eead4",
        "font-stack": "IBM Plex Sans, Segoe UI, system-ui, sans-serif",
        "font-stack--monospace": "IBM Plex Mono, Consolas, monospace",
        "font-stack--headings": "IBM Plex Sans, Segoe UI, system-ui, sans-serif",
    },
}

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}