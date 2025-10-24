"""Peepo-Powered Matplotlib colormaps and tools for scientific visualization.

Examples
--------
>>> import peepomap
>>> cmap = peepomap.get("storm")
>>> fig, axes = peepomap.show_colormaps()
"""

__version__ = "0.2.0"

# Core colormap registry
from peepomap.colormaps import ColormapInfo
from peepomap.colormaps import ColormapType
from peepomap.colormaps import cmaps
from peepomap.colormaps import colormap_registry
from peepomap.colormaps import get
from peepomap.colormaps import get_info
from peepomap.colormaps import list_colormaps
from peepomap.colormaps import list_matplotlib_colormaps

# Visualization functions
from peepomap.plot import compare_colormaps
from peepomap.plot import plot_colormap_properties
from peepomap.plot import show_colormaps

# Manipulation tools
from peepomap.tools import adjust
from peepomap.tools import combine
from peepomap.tools import concat
from peepomap.tools import create_diverging
from peepomap.tools import create_linear
from peepomap.tools import export
from peepomap.tools import hex_to_decimal_rgb
from peepomap.tools import reverse
from peepomap.tools import set_special_colors
from peepomap.tools import shift
from peepomap.tools import truncate


# ruff: noqa: RUF022
__all__ = [
    # Version
    "__version__",
    # Types
    "ColormapInfo",
    "ColormapType",
    # Core registry
    "cmaps",
    "colormap_registry",
    "get",
    "get_info",
    "list_colormaps",
    "list_matplotlib_colormaps",
    # Visualization
    "compare_colormaps",
    "plot_colormap_properties",
    "show_colormaps",
    # Manipulation tools
    "adjust",
    "combine",
    "concat",
    "create_diverging",
    "create_linear",
    "export",
    "hex_to_decimal_rgb",
    "reverse",
    "set_special_colors",
    "shift",
    "truncate",
]
