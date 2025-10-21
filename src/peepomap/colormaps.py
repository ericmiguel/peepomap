"""Colormap definitions and registry for peepomap.

This module contains all colormap definitions with their metadata,
including type classification and descriptions.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from matplotlib.colors import LinearSegmentedColormap


ColormapType = Literal["sequential", "diverging", "cyclic", "multi-diverging"]


@dataclass
class ColormapInfo:
    """Colormap metadata.

    Attributes
    ----------
    name : str
        Colormap name
    colors : list[list[float]]
        RGB color values (0-1 range)
    cmap_type : ColormapType
        Colormap type (sequential, diverging, cyclic, multi-diverging)
    description : str
        Brief description
    tags : set[str]
        Auto-generated accessibility tags (colorblind-safe, perceptually-uniform, etc.)
    """

    name: str
    colors: list[list[float]]
    cmap_type: ColormapType
    description: str
    tags: set[str] = field(default_factory=set[str])


# Colormap definitions
_COLORMAPS_DATA: dict[str, ColormapInfo] = {
    "storm": ColormapInfo(
        name="storm",
        colors=[
            [0.2, 0.2, 0.7],
            [0.4, 0.3, 0.9],
            [0.5, 0.5, 1.0],
            [0.7, 0.6, 1.0],
            [0.8, 0.8, 1.0],
            [1.0, 0.9, 0.9],
            [1.0, 0.6, 0.6],
            [0.9, 0.4, 0.4],
            [0.8, 0.2, 0.2],
            [0.6, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, 0.6, 0.0],
            [1.0, 0.8, 0.2],
            [1.0, 0.9, 0.5],
            [1.0, 1.0, 0.7],
            [0.9, 1.0, 1.0],
            [0.6, 0.8, 1.0],
            [0.3, 0.6, 1.0],
            [0.2, 0.5, 0.9],
            [0.1, 0.4, 0.8],
            [0.0, 0.3, 0.7],
            [0.0, 0.5, 0.0],
            [0.1, 0.7, 0.1],
            [0.3, 0.9, 0.3],
            [0.6, 1.0, 0.5],
            [0.8, 1.0, 0.7],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.7, 0.7, 0.7],
            [0.9, 0.9, 0.9],
        ],
        cmap_type="sequential",
        description="Multi-hue sequential colormap for storm/weather visualization",
    ),
    "avanti": ColormapInfo(
        name="avanti",
        colors=[
            [0.984313725490196, 0.2549019607843137, 0.25882352941176473],
            [0.5803921568627451, 0.21568627450980393, 0.4235294117647059],
            [0.807843137254902, 0.4588235294117647, 0.6784313725490196],
            [0.4627450980392157, 0.7411764705882353, 0.8117647058823529],
            [0.615686274509804, 0.8117647058823529, 0.9411764705882353],
            [1.0, 1.0, 1.0],
        ],
        cmap_type="sequential",
        description="Red-purple-blue sequential colormap",
    ),
    "jazz": ColormapInfo(
        name="jazz",
        colors=[
            [0.2235294117647059, 0.14901960784313725, 0.5098039215686274],
            [0.47843137254901963, 0.22745098039215686, 0.6039215686274509],
            [0.24705882352941178, 0.5254901960784314, 0.7372549019607844],
            [0.1568627450980392, 0.6784313725490196, 0.6588235294117647],
            [0.5137254901960784, 0.8666666666666667, 0.8784313725490196],
        ],
        cmap_type="sequential",
        description="Purple-blue-cyan sequential colormap",
    ),
    "ons": ColormapInfo(
        name="ons",
        colors=[
            [1.0, 1.0, 1.0],
            [0.8823529411764706, 1.0, 1.0],
            [0.7058823529411765, 0.9411764705882353, 0.9803921568627451],
            [0.5882352941176471, 0.8235294117647058, 0.9803921568627451],
            [0.1568627450980392, 0.5098039215686274, 0.9411764705882353],
            [0.0784313725490196, 0.39215686274509803, 0.8235294117647058],
            [0.403921568627451, 0.996078431372549, 0.5215686274509804],
            [0.09411764705882353, 0.8431372549019608, 0.023529411764705882],
            [0.11764705882352941, 0.7058823529411765, 0.11764705882352941],
            [1.0, 0.9098039215686274, 0.47058823529411764],
            [1.0, 0.7529411764705882, 0.23529411764705882],
            [1.0, 0.3764705882352941, 0.0],
            [0.8823529411764706, 0.0784313725490196, 0.0],
            [0.984313725490196, 0.3686274509803922, 0.4196078431372549],
            [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
        ],
        cmap_type="multi-diverging",
        description="Multi-color multi-diverging colormap for categorical data",
    ),
    "plus": ColormapInfo(
        name="plus",
        colors=[
            [0.10588235294117647, 0.25882352941176473, 0.2784313725490196],
            [0.03529411764705882, 0.592156862745098, 0.6078431372549019],
            [0.4588235294117647, 0.8470588235294118, 0.8352941176470589],
            [1.0, 0.7529411764705882, 0.796078431372549],
            [0.996078431372549, 0.4980392156862745, 0.615686274509804],
            [0.396078431372549, 0.19607843137254902, 0.24313725490196078],
        ],
        cmap_type="diverging",
        description="Cyan-white-pink diverging colormap",
    ),
    "tok": ColormapInfo(
        name="tok",
        colors=[
            [1.0, 1.0, 1.0],
            [0.7843137254901961, 0.9568627450980393, 0.9921568627450981],
            [0.6352941176470588, 0.8627450980392157, 0.9686274509803922],
            [0.3686274509803922, 0.6039215686274509, 0.8235294117647058],
            [0.0392156862745098, 0.43137254901960786, 1.0],
            [0.6509803921568628, 0.9647058823529412, 0.0196078431372549],
            [0.3411764705882353, 0.8117647058823529, 0.08235294117647059],
            [0.13333333333333333, 0.5882352941176471, 0.19215686274509805],
            [0.12941176470588237, 0.3607843137254902, 0.27058823529411763],
            [1.0, 1.0, 0.29411764705882354],
            [0.9764705882352941, 0.7764705882352941, 0.3058823529411765],
            [0.9607843137254902, 0.4549019607843137, 0.17647058823529413],
            [0.8392156862745098, 0.1450980392156863, 0.1568627450980392],
            [0.9254901960784314, 0.8156862745098039, 0.9882352941176471],
            [0.796078431372549, 0.49411764705882355, 0.9647058823529412],
            [0.5843137254901961, 0.058823529411764705, 0.8745098039215686],
            [0.4196078431372549, 0.07058823529411765, 0.5725490196078431],
        ],
        cmap_type="multi-diverging",
        description="Multi-color multi-diverging colormap with vibrant hues",
    ),
    "vapor": ColormapInfo(
        name="vapor",
        colors=[
            [0.5803921568627451, 0.8156862745098039, 1.0],
            [0.5294117647058824, 0.5843137254901961, 0.9098039215686274],
            [0.5882352941176471, 0.4196078431372549, 1.0],
            [0.6784313725490196, 0.5490196078431373, 1.0],
            [0.7803921568627451, 0.4549019607843137, 0.9098039215686274],
            [0.7803921568627451, 0.4549019607843137, 0.6627450980392157],
            [1.0, 0.41568627450980394, 0.8352941176470589],
            [1.0, 0.41568627450980394, 0.5450980392156862],
            [1.0, 0.5450980392156862, 0.5450980392156862],
            [1.0, 0.6470588235294118, 0.5450980392156862],
            [1.0, 0.8705882352941177, 0.5450980392156862],
            [0.803921568627451, 0.8705882352941177, 0.5450980392156862],
            [0.5450980392156862, 0.8705882352941177, 0.5450980392156862],
            [0.12549019607843137, 0.8705882352941177, 0.5450980392156862],
        ],
        cmap_type="sequential",
        description="Blue-magenta-pink-peach sequential colormap (vaporwave aesthetic)",
    ),
}

# Create matplotlib colormaps
_COLORMAPS: dict[str, LinearSegmentedColormap] = {}

for info in _COLORMAPS_DATA.values():
    _COLORMAPS[info.name] = LinearSegmentedColormap.from_list(info.name, info.colors)
    _COLORMAPS[f"{info.name}_r"] = LinearSegmentedColormap.from_list(
        f"{info.name}_r", info.colors[::-1]
    )


def get(name: str) -> LinearSegmentedColormap:
    """Get a colormap by name.

    Parameters
    ----------
    name : str
        Colormap name (append '_r' for reversed). Works with both peepomap
        and matplotlib colormaps.

    Returns
    -------
    LinearSegmentedColormap
        The requested colormap

    Raises
    ------
    KeyError
        If colormap not found

    Examples
    --------
    >>> cmap = peepomap.get("storm")
    >>> cmap_r = peepomap.get("storm_r")
    """
    # First try peepomap colormaps
    if name in _COLORMAPS:
        return _COLORMAPS[name]

    # Then try matplotlib colormaps
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        # Get matplotlib colormap
        mpl_cmap = plt.get_cmap(name)

        # Convert to LinearSegmentedColormap if it isn't already
        if isinstance(mpl_cmap, LinearSegmentedColormap):
            return mpl_cmap
        else:
            # Convert other colormap types (like ListedColormap) to LinearSegmentedColormap
            colors = mpl_cmap(np.linspace(0, 1, 256))
            return LinearSegmentedColormap.from_list(f"mpl_{name}", colors)

    except (ValueError, KeyError):
        # If matplotlib doesn't have it either, show available options
        available_peepo = ", ".join(sorted(_COLORMAPS.keys()))

        # Get matplotlib colormap names dynamically
        try:
            import matplotlib.pyplot as plt

            mpl_cmaps = sorted(plt.colormaps())
            # Show first 20 for brevity, then indicate there are more
            if len(mpl_cmaps) > 20:
                available_mpl = (
                    ", ".join(mpl_cmaps[:20]) + f", ... ({len(mpl_cmaps) - 20} more)"
                )
            else:
                available_mpl = ", ".join(mpl_cmaps)
        except Exception:
            available_mpl = "viridis, plasma, inferno, magma, ..."

        msg = (
            f'Colormap "{name}" not found.\n'
            f"Peepomap colormaps: {available_peepo}\n"
            f"Matplotlib colormaps: {available_mpl}"
        )
        raise KeyError(msg)


def get_info(name: str) -> ColormapInfo:
    """Get colormap metadata.

    Parameters
    ----------
    name : str
        Colormap name (without '_r' suffix)

    Returns
    -------
    ColormapInfo
        Colormap metadata

    Raises
    ------
    KeyError
        If colormap not found
    """
    if name not in _COLORMAPS_DATA:
        available = ", ".join(sorted(_COLORMAPS_DATA.keys()))
        msg = f'Colormap "{name}" not found. Available: {available}'
        raise KeyError(msg)
    return _COLORMAPS_DATA[name]


def list_colormaps(
    cmap_type: ColormapType | None = None,
    tags: set[str] | None = None,
    matplotlib: bool = False,
) -> list[str]:
    """List available colormaps with optional filtering.

    Parameters
    ----------
    cmap_type : ColormapType | None, optional
        Filter by type (sequential, diverging, cyclic, multi-diverging)
    tags : set[str] | None, optional
        Filter by tags (colorblind-safe, perceptually-uniform, high-contrast)
    matplotlib : bool, default=False
        If True, include matplotlib colormaps

    Returns
    -------
    list[str]
        Colormap names (without '_r' suffix)

    Examples
    --------
    >>> cmaps = peepomap.list_colormaps()
    >>> seq_cmaps = peepomap.list_colormaps(cmap_type="sequential")
    >>> safe_cmaps = peepomap.list_colormaps(tags={"colorblind-safe"})
    """
    cmaps = _COLORMAPS_DATA.items()

    # Filter by type
    if cmap_type is not None:
        cmaps = [(name, info) for name, info in cmaps if info.cmap_type == cmap_type]

    # Filter by tags (must have ALL specified tags)
    if tags is not None:
        cmaps = [(name, info) for name, info in cmaps if tags.issubset(info.tags)]

    # Get peepomap colormap names
    peepomap_names = sorted(name for name, _ in cmaps)

    # If matplotlib is True, add matplotlib colormaps
    if matplotlib:
        matplotlib_names = list_matplotlib_colormaps()
        # Filter out any reversed versions (_r) from matplotlib to match peepomap behavior
        matplotlib_names = [
            name for name in matplotlib_names if not name.endswith("_r")
        ]
        # Combine and sort, removing duplicates
        all_names = sorted(set(peepomap_names + matplotlib_names))
        return all_names

    return peepomap_names


def list_matplotlib_colormaps() -> list[str]:
    """List all matplotlib colormaps.

    Returns
    -------
    list[str]
        Sorted matplotlib colormap names
    """
    try:
        import matplotlib.pyplot as plt

        return sorted(plt.colormaps())
    except ImportError:
        return []


# Public API
cmaps = _COLORMAPS
colormap_registry = _COLORMAPS_DATA


__all__ = [
    "ColormapInfo",
    "ColormapType",
    "cmaps",
    "colormap_registry",
    "get",
    "get_info",
    "list_colormaps",
    "list_matplotlib_colormaps",
]
