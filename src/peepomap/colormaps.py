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
    "Tria": ColormapInfo(
        name="Tria",
        colors=[
            [1.0, 1.0, 1.0],
            [0.9003921568627451, 0.9003921568627451, 0.9003921568627451],
            [0.7753940792003076, 0.7753940792003076, 0.7753940792003076],
            [0.6757862360630527, 0.6757862360630527, 0.6757862360630527],
            [0.5507881584006151, 0.5507881584006151, 0.5507881584006151],
            [0.5089324618736384, 0.5498910675381263, 0.5089324618736384],
            [0.5228758169934641, 0.6457516339869281, 0.5228758169934641],
            [0.5403050108932462, 0.7655773420479303, 0.5403050108932462],
            [0.5577342047930284, 0.8854030501089325, 0.5577342047930284],
            [0.5381314878892733, 0.913033448673587, 0.5381314878892733],
            [0.396401384083045, 0.8047673971549404, 0.396401384083045],
            [0.2546712802768166, 0.6965013456362937, 0.2546712802768166],
            [0.13951557093425607, 0.6085351787773933, 0.13951557093425607],
            [0.0, 0.5019607843137255, 0.0],
            [0.15076252723311548, 0.5786492374727669, 0.20043572984749455],
            [0.30152505446623096, 0.6553376906318082, 0.4008714596949891],
            [0.48997821350762527, 0.7511982570806099, 0.6514161220043573],
            [0.6784313725490196, 0.8470588235294118, 0.9019607843137255],
            [0.5720107650903499, 0.7141868512110726, 0.8459823144944252],
            [0.39375624759707806, 0.49162629757785464, 0.7522183775470972],
            [0.21284121491733948, 0.26574394463667816, 0.6570549788542868],
            [0.06917339484813534, 0.08636678200692044, 0.5814840445982314],
            [0.1111111111111111, 0.1111111111111111, 0.5821350762527232],
            [0.3888888888888889, 0.3888888888888889, 0.6747276688453159],
            [0.6111111111111112, 0.6111111111111112, 0.74880174291939],
            [0.8888888888888888, 0.8888888888888888, 0.8413943355119825],
            [0.9716109188773548, 0.9516186082276048, 0.7932641291810842],
            [0.9126489811610919, 0.8511341791618607, 0.6163783160322953],
            [0.8384006151480201, 0.7245982314494426, 0.39363321799307965],
            [0.779438677431757, 0.6241138023836985, 0.21674740484429064],
            [0.7215686274509804, 0.5254901960784314, 0.043137254901960784],
            [0.7989106753812636, 0.5886710239651416, 0.2522875816993464],
            [0.8607843137254902, 0.6392156862745098, 0.41960784313725485],
            [0.9381263616557735, 0.7023965141612201, 0.6287581699346405],
            [1.0, 0.7529411764705882, 0.796078431372549],
            [0.9054517493271819, 0.5964475201845444, 0.6306189926951172],
            [0.7841445597846981, 0.39566320645905423, 0.41833141099577087],
            [0.6646212995001922, 0.19783160322952711, 0.20916570549788543],
            [0.5682891195693963, 0.03838523644752018, 0.04058439061899266],
            [0.6209150326797386, 0.11895424836601307, 0.1261437908496732],
            [0.747276688453159, 0.3172113289760348, 0.3363834422657952],
            [0.8483660130718954, 0.47581699346405226, 0.5045751633986928],
            [0.9747276688453159, 0.674074074074074, 0.7148148148148148],
            [0.928642829680892, 0.6017685505574779, 0.7236447520184544],
            [0.832310649750096, 0.45062668204536716, 0.6788004613610149],
            [0.7127873894655901, 0.2630988081507113, 0.6231603229527104],
            [0.5914801999231065, 0.07277201076509038, 0.5666897347174163],
            [0.5636165577342047, 0.05555555555555555, 0.5703703703703703],
            [0.6562091503267973, 0.3333333333333333, 0.6967320261437908],
            [0.74880174291939, 0.6111111111111112, 0.8230936819172113],
            [0.8228758169934641, 0.8333333333333334, 0.9241830065359478],
            [0.8336485966935794, 0.9768089196462899, 0.9768089196462899],
            [0.6028450595924644, 0.8572856593617839, 0.8572856593617839],
            [0.4168242983467897, 0.760953479430988, 0.760953479430988],
            [0.18257593233371777, 0.6396462898885044, 0.6396462898885044],
            [0.0, 0.5450980392156862, 0.5450980392156862],
        ],
        cmap_type="sequential",
        description="",
        tags=set(),
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
