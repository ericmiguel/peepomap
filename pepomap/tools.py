"""Tools fro creating colormaps."""

from typing import Iterable
from typing import List
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def hex_to_rgb(value: str) -> List[int]:
    """
    Converts hex to rgb colours.

    Parameters
    ----------
    value : str
        6 characters representing a hex colour.

    Returns
    -------
    List[int]
        list length 3 of RGB values
    """
    value = value.strip("#")
    lv = len(value)
    return [int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)]


def rgb_to_dec(value: Iterable[int]) -> List[float]:
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)

    Parameters
    ----------
    value : Iterable[int]
        RGB values

    Returns
    -------
    List[float]
        list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def create_cmap_from_hex(
    hex_list: List[str], name: str = "new_cmap"
) -> mcolors.LinearSegmentedColormap:
    """
    Creates a color map from a list of hex codes.

    Parameters
    ----------
    hex_list : List[str]
        a list of hex codes
    name : str, optional
        name of the new cmap, by default "new_cmap"

    Returns
    -------
    mcolors.LinearSegmentedColormap
        [description]
    """

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap(name, segmentdata=cdict, N=256)
    return cmp


def colortable(hex_list: Iterable[str]) -> Tuple[Figure, Axes]:
    """
    Given a list of HEX strings display the color as a cell.

    Parameters
    ----------
    hex_list : Iterable[str]
        list of hex codes

    Returns
    -------
    Tuple[Figure, Axes]
        figure and axes of the plot
    """
    cell_width = 150
    cell_height = 50
    swatch_width = 48
    margin = 0
    topmargin = 40

    rgb_list = [hex_to_rgb(value) for value in hex_list]
    dec_list = [rgb_to_dec(value) for value in rgb_list]
    names = [
        f"HEX: {col[0]}\nRGB: {col[1]}" for col in zip(hex_list, rgb_list, dec_list)
    ]
    n = len(names)
    ncols = 4
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 8 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - topmargin) / height,
    )
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            name,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.hlines(y, swatch_start_x, swatch_end_x, color=dec_list[i], linewidth=18)

    return fig, ax
