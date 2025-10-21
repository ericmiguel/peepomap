"""Visualization functions for peepomap colormaps.

This module provides functions to display, compare, and analyze colormaps.
"""

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from peepomap.colormaps import get
from peepomap.colormaps import list_colormaps
from peepomap.tools import rgb_to_lab_l


def show_colormaps(
    colormaps: list[str | LinearSegmentedColormap] | None = None,
    *,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Display colormaps in a gallery view.

    Parameters
    ----------
    colormaps : list[str | LinearSegmentedColormap] | None, optional
        List of colormap names or objects. If None, shows all peepomap colormaps.
    figsize : tuple[float, float] | None, optional
        Figure size (auto-calculated if None)

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and axes

    Raises
    ------
    ValueError
        If no colormaps provided or list is empty

    Examples
    --------
    >>> fig, axes = peepomap.show_colormaps()
    >>> fig, axes = peepomap.show_colormaps(["storm", "vapor", "jazz"])
    >>> fig, axes = peepomap.show_colormaps([peepomap.get("storm"), "vapor"])
    """
    # If no colormaps provided, show all peepomap colormaps
    if colormaps is None:
        colormaps = list(list_colormaps())

    if not colormaps:
        msg = "No colormaps provided"
        raise ValueError(msg)

    # Create gradient for display
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Calculate figure size
    nrows = len(colormaps)
    if figsize is None:
        figh = (0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22) * 2
        figsize = (12, figh)

    # Create figure
    fig, axs = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=nrows,
        figsize=figsize,
        squeeze=False,
    )
    axs = axs.flatten()

    fig.subplots_adjust(
        top=1 - 0.35 / figsize[1],
        bottom=0.15 / figsize[1],
        left=0.2,
        right=0.99,
    )

    # Plot each colormap
    for ax, cmap_input in zip(axs, colormaps, strict=True):
        # Handle both string names and colormap objects
        if isinstance(cmap_input, str):
            cmap = get(cmap_input)
            cmap_name = cmap_input
        else:
            cmap = cmap_input
            cmap_name = getattr(cmap_input, "name", "custom")

        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.text(
            -0.01,
            0.5,
            cmap_name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    fig.tight_layout()

    return fig, list(axs)


def compare_colormaps(
    *colormaps: str | LinearSegmentedColormap,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Figure, list[Axes]]:
    """Compare colormaps side by side.

    Parameters
    ----------
    *colormaps : str or LinearSegmentedColormap
        Colormap names or objects
    figsize : tuple[float, float], default=(12, 6)
        Figure size

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and axes

    Raises
    ------
    ValueError
        If no colormaps provided

    Examples
    --------
    >>> fig, axes = peepomap.compare_colormaps("storm", "vapor", "jazz")
    >>> fig, axes = peepomap.compare_colormaps(peepomap.get("storm"), "vapor")
    """
    if not colormaps:
        msg = "At least one colormap must be provided"
        raise ValueError(msg)

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    nrows = len(colormaps)
    fig, axs = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=nrows, figsize=figsize, squeeze=False
    )
    axs = axs.flatten()

    for ax, cmap_input in zip(axs, colormaps, strict=True):
        # Handle both string names and colormap objects
        if isinstance(cmap_input, str):
            cmap = get(cmap_input)
            cmap_name = cmap_input
        else:
            cmap = cmap_input
            cmap_name = getattr(cmap_input, "name", "custom")

        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.text(
            -0.01,
            0.5,
            cmap_name,
            va="center",
            ha="right",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    fig.tight_layout()

    return fig, list(axs)


def plot_colormap_properties(
    colormap: str | LinearSegmentedColormap,
    *,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[Figure, dict[str, Axes]]:
    """Plot colormap with RGB and perceptual lightness analysis.

    Parameters
    ----------
    colormap : str or LinearSegmentedColormap
        Colormap name or object
    figsize : tuple[float, float], default=(12, 8)
        Figure size

    Returns
    -------
    tuple[Figure, dict[str, Axes]]
        Figure and axes dictionary

    Examples
    --------
    >>> fig, axes = peepomap.plot_colormap_properties("storm")
    >>> fig, axes = peepomap.plot_colormap_properties(peepomap.get("storm"))
    """
    # Handle both string names and colormap objects
    if isinstance(colormap, str):
        cmap = get(colormap)
        cmap_name = colormap
    else:
        cmap = colormap
        cmap_name = getattr(colormap, "name", "custom")

    # Sample the colormap
    n_samples = 256
    x = np.linspace(0, 1, n_samples)
    rgb = cmap(x)[:, :3]  # Drop alpha channel

    # Calculate perceptual lightness using CIELAB L*
    # This is the proper way to measure perceptual uniformity

    # Calculate L* for all colors
    lightness = np.array([rgb_to_lab_l(color) for color in rgb])

    # Create subplots - always show colormap, RGB, and lightness (3 plots)
    n_plots = 3
    fig, axs_list = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        n_plots, 1, figsize=figsize
    )

    axes_dict: dict[str, Axes] = {}
    idx = 0

    # Plot colormap
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax = axs_list[idx]
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
    ax.set_title(f"Colormap: {cmap_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Data Value")
    ax.set_yticks([])
    axes_dict["colormap"] = ax
    idx += 1

    # Plot RGB channels
    ax = axs_list[idx]
    ax.plot(x, rgb[:, 0], "r-", label="Red", linewidth=2)
    ax.plot(x, rgb[:, 1], "g-", label="Green", linewidth=2)
    ax.plot(x, rgb[:, 2], "b-", label="Blue", linewidth=2)
    ax.set_title("RGB Channels", fontsize=11)
    ax.set_xlabel("Data Value")
    ax.set_ylabel("Intensity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    axes_dict["rgb"] = ax
    idx += 1

    # Plot perceptual lightness
    ax = axs_list[idx]
    ax.plot(x, lightness, "k-", linewidth=2)
    ax.set_title("Perceptual Lightness (CIELAB L*)", fontsize=11)
    ax.set_xlabel("Data Value")
    ax.set_ylabel("L* (0-100)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    # Add horizontal line at L*=50 for reference
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes_dict["lightness"] = ax

    fig.tight_layout()
    return fig, axes_dict


__all__ = [
    "compare_colormaps",
    "plot_colormap_properties",
    "show_colormaps",
]
