import matplotlib.pyplot as plt
import numpy as np


def display_colormaps(cmaps):
    # adapted from https://matplotlib.org/stable/tutorials/colors/colormaps.html
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    nrows = len(cmaps)
    figh = (0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22) * 2
    fig, axs = plt.subplots(figsize=(12, figh), nrows=nrows + 1)
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)

    for ax, (name, values) in zip(axs, cmaps.items()):
        ax.imshow(gradient, aspect="auto", cmap=values)
        ax.text(
            -0.01,
            0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    for ax in axs:
        ax.set_axis_off()

    fig.tight_layout()

    return fig, axs
