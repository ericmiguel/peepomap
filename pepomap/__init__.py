from pepomap import ons
from pepomap import storm
from pepomap import tools  # flake8: noqa


cmaps = {
    "storm": storm.cmap,
    "storm_r": storm.cmap_r,
    "ons": ons.cmap,
    "ons_r": ons.cmap_r,
}


__doc__ = """Just some extra Matplotlib colormaps."""


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = tools.display_colormaps(cmaps)
    fig.savefig("pepomap_colormaps_lightbg.png", bbox_inches="tight", facecolor="white")

    with plt.rc_context(
        {
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "figure.facecolor": "none",
        }
    ):
        fig, ax = tools.display_colormaps(cmaps)
        fig.savefig("pepomap_colormaps_darkbg.png", bbox_inches="tight")
