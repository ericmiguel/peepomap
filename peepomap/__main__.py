"""Plot colormap samples used in README."""

if __name__ == "__main__":
    from pathlib import Path as path

    import matplotlib.pyplot as plt

    from peepomap import cmaps
    from peepomap import tools

    base_output_path = path("samples/")
    lightbg_output_path = base_output_path / "pepomap_colormaps_lightbg.png"
    darkbg_output_path = base_output_path / "pepomap_colormaps_darkbg.png"

    fig, ax = tools.display_colormaps(cmaps)
    fig.savefig(str(lightbg_output_path), bbox_inches="tight", facecolor="white")

    with plt.rc_context(
        {
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "figure.facecolor": "none",
        }
    ):
        fig, ax = tools.display_colormaps(cmaps)
        fig.savefig(str(darkbg_output_path), bbox_inches="tight")
