"""Generate colormap demo images for README."""

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import peepomap


STATIC_DIR = Path(__file__).parents[2].resolve() / "static"


def save_demo_images(
    base_name: str,
    colormaps: list[str | LinearSegmentedColormap] | None = None,
) -> None:
    """Save light and dark background versions of colormap demos.

    Parameters
    ----------
    base_name : str
        Base filename (without _lightbg/_darkbg suffix and extension)
    colormaps : list[str | LinearSegmentedColormap] | None, optional
        List of colormaps to display. If None, shows all peepomap colormaps.
    """
    lightbg_output_path = STATIC_DIR / f"{base_name}_light.png"
    darkbg_output_path = STATIC_DIR / f"{base_name}_dark.png"

    # Light background version
    fig, _ = peepomap.show_colormaps(colormaps)
    fig.savefig(  # pyright: ignore[reportUnknownMemberType]
        lightbg_output_path, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    # Dark background version
    with plt.rc_context({  # pyright: ignore[reportUnknownMemberType]
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "figure.facecolor": "none",
    }):
        fig, _ = peepomap.show_colormaps(colormaps)
        fig.savefig(darkbg_output_path, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)

    print(f"✓ Generated {lightbg_output_path}")
    print(f"✓ Generated {darkbg_output_path}")


def plot_colormaps_demo() -> None:
    """Generate colormap demo images for README."""
    save_demo_images("colormaps")


def plot_combine_demo() -> None:
    """Generate colormap combination demo image for README."""
    blues = peepomap.get("Blues")
    reds = peepomap.get("Reds")
    combined_cmap = peepomap.combine(blues, reds, weights=[0.4, 0.6], name="Wines")

    save_demo_images("combine_demo", [blues, reds, combined_cmap])


def plot_create_linear_demo() -> None:
    """Generate create_linear demo image for README."""
    ocean_sunset = peepomap.create_linear("navy", "crimson", name="Ocean Sunset")

    save_demo_images("create_linear_demo", [ocean_sunset])


def plot_create_diverging_demo() -> None:
    """Generate create_diverging demo image for README."""
    cool_warm = peepomap.create_diverging("Blues_r", "Reds", name="Cool Warm")
    rdylbl = peepomap.create_diverging(
        "Reds_r", "Blues", center="yellow", blend=0.3, name="RdYlBl"
    )

    save_demo_images("create_diverging_demo", [cool_warm, rdylbl])


def plot_create_concat_demo() -> None:
    """Generate concat demo images for README."""
    div1 = peepomap.create_linear("blue", "red", name="div1")
    div2 = peepomap.create_linear("purple", "orange", name="div2")
    combined = peepomap.concat(div1, div2, blend=0.25, n=512, name="Fusion")

    save_demo_images("concat_demo", [div1, div2, combined])


def plot_create_concat_odd() -> None:
    """Generate odd concat demo images for README."""
    sunset = peepomap.create_linear("gold", "orangered", name="Sunset", reverse=True)
    tab20b = peepomap.get("tab20b")
    odd = peepomap.concat(sunset, tab20b, blend=0.25, name="Odd1")

    save_demo_images("concat_odd_demo", [sunset, tab20b, odd])


def plot_adjust_demo() -> None:
    """Generate adjust demo images for README."""
    original = peepomap.get("storm")
    saturated = peepomap.adjust("storm", saturation=1.8, cmap_name="Storm Saturated")
    desaturated = peepomap.adjust(
        "storm", saturation=0.3, cmap_name="Storm Desaturated"
    )
    brighter = peepomap.adjust("storm", lightness=1.4, cmap_name="Storm Brighter")
    blue_boosted = peepomap.adjust(
        "storm", blue_boost=0.3, cmap_name="Storm Blue Boost"
    )

    save_demo_images(
        "adjust_demo",
        [original, saturated, desaturated, brighter, blue_boosted],
    )


def plot_truncate_demo() -> None:
    """Generate truncate demo images for README."""
    original = peepomap.get("vapor")
    first_half = peepomap.truncate("vapor", 0.0, 0.5, cmap_name="Vapor First Half")
    second_half = peepomap.truncate("vapor", 0.5, 1.0, cmap_name="Vapor Second Half")
    middle = peepomap.truncate("vapor", 0.25, 0.75, cmap_name="Vapor Middle")

    save_demo_images("truncate_demo", [original, first_half, second_half, middle])


def plot_shift_demo() -> None:
    """Generate shift demo images for README."""
    original = peepomap.get("hsv")
    shift_25 = peepomap.shift("hsv", start=0.25, cmap_name="HSV Shift 0.25")
    shift_50 = peepomap.shift("hsv", start=0.5, cmap_name="HSV Shift 0.50")
    shift_75 = peepomap.shift("hsv", start=0.75, cmap_name="HSV Shift 0.75")

    save_demo_images("shift_demo", [original, shift_25, shift_50, shift_75])


def plot_complex_concat_demo() -> None:
    """Generate complex concat demo images for README."""
    greys = peepomap.create_linear("white", "grey", name="Greys")
    greens = peepomap.create_linear("lightgreen", "green", name="Greens")
    blues = peepomap.create_linear("lightblue", "darkblue", name="Blues")
    goldens = peepomap.create_linear("lightyellow", "darkgoldenrod", name="Goldens")
    reds = peepomap.create_linear("pink", "darkred", name="Reds")
    pinks = peepomap.create_linear("lightpink", "darkmagenta", name="Pinks")
    cyans = peepomap.create_linear("lightcyan", "darkcyan", name="Cyans")

    tria = peepomap.concat(
        greys,
        greens,
        blues,
        goldens,
        reds,
        pinks,
        cyans,
        name="Tria",
        blend=0.45,
    )

    save_demo_images(
        "complex_concat_demo", [greys, greens, blues, goldens, reds, pinks, cyans, tria]
    )


if __name__ == "__main__":
    plot_colormaps_demo()
    plot_combine_demo()
    plot_create_linear_demo()
    plot_create_diverging_demo()
    plot_create_concat_demo()
    plot_create_concat_odd()
    plot_adjust_demo()
    plot_truncate_demo()
    plot_shift_demo()
    plot_complex_concat_demo()
