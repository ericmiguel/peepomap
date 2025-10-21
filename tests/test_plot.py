"""Tests for peepomap.plot module."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pytest

import peepomap


class TestShowColormaps:
    """Test show_colormaps function."""

    def test_show_all_colormaps(self):
        """Test showing all peepomap colormaps."""
        fig, axes = peepomap.show_colormaps()
        assert isinstance(fig, Figure)
        assert isinstance(axes, list)
        assert len(axes) > 0
        assert all(isinstance(ax, Axes) for ax in axes)
        plt.close(fig)

    def test_show_specific_colormaps(self):
        """Test showing specific colormaps."""
        fig, axes = peepomap.show_colormaps(["storm", "vapor"])
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_show_colormaps_with_objects(self):
        """Test showing colormaps using colormap objects."""
        cmap = peepomap.get("storm")
        fig, axes = peepomap.show_colormaps([cmap, "vapor"])
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_show_colormaps_custom_figsize(self):
        """Test showing colormaps with custom figure size."""
        fig, _ = peepomap.show_colormaps(["storm"], figsize=(10, 5))
        assert isinstance(fig, Figure)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 5
        plt.close(fig)

    def test_show_colormaps_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            peepomap.show_colormaps([])
        assert "No colormaps provided" in str(exc_info.value)


class TestCompareColormaps:
    """Test compare_colormaps function."""

    def test_compare_two_colormaps(self):
        """Test comparing two colormaps."""
        fig, axes = peepomap.compare_colormaps("storm", "vapor")
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_compare_three_colormaps(self):
        """Test comparing three colormaps."""
        fig, axes = peepomap.compare_colormaps("storm", "vapor", "jazz")
        assert isinstance(fig, Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_compare_with_objects(self):
        """Test comparing using colormap objects."""
        cmap = peepomap.get("storm")
        fig, axes = peepomap.compare_colormaps(cmap, "vapor")
        assert isinstance(fig, Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_compare_no_colormaps(self):
        """Test that comparing no colormaps raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            peepomap.compare_colormaps()
        assert "At least one colormap must be provided" in str(exc_info.value)


class TestPlotColormapProperties:
    """Test plot_colormap_properties function."""

    def test_plot_properties_by_name(self):
        """Test plotting colormap properties by name."""
        fig, axes = peepomap.plot_colormap_properties("storm")
        assert isinstance(fig, Figure)
        assert isinstance(axes, dict)
        assert "colormap" in axes
        assert "rgb" in axes
        assert "lightness" in axes
        plt.close(fig)

    def test_plot_properties_by_object(self):
        """Test plotting colormap properties by object."""
        cmap = peepomap.get("storm")
        fig, axes = peepomap.plot_colormap_properties(cmap)
        assert isinstance(fig, Figure)
        assert isinstance(axes, dict)
        plt.close(fig)

    def test_plot_properties_custom_figsize(self):
        """Test plotting with custom figure size."""
        fig, _ = peepomap.plot_colormap_properties("storm", figsize=(10, 6))
        assert isinstance(fig, Figure)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)
