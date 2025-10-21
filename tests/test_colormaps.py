"""Tests for peepomap.colormaps module."""

from matplotlib.colors import LinearSegmentedColormap
import pytest

import peepomap


class TestColormapRegistry:
    """Test colormap registry and retrieval."""

    def test_get_existing_colormap(self):
        """Test retrieving an existing peepomap colormap."""
        cmap = peepomap.get("storm")
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.name == "storm"

    def test_get_reversed_colormap(self):
        """Test retrieving a reversed colormap."""
        cmap = peepomap.get("storm_r")
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.name == "storm_r"

    def test_get_matplotlib_colormap(self):
        """Test retrieving a matplotlib colormap."""
        cmap = peepomap.get("viridis")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_get_nonexistent_colormap(self):
        """Test that requesting non-existent colormap raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            peepomap.get("nonexistent_colormap")
        assert "not found" in str(exc_info.value)

    def test_list_colormaps(self):
        """Test listing all peepomap colormaps."""
        cmaps = peepomap.list_colormaps()
        assert isinstance(cmaps, list)
        assert len(cmaps) > 0
        assert "storm" in cmaps
        assert "vapor" in cmaps

    def test_list_colormaps_by_type(self):
        """Test filtering colormaps by type."""
        sequential = peepomap.list_colormaps(cmap_type="sequential")
        assert isinstance(sequential, list)
        assert all(
            peepomap.get_info(name).cmap_type == "sequential" for name in sequential
        )

    def test_list_matplotlib_colormaps(self):
        """Test listing matplotlib colormaps."""
        mpl_cmaps = peepomap.list_matplotlib_colormaps()
        assert isinstance(mpl_cmaps, list)
        assert "viridis" in mpl_cmaps


class TestColormapInfo:
    """Test colormap metadata."""

    def test_get_info_existing(self):
        """Test retrieving metadata for existing colormap."""
        info = peepomap.get_info("storm")
        assert info.name == "storm"
        assert isinstance(info.colors, list)
        assert isinstance(info.description, str)
        assert info.cmap_type in {
            "sequential",
            "diverging",
            "cyclic",
            "multi-diverging",
        }

    def test_get_info_nonexistent(self):
        """Test that requesting info for non-existent colormap raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            peepomap.get_info("nonexistent_colormap")
        assert "not found" in str(exc_info.value)
