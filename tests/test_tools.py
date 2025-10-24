"""Tests for peepomap.tools module."""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pytest

import peepomap


class TestCreateLinear:
    """Test linear colormap creation."""

    def test_create_linear_basic(self):
        """Test creating a basic linear colormap."""
        cmap = peepomap.create_linear("blue", "red")
        assert isinstance(cmap, LinearSegmentedColormap)
        assert cmap.N == 256

    def test_create_linear_with_name(self):
        """Test creating linear colormap with custom name."""
        cmap = peepomap.create_linear("blue", "red", name="custom")
        assert cmap.name == "custom"

    def test_create_linear_with_n(self):
        """Test creating linear colormap with custom number of colors."""
        cmap = peepomap.create_linear("blue", "red", n=128)
        assert cmap.N == 128

    def test_create_linear_reversed(self):
        """Test creating reversed linear colormap."""
        cmap = peepomap.create_linear("blue", "red", reverse=True)
        assert isinstance(cmap, LinearSegmentedColormap)


class TestReverse:
    """Test colormap reversal."""

    def test_reverse_by_name(self):
        """Test reversing a colormap by name."""
        cmap = peepomap.reverse("storm")
        assert isinstance(cmap, LinearSegmentedColormap)
        assert "_r" in cmap.name

    def test_reverse_by_object(self):
        """Test reversing a colormap object."""
        original = peepomap.get("storm")
        reversed_cmap = peepomap.reverse(original)
        assert isinstance(reversed_cmap, LinearSegmentedColormap)


class TestTruncate:
    """Test colormap truncation."""

    def test_truncate_basic(self):
        """Test basic colormap truncation."""
        cmap = peepomap.truncate("storm", 0.25, 0.75)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_truncate_by_object(self):
        """Test truncating a colormap object."""
        original = peepomap.get("storm")
        truncated = peepomap.truncate(original, 0.25, 0.75)
        assert isinstance(truncated, LinearSegmentedColormap)

    def test_truncate_with_custom_name(self):
        """Test truncation with custom name."""
        cmap = peepomap.truncate("storm", 0.0, 0.5, cmap_name="Half Storm")
        assert cmap.name == "Half Storm"

    def test_truncate_invalid_range(self):
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError):
            peepomap.truncate("storm", 0.75, 0.25)

    def test_truncate_out_of_bounds(self):
        """Test that out of bounds values raise ValueError."""
        with pytest.raises(ValueError):
            peepomap.truncate("storm", -0.1, 0.5)
        with pytest.raises(ValueError):
            peepomap.truncate("storm", 0.5, 1.1)


class TestCombine:
    """Test colormap combination."""

    def test_combine_two_colormaps(self):
        """Test combining two colormaps."""
        cmap = peepomap.combine("storm", "vapor")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_combine_with_weights(self):
        """Test combining with custom weights."""
        cmap = peepomap.combine("storm", "vapor", weights=[0.7, 0.3])
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_combine_with_name(self):
        """Test combining with custom name."""
        cmap = peepomap.combine("storm", "vapor", name="custom")
        assert cmap.name == "custom"

    def test_combine_no_colormaps(self):
        """Test that combining no colormaps raises error."""
        with pytest.raises(peepomap.exceptions.NoColormapsProvidedError):
            peepomap.combine()

    def test_combine_weights_mismatch(self):
        """Test that mismatched weights raise error."""
        with pytest.raises(peepomap.exceptions.WeightsMismatchError):
            peepomap.combine("storm", "vapor", weights=[0.5])

    def test_combine_weights_sum_error(self):
        """Test that weights not summing to 1.0 raise error."""
        with pytest.raises(peepomap.exceptions.WeightsSumError):
            peepomap.combine("storm", "vapor", weights=[0.5, 0.3])


class TestHexToDecimalRgb:
    """Test hex color conversion."""

    def test_hex_to_decimal_rgb(self):
        """Test converting hex colors to decimal RGB."""
        result = peepomap.hex_to_decimal_rgb(["#FF0000", "#00FF00", "#0000FF"])
        assert len(result) == 3
        assert np.allclose(result[0], [1.0, 0.0, 0.0])
        assert np.allclose(result[1], [0.0, 1.0, 0.0])
        assert np.allclose(result[2], [0.0, 0.0, 1.0])


class TestShift:
    """Test colormap shifting."""

    def test_shift_basic(self):
        """Test basic colormap shifting."""
        cmap = peepomap.shift("vapor", start=0.5)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_shift_by_object(self):
        """Test shifting a colormap object."""
        original = peepomap.get("vapor")
        shifted = peepomap.shift(original, start=0.5)
        assert isinstance(shifted, LinearSegmentedColormap)

    def test_shift_with_name(self):
        """Test shifting with custom name."""
        cmap = peepomap.shift("vapor", start=0.25, cmap_name="Vapor Shifted")
        assert cmap.name == "Vapor Shifted"

    def test_shift_invalid_start(self):
        """Test that invalid start value raises ValueError."""
        with pytest.raises(ValueError):
            peepomap.shift("vapor", start=1.5)


class TestAdjust:
    """Test colormap adjustment."""

    def test_adjust_saturation(self):
        """Test adjusting saturation."""
        cmap = peepomap.adjust("storm", saturation=1.5)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_adjust_by_object(self):
        """Test adjusting a colormap object."""
        original = peepomap.get("storm")
        adjusted = peepomap.adjust(original, saturation=1.5)
        assert isinstance(adjusted, LinearSegmentedColormap)

    def test_adjust_lightness(self):
        """Test adjusting lightness."""
        cmap = peepomap.adjust("storm", lightness=0.8)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_adjust_rgb_boost(self):
        """Test adjusting RGB channels."""
        cmap = peepomap.adjust("storm", red_boost=0.2, blue_boost=-0.1)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_adjust_invalid_boost(self):
        """Test that invalid boost values raise ValueError."""
        with pytest.raises(ValueError):
            peepomap.adjust("storm", red_boost=1.5)

    def test_adjust_negative_saturation(self):
        """Test that negative saturation raises ValueError."""
        with pytest.raises(ValueError):
            peepomap.adjust("storm", saturation=-0.5)


class TestCreateDiverging:
    """Test diverging colormap creation."""

    def test_create_diverging_basic(self):
        """Test creating a basic diverging colormap."""
        cmap = peepomap.create_diverging("Blues_r", "Reds")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_create_diverging_with_center(self):
        """Test creating diverging colormap with custom center."""
        cmap = peepomap.create_diverging("Blues_r", "Reds", center="white")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_create_diverging_with_diffusion(self):
        """Test creating diverging colormap with diffusion."""
        cmap = peepomap.create_diverging("Blues_r", "Reds", diffusion=0.5)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_create_diverging_with_name(self):
        """Test creating diverging colormap with custom name."""
        cmap = peepomap.create_diverging("Blues_r", "Reds", name="Custom Div")
        assert cmap.name == "Custom Div"


class TestConcat:
    """Test colormap concatenation."""

    def test_concat_two_colormaps(self):
        """Test concatenating two colormaps."""
        cmap = peepomap.concat("viridis", "plasma")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_concat_three_colormaps(self):
        """Test concatenating three colormaps."""
        cmap = peepomap.concat("viridis", "plasma", "inferno")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_concat_with_diffusion(self):
        """Test concatenating with diffusion."""
        cmap = peepomap.concat("viridis", "plasma", diffusion=0.5)
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_concat_insufficient_colormaps(self):
        """Test that concatenating less than 2 colormaps raises error."""
        with pytest.raises(ValueError):
            peepomap.concat("viridis")


class TestSetSpecialColors:
    """Test setting special colors."""

    def test_set_special_colors_bad(self):
        """Test setting bad color."""
        cmap = peepomap.get("storm")
        modified = peepomap.set_special_colors(cmap, bad="gray")
        assert isinstance(modified, LinearSegmentedColormap)

    def test_set_special_colors_all(self):
        """Test setting all special colors."""
        cmap = peepomap.get("storm")
        modified = peepomap.set_special_colors(
            cmap, bad="gray", under="navy", over="red"
        )
        assert isinstance(modified, LinearSegmentedColormap)
