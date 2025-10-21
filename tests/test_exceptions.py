"""Tests for peepomap.exceptions module."""

from peepomap.exceptions import ColormapError
from peepomap.exceptions import NoColormapsProvidedError
from peepomap.exceptions import PeepomapError
from peepomap.exceptions import WeightsMismatchError
from peepomap.exceptions import WeightsSumError


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_peepomap_error_base(self):
        """Test that PeepomapError is base exception."""
        assert issubclass(PeepomapError, Exception)

    def test_colormap_error_inherits(self):
        """Test that ColormapError inherits from PeepomapError."""
        assert issubclass(ColormapError, PeepomapError)

    def test_no_colormaps_provided_error_inherits(self):
        """Test that NoColormapsProvidedError inherits from ColormapError."""
        assert issubclass(NoColormapsProvidedError, ColormapError)

    def test_weights_mismatch_error_inherits(self):
        """Test that WeightsMismatchError inherits from ColormapError."""
        assert issubclass(WeightsMismatchError, ColormapError)

    def test_weights_sum_error_inherits(self):
        """Test that WeightsSumError inherits from ColormapError."""
        assert issubclass(WeightsSumError, ColormapError)


class TestNoColormapsProvidedError:
    """Test NoColormapsProvidedError."""

    def test_error_message(self):
        """Test error message."""
        error = NoColormapsProvidedError()
        assert "At least one colormap name must be provided" in str(error)


class TestWeightsMismatchError:
    """Test WeightsMismatchError."""

    def test_error_message(self):
        """Test error message with weights and colormaps count."""
        error = WeightsMismatchError(2, 3)
        assert "2" in str(error)
        assert "3" in str(error)
        assert error.n_weights == 2
        assert error.n_colormaps == 3


class TestWeightsSumError:
    """Test WeightsSumError."""

    def test_error_message(self):
        """Test error message with sum value."""
        error = WeightsSumError(0.8)
        assert "0.8" in str(error)
        assert "1.0" in str(error)
        assert error.weights_sum == 0.8
