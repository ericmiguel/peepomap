"""Custom exceptions for peepomap.

This module defines custom exception classes for better error handling
and more informative error messages.
"""


class PeepomapError(Exception):
    """Base exception for all peepomap errors."""

    pass


class ColormapError(PeepomapError):
    """Base exception for colormap-related errors."""

    pass


class NoColormapsProvidedError(ColormapError):
    """Raised when no colormaps are provided to combine."""

    def __init__(self) -> None:
        """Initialize the exception."""
        super().__init__("At least one colormap name must be provided")


class WeightsMismatchError(ColormapError):
    """Raised when weights count doesn't match colormaps count."""

    def __init__(self, n_weights: int, n_colormaps: int) -> None:
        """Initialize the exception.

        Parameters
        ----------
        n_weights : int
            Number of weights
        n_colormaps : int
            Number of colormaps
        """
        super().__init__(
            f"Number of weights ({n_weights}) must match "
            f"number of colormaps ({n_colormaps})"
        )
        self.n_weights = n_weights
        self.n_colormaps = n_colormaps


class WeightsSumError(ColormapError):
    """Raised when weights don't sum to 1.0."""

    def __init__(self, weights_sum: float) -> None:
        """Initialize the exception.

        Parameters
        ----------
        weights_sum : float
            Actual sum of weights
        """
        super().__init__(f"Weights must sum to 1.0, got {weights_sum:.6f}")
        self.weights_sum = weights_sum


__all__ = [
    "ColormapError",
    "NoColormapsProvidedError",
    "PeepomapError",
    "WeightsMismatchError",
    "WeightsSumError",
]
