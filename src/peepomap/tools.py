"""Tools for colormap manipulation and transformation."""

from copy import copy

from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from peepomap.colormaps import get
from peepomap.exceptions import NoColormapsProvidedError
from peepomap.exceptions import WeightsMismatchError
from peepomap.exceptions import WeightsSumError


def create_linear(
    start: str,
    end: str,
    name: str | None = None,
    n: int = 256,
    reverse: bool = False,
) -> LinearSegmentedColormap:
    """Create linear colormap between two colors.

    Parameters
    ----------
    start : str
        Starting color (matplotlib color format)
    end : str
        Ending color (matplotlib color format)
    name : str | None, optional
        Colormap name (auto-generated if None)
    n : int, default=256
        Number of colors
    reverse : bool, default=False
        Reverse the colormap

    Returns
    -------
    LinearSegmentedColormap
        Linear colormap

    Examples
    --------
    >>> cmap = peepomap.create_linear("blue", "red")
    >>> cmap = peepomap.create_linear("navy", "crimson", name="ocean", reverse=True)
    """
    # Convert colors to RGBA
    start_rgba = np.array(mcolors.to_rgba(start))
    end_rgba = np.array(mcolors.to_rgba(end))

    middle_rgba = (start_rgba + end_rgba) / 2.0
    colors = [start_rgba, middle_rgba, end_rgba]

    if name is None:
        start_name = start.replace("#", "hex") if start.startswith("#") else start
        end_name = end.replace("#", "hex") if end.startswith("#") else end
        name = f"{start_name}_to_{end_name}"

    cmap = LinearSegmentedColormap.from_list(name, colors, N=n)

    if reverse:
        cmap = create_reversed(cmap, add_suffix=False)

    return cmap


def reverse(
    name: str | LinearSegmentedColormap, add_suffix: bool = True
) -> LinearSegmentedColormap:
    """Reverse a colormap.

    Parameters
    ----------
    name : str or LinearSegmentedColormap
        Colormap name or object
    add_suffix : bool, default=True
        Add "_r" suffix to name

    Returns
    -------
    LinearSegmentedColormap
        Reversed colormap

    Examples
    --------
    >>> cmap = peepomap.reverse("storm")
    >>> cmap = peepomap.reverse(peepomap.get("storm"))
    """
    return create_reversed(name, add_suffix)


def create_reversed(
    name: str | LinearSegmentedColormap, add_suffix: bool = True
) -> LinearSegmentedColormap:
    """Reverse a colormap.

    Parameters
    ----------
    name : str or LinearSegmentedColormap
        Colormap name or object
    add_suffix : bool, default=True
        Add "_r" suffix to name

    Returns
    -------
    LinearSegmentedColormap
        Reversed colormap

    Examples
    --------
    >>> cmap = peepomap.create_reversed("storm")
    >>> cmap = peepomap.create_reversed(peepomap.get("storm"))
    """
    if isinstance(name, str):
        cmap = get(name)
        cmap_name = name
    else:
        # It's already a colormap object
        cmap = name
        cmap_name = getattr(name, "name", "custom")

    if add_suffix:
        cmap_name = f"{cmap_name}_r"

    colors = cmap(np.linspace(0, 1, 256))
    return LinearSegmentedColormap.from_list(cmap_name, colors[::-1])


def truncate(
    name: str | LinearSegmentedColormap,
    min_val: float = 0.0,
    max_val: float = 1.0,
    n: int = 256,
    *,
    cmap_name: str | None = None,
) -> LinearSegmentedColormap:
    """Extract portion of a colormap.

    Parameters
    ----------
    name : str or LinearSegmentedColormap
        Colormap name or object
    min_val : float, default=0.0
        Minimum value (0.0 to 1.0)
    max_val : float, default=1.0
        Maximum value (0.0 to 1.0)
    n : int, default=256
        Number of colors
    cmap_name : str | None, optional
        Custom name for the result (auto-generated if None)

    Returns
    -------
    LinearSegmentedColormap
        Truncated colormap

    Raises
    ------
    ValueError
        If values outside [0, 1] or min_val >= max_val

    Examples
    --------
    >>> cmap = peepomap.truncate("storm", 0.0, 0.5)
    >>> cmap = peepomap.truncate("storm", 0.25, 0.75, cmap_name="Storm Middle")
    >>> cmap = peepomap.truncate(peepomap.get("storm"), 0.0, 0.5)
    """
    if not 0.0 <= min_val <= 1.0:
        msg = f"min_val must be between 0 and 1, got {min_val}"
        raise ValueError(msg)
    if not 0.0 <= max_val <= 1.0:
        msg = f"max_val must be between 0 and 1, got {max_val}"
        raise ValueError(msg)
    if min_val >= max_val:
        msg = f"min_val ({min_val}) must be less than max_val ({max_val})"
        raise ValueError(msg)

    if isinstance(name, str):
        cmap = get(name)
        original_name = name
    else:
        # It's already a colormap object
        cmap = name
        original_name = getattr(name, "name", "custom")

    if cmap_name is None:
        truncated_name = f"{original_name}_truncated_{min_val:.2f}_{max_val:.2f}"
    else:
        truncated_name = cmap_name

    colors = cmap(np.linspace(min_val, max_val, n))

    return LinearSegmentedColormap.from_list(truncated_name, colors)


def combine(
    *colormaps: str | LinearSegmentedColormap,
    weights: list[float] | None = None,
    name: str | None = None,
    reverse: bool = False,
) -> LinearSegmentedColormap:
    """Combine colormaps using weighted average.

    Parameters
    ----------
    *colormaps : str or LinearSegmentedColormap
        Colormap names or objects
    weights : list[float] | None, optional
        Weights for each colormap (must sum to 1.0, equal if None)
    name : str | None, optional
        Colormap name (auto-generated if None)
    reverse : bool, default=False
        Reverse the result

    Returns
    -------
    LinearSegmentedColormap
        Combined colormap

    Raises
    ------
    NoColormapsProvidedError
        If no colormaps provided
    WeightsMismatchError
        If weights count doesn't match colormaps count
    WeightsSumError
        If weights don't sum to 1.0

    Examples
    --------
    >>> cmap = peepomap.combine("storm", "vapor")
    >>> cmap = peepomap.combine("storm", "viridis", weights=[0.7, 0.3])
    """
    if not colormaps:
        raise NoColormapsProvidedError

    n_maps = len(colormaps)

    # Handle weights
    if weights is None:
        weights = [1.0 / n_maps] * n_maps
    elif len(weights) != n_maps:
        raise WeightsMismatchError(len(weights), n_maps)
    elif not np.isclose(sum(weights), 1.0):
        raise WeightsSumError(sum(weights))

    # Sample each colormap
    n_samples = 256
    x = np.linspace(0, 1, n_samples)

    # Get colors from each colormap
    all_colors: list[np.ndarray] = []
    colormap_names: list[str] = []

    for colormap in colormaps:
        if isinstance(colormap, str):
            # It's a colormap name, get the colormap object
            cmap = get(colormap)
            colormap_names.append(colormap)
        else:
            # It's already a colormap object (LinearSegmentedColormap)
            cmap = colormap
            colormap_names.append(getattr(colormap, "name", "custom"))

        all_colors.append(cmap(x))

    # Combine colors using weighted average
    combined_colors = np.zeros((n_samples, 4))
    for colors, weight in zip(all_colors, weights, strict=True):
        combined_colors += colors * weight

    # Clip to valid range
    combined_colors = np.clip(combined_colors, 0, 1)

    # Use custom name if provided, otherwise auto-generate
    if name is not None:
        combined_name = name
    else:
        combined_name = "_".join(colormap_names) + "_combined"
        if reverse:
            combined_name += "_r"

    cmap = LinearSegmentedColormap.from_list(combined_name, combined_colors)

    if reverse:
        colors_reversed = cmap(np.linspace(0, 1, n_samples))[::-1]
        cmap = LinearSegmentedColormap.from_list(combined_name, colors_reversed)

    return cmap


def hex_to_decimal_rgb(colors: list[str]) -> list[list[float]]:
    """Convert hex colors to decimal RGB.

    Parameters
    ----------
    colors : list[str]
        Hex color codes (e.g., ['#FF0000', '#00FF00'])

    Returns
    -------
    list[list[float]]
        RGB values (0-1 range)

    Examples
    --------
    >>> peepomap.hex_to_decimal_rgb(["#FF0000", "#00FF00", "#0000FF"])
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    rgb_colors = np.array([np.array(mcolors.to_rgb(color)) for color in colors])

    return rgb_colors.tolist()


def shift(
    name: str | LinearSegmentedColormap,
    start: float = 0.0,
    n: int = 256,
    *,
    cmap_name: str | None = None,
) -> LinearSegmentedColormap:
    """Shift colormap by rotating starting point.

    Parameters
    ----------
    name : str or LinearSegmentedColormap
        Colormap name or object
    start : float, default=0.0
        New starting position (0.0 to 1.0)
    n : int, default=256
        Number of colors
    cmap_name : str | None, optional
        Custom name for the result (auto-generated if None)

    Returns
    -------
    LinearSegmentedColormap
        Shifted colormap

    Raises
    ------
    ValueError
        If start outside [0, 1]

    Examples
    --------
    >>> cmap = peepomap.shift("vapor", start=0.5)
    >>> cmap = peepomap.shift("hsv", start=0.25, cmap_name="HSV Rotated")
    >>> cmap = peepomap.shift(peepomap.get("vapor"), start=0.5)
    """
    if not 0.0 <= start <= 1.0:
        msg = f"start must be between 0 and 1, got {start}"
        raise ValueError(msg)

    if isinstance(name, str):
        cmap = get(name)
        original_name = name
    else:
        # It's already a colormap object
        cmap = name
        original_name = getattr(name, "name", "custom")

    if cmap_name is None:
        shifted_name = f"{original_name}_shifted_{start:.2f}"
    else:
        shifted_name = cmap_name

    # Sample the original colormap
    x = np.linspace(0, 1, n)
    colors = cmap(x)

    # Calculate the shift index
    shift_idx = int(start * n)

    # Rotate the colors
    shifted_colors = np.roll(colors, -shift_idx, axis=0)

    return LinearSegmentedColormap.from_list(shifted_name, shifted_colors)


def adjust(
    name: str | LinearSegmentedColormap,
    *,
    red_boost: float = 0.0,
    green_boost: float = 0.0,
    blue_boost: float = 0.0,
    saturation: float = 1.0,
    lightness: float = 1.0,
    n: int = 256,
    cmap_name: str | None = None,
) -> LinearSegmentedColormap:
    """Adjust colormap RGB, saturation, and lightness.

    Parameters
    ----------
    name : str or LinearSegmentedColormap
        Colormap name or object
    red_boost : float, default=0.0
        Red channel adjustment (-1.0 to 1.0)
    green_boost : float, default=0.0
        Green channel adjustment (-1.0 to 1.0)
    blue_boost : float, default=0.0
        Blue channel adjustment (-1.0 to 1.0)
    saturation : float, default=1.0
        Saturation multiplier (0.0=grayscale, >1.0=more saturated)
    lightness : float, default=1.0
        Lightness multiplier (0.0=black, >1.0=brighter)
    n : int, default=256
        Number of colors
    cmap_name : str | None, optional
        Custom name for the result (auto-generated if None)

    Returns
    -------
    LinearSegmentedColormap
        Adjusted colormap

    Raises
    ------
    ValueError
        If boosts outside [-1, 1] or saturation/lightness negative

    Examples
    --------
    >>> cmap = peepomap.adjust("storm", saturation=1.5)
    >>> cmap = peepomap.adjust(
    ...     "storm", blue_boost=0.2, lightness=0.8, cmap_name="Storm Adjusted"
    ... )
    >>> cmap = peepomap.adjust(peepomap.get("storm"), saturation=1.5)
    """
    # Validate inputs
    for boost, name_str in [
        (red_boost, "red_boost"),
        (green_boost, "green_boost"),
        (blue_boost, "blue_boost"),
    ]:
        if not -1.0 <= boost <= 1.0:
            msg = f"{name_str} must be between -1 and 1, got {boost}"
            raise ValueError(msg)

    if saturation < 0.0:
        msg = f"saturation must be >= 0, got {saturation}"
        raise ValueError(msg)

    if lightness < 0.0:
        msg = f"lightness must be >= 0, got {lightness}"
        raise ValueError(msg)

    if isinstance(name, str):
        cmap = get(name)
        original_name = name
    else:
        # It's already a colormap object
        cmap = name
        original_name = getattr(name, "name", "custom")

    if cmap_name is None:
        adjusted_name = f"{original_name}_adjusted"
    else:
        adjusted_name = cmap_name

    # Sample the colormap
    x = np.linspace(0, 1, n)
    rgba = cmap(x)
    rgb = rgba[:, :3].copy()

    # Apply RGB boosts
    rgb[:, 0] = np.clip(rgb[:, 0] + red_boost, 0, 1)
    rgb[:, 1] = np.clip(rgb[:, 1] + green_boost, 0, 1)
    rgb[:, 2] = np.clip(rgb[:, 2] + blue_boost, 0, 1)

    # Convert to HSV for saturation and lightness adjustments
    hsv = np.array([mcolors.rgb_to_hsv(color) for color in rgb])

    # Adjust saturation
    hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0, 1)

    # Adjust lightness (value in HSV)
    hsv[:, 2] = np.clip(hsv[:, 2] * lightness, 0, 1)

    # Convert back to RGB
    adjusted_rgb = np.array([mcolors.hsv_to_rgb(color) for color in hsv])

    # Preserve alpha channel
    adjusted_rgba = np.column_stack([adjusted_rgb, rgba[:, 3]])

    return LinearSegmentedColormap.from_list(adjusted_name, adjusted_rgba)


def create_diverging(
    left: str | LinearSegmentedColormap,
    right: str | LinearSegmentedColormap,
    center: str | tuple[float, float, float, float] | None = None,
    n: int = 256,
    diffusion: float = 1.0,
    reverse: bool = False,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Create diverging colormap from two colormaps.

    Each colormap occupies 50% of the space with smooth center transition.

    Parameters
    ----------
    left : str or LinearSegmentedColormap
        Left half colormap (name or object)
    right : str or LinearSegmentedColormap
        Right half colormap (name or object)
    center : str or tuple of float, optional
        Center color (auto-interpolated if None)
    n : int, default=256
        Number of colors
    diffusion : float, default=1.0
        Center transition smoothness
    reverse : bool, default=False
        Reverse the result
    name : str | None, optional
        Colormap name (auto-generated if None)

    Returns
    -------
    LinearSegmentedColormap
        Diverging colormap

    Examples
    --------
    >>> cmap = peepomap.create_diverging("jazz", "storm")
    >>> cmap = peepomap.create_diverging("jazz", "storm", center="white")
    """
    # Validate diffusion parameter
    if np.isnan(diffusion):
        msg = f"diffusion must be a finite number, got {diffusion}"
        raise ValueError(msg)

    # Get the two colormaps - handle both strings and colormap objects
    if isinstance(left, str):
        cmap_n = get(left)
        neg_name = left
    else:
        cmap_n = left
        neg_name = getattr(left, "name", "custom_neg")

    if isinstance(right, str):
        cmap_p = get(right)
        pos_name = right
    else:
        cmap_p = right
        pos_name = getattr(right, "name", "custom_pos")

    # Determine center color
    if center is None:
        # Automatic interpolation: get the mean of the endpoint colors
        # Sample colors at the endpoints where they meet in the center
        left_endpoint = cmap_n(1.0)  # End of left colormap (rightmost point)
        right_endpoint = cmap_p(0.0)  # Start of right colormap (leftmost point)

        # Calculate mean color
        center_rgba = (np.array(left_endpoint) + np.array(right_endpoint)) / 2.0
    else:
        # Manual center color
        center_rgba = np.array(mcolors.to_rgba(center))

    # Calculate how many colors for each side
    # Use odd n to ensure we have a true center point
    if n % 2 == 0:
        n += 1

    n_half = n // 2

    # Left half: sample the full range of the negative colormap
    # and apply center diffusion
    x_left = np.linspace(0, 1, n_half)
    colors_left_base = cmap_n(x_left)
    colors_left = np.zeros((n_half, 4))

    for i in range(n_half):
        # Diffusion weight: 0 at far left, increasing towards center
        raw_weight = i / (n_half - 1) if n_half > 1 else 0

        # Apply diffusion control
        if diffusion == 0.0:
            weight = 0.0  # No center influence
        elif diffusion > 0.0:
            weight = raw_weight ** (1.0 / diffusion)
        else:
            # Negative diffusion: reverse the effect
            weight = 1.0 - (1.0 - raw_weight) ** (1.0 / abs(diffusion))

        weight = np.clip(weight, 0.0, 1.0)
        colors_left[i] = (1 - weight) * colors_left_base[i] + weight * center_rgba

    # Right half: sample the full range of the positive colormap
    # and apply center diffusion
    x_right = np.linspace(0, 1, n_half)
    colors_right_base = cmap_p(x_right)
    colors_right = np.zeros((n_half, 4))

    for i in range(n_half):
        # Diffusion weight: 1 near center, decreasing towards far right
        raw_weight = (n_half - 1 - i) / (n_half - 1) if n_half > 1 else 0

        # Apply diffusion control
        if diffusion == 0.0:
            weight = 0.0  # No center influence
        elif diffusion > 0.0:
            weight = raw_weight ** (1.0 / diffusion)
        else:
            # Negative diffusion: reverse the effect
            weight = 1.0 - (1.0 - raw_weight) ** (1.0 / abs(diffusion))

        weight = np.clip(weight, 0.0, 1.0)
        colors_right[i] = (1 - weight) * colors_right_base[i] + weight * center_rgba

    # Combine: left half + center + right half
    all_colors = np.vstack([colors_left, [center_rgba], colors_right])

    # Use custom name if provided, otherwise auto-generate
    if name is not None:
        diverging_name = name
    else:
        diverging_name = f"{neg_name}_to_{pos_name}_diverging"

    # Use custom name if provided, otherwise auto-generate
    if name is not None:
        diverging_name = name
    else:
        diverging_name = f"{neg_name}_to_{pos_name}_diverging"
        if reverse:
            diverging_name += "_r"

    cmap = LinearSegmentedColormap.from_list(diverging_name, all_colors)

    if reverse:
        colors_reversed = cmap(np.linspace(0, 1, n))[::-1]
        cmap = LinearSegmentedColormap.from_list(diverging_name, colors_reversed)

    return cmap


def concat(
    *colormaps: str | LinearSegmentedColormap,
    center: str | None = None,
    n: int = 256,
    diffusion: float = 1.0,
    reverse: bool = False,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Concatenate colormaps with equal space allocation.

    Each colormap gets equal space (e.g., 2 maps=50% each, 3 maps=33.33% each).

    Parameters
    ----------
    *colormaps : str or LinearSegmentedColormap
        Colormap names or objects (minimum 2)
    center : str or None, default=None
        Transition color (auto-interpolated if None)
    n : int, default=256
        Number of colors
    diffusion : float, default=1.0
        Transition smoothness
    reverse : bool, default=False
        Reverse the result
    name : str | None, optional
        Colormap name (auto-generated if None)

    Returns
    -------
    LinearSegmentedColormap
        Concatenated colormap

    Examples
    --------
    >>> cmap = peepomap.concat("viridis", "plasma")
    >>> cmap = peepomap.concat("blues", "reds", "greens")
    """
    if len(colormaps) < 2:
        raise ValueError("At least 2 colormaps are required")

    # Get colormap objects and names
    cmap_objects: list[LinearSegmentedColormap] = []
    cmap_names: list[str] = []

    for cmap in colormaps:
        if isinstance(cmap, str):
            cmap_obj = get(cmap)
            cmap_objects.append(cmap_obj)
            cmap_names.append(cmap)
        else:
            cmap_objects.append(cmap)
            cmap_names.append(getattr(cmap, "name", "custom"))

    # Determine center color
    if center is None:
        # Automatic interpolation: get mean of all colormap endpoint colors
        endpoint_colors: list[tuple[float, float, float, float]] = []
        for cmap_obj in cmap_objects:
            # Get both endpoints (start and end) of each colormap
            endpoint_colors.append(cmap_obj(0.0))  # Start
            endpoint_colors.append(cmap_obj(1.0))  # End

        # Calculate mean color from all endpoints
        center_rgba = np.mean(endpoint_colors, axis=0)
    else:
        # Manual center color
        center_rgba = np.array(mcolors.to_rgba(center))

    # Calculate space allocation
    n_maps = len(cmap_objects)

    if n_maps == 2:
        # For 2 colormaps, behave like create_diverging with proper center segment
        n_half = (n - 1) // 2  # Reserve 1 color for center

        # Left half
        left_cmap = cmap_objects[0]
        x_left = np.linspace(0, 1, n_half)
        colors_left_base = left_cmap(x_left)
        colors_left = np.zeros((n_half, 4))

        # Apply diffusion from center
        for i in range(n_half):
            # Diffusion weight: 0 at far left, increasing towards center
            raw_weight = i / (n_half - 1) if n_half > 1 else 0

            if diffusion == 0.0:
                weight = 0.0
            elif diffusion > 0.0:
                weight = raw_weight ** (1.0 / diffusion)
            else:
                weight = 1.0 - (1.0 - raw_weight) ** (1.0 / abs(diffusion))

            weight = np.clip(weight, 0.0, 1.0)
            colors_left[i] = (1 - weight) * colors_left_base[i] + weight * center_rgba

        # Right half
        right_cmap = cmap_objects[1]
        x_right = np.linspace(0, 1, n_half)
        colors_right_base = right_cmap(x_right)
        colors_right = np.zeros((n_half, 4))

        for i in range(n_half):
            # Diffusion weight: high near center, decreasing towards far right
            raw_weight = (n_half - 1 - i) / (n_half - 1) if n_half > 1 else 0

            if diffusion == 0.0:
                weight = 0.0
            elif diffusion > 0.0:
                weight = raw_weight ** (1.0 / diffusion)
            else:
                weight = 1.0 - (1.0 - raw_weight) ** (1.0 / abs(diffusion))

            weight = np.clip(weight, 0.0, 1.0)
            colors_right[i] = (1 - weight) * colors_right_base[i] + weight * center_rgba

        # Combine: left + center + right
        final_colors = np.vstack([colors_left, [center_rgba], colors_right])

    else:
        # For 3+ colormaps, create dedicated center segments between each pair
        # Calculate space per segment including center transitions
        n_segments = n_maps
        n_transitions = n_maps - 1  # Number of center transition areas

        # Reserve space for center transitions (10% of total for each transition)
        center_space_per_transition = max(1, n // (10 * n_transitions))
        total_center_space = center_space_per_transition * n_transitions

        # Remaining space for colormap segments
        remaining_space = n - total_center_space
        colors_per_segment = remaining_space // n_segments
        remainder = remaining_space % n_segments

        all_colors: list[np.ndarray] = []

        for i, cmap_obj in enumerate(cmap_objects):
            # Calculate segment size
            segment_size = colors_per_segment + (1 if i < remainder else 0)
            segment_size = max(1, segment_size)  # Ensure at least 1 color

            # Sample the colormap
            x_segment = np.linspace(0, 1, segment_size)
            colors_segment = cmap_obj(x_segment)
            all_colors.append(colors_segment)

            # Add center transition (except after last segment)
            if i < n_maps - 1:
                # Create smooth transition using center color
                transition_colors = np.tile(
                    center_rgba, (center_space_per_transition, 1)
                )
                all_colors.append(transition_colors)

        # Combine all segments
        final_colors = np.vstack(all_colors)

    # Use custom name if provided, otherwise auto-generate
    if name is not None:
        combined_name = name
    else:
        combined_name = "_".join(cmap_names) + "_spatial"
        if reverse:
            combined_name += "_r"

    cmap = LinearSegmentedColormap.from_list(combined_name, final_colors)

    if reverse:
        colors_reversed = cmap(np.linspace(0, 1, n))[::-1]
        cmap = LinearSegmentedColormap.from_list(combined_name, colors_reversed)

    return cmap


def set_special_colors(
    cmap: LinearSegmentedColormap,
    *,
    bad: str | None = None,
    under: str | None = None,
    over: str | None = None,
) -> LinearSegmentedColormap:
    """Set colors for NaN, under-range, and over-range values.

    Parameters
    ----------
    cmap : LinearSegmentedColormap
        Colormap to modify
    bad : str | None, optional
        Color for NaN/masked values
    under : str | None, optional
        Color for values below range
    over : str | None, optional
        Color for values above range

    Returns
    -------
    LinearSegmentedColormap
        Modified colormap

    Examples
    --------
    >>> cmap = peepomap.get("storm")
    >>> cmap = peepomap.set_special_colors(cmap, bad="gray", under="navy", over="red")
    """
    # Create a copy to avoid modifying the original
    new_cmap = copy(cmap)

    if bad is not None:
        new_cmap.set_bad(bad)

    if under is not None:
        new_cmap.set_under(under)

    if over is not None:
        new_cmap.set_over(over)

    return new_cmap


def rgb_to_lab_l(rgb_color: np.ndarray) -> float:
    """Convert RGB to CIELAB L* lightness."""
    # First convert RGB to XYZ
    # Using D65 illuminant (standard daylight)
    r, g, b = rgb_color

    # Inverse sRGB companding (gamma correction)
    def srgb_to_linear(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # Convert to XYZ using sRGB matrix
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Normalize by D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # Convert to LAB L*
    def f(t: float) -> float:
        delta = 6 / 29
        if t > delta**3:
            return t ** (1 / 3)
        return t / (3 * delta**2) + 4 / 29

    l_star = 116 * f(y) - 16
    return l_star


__all__ = [
    "adjust",
    "combine",
    "create_diverging",
    "create_linear",
    "hex_to_decimal_rgb",
    "reverse",
    "rgb_to_lab_l",
    "set_special_colors",
    "shift",
    "truncate",
]
