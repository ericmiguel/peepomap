"""Tools for colormap manipulation and transformation."""

from copy import copy

from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from peepomap.colormaps import ColormapInfo
from peepomap.colormaps import ColormapType
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
    blend: float = 1.0,
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
    blend : float, default=1.0
        Center transition smoothness (0.0 = sharp, 1.0 = smooth, >1.0 = very smooth)
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
    >>> cmap = peepomap.create_diverging("Blues_r", "Reds", blend=0.5)
    """
    # Validate blend parameter
    if np.isnan(blend):
        msg = f"blend must be a finite number, got {blend}"
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
    # and apply center blend
    x_left = np.linspace(0, 1, n_half)
    colors_left_base = cmap_n(x_left)
    colors_left = np.zeros((n_half, 4))

    for i in range(n_half):
        # Blend weight: 0 at far left, increasing towards center
        raw_weight = i / (n_half - 1) if n_half > 1 else 0

        # Apply blend control
        if blend == 0.0:
            weight = 0.0  # No center influence
        elif blend > 0.0:
            weight = raw_weight ** (1.0 / blend)
        else:
            # Negative blend: reverse the effect
            weight = 1.0 - (1.0 - raw_weight) ** (1.0 / abs(blend))

        weight = np.clip(weight, 0.0, 1.0)
        colors_left[i] = (1 - weight) * colors_left_base[i] + weight * center_rgba

    # Right half: sample the full range of the positive colormap
    # and apply center blend
    x_right = np.linspace(0, 1, n_half)
    colors_right_base = cmap_p(x_right)
    colors_right = np.zeros((n_half, 4))

    for i in range(n_half):
        # Blend weight: 1 near center, decreasing towards far right
        raw_weight = (n_half - 1 - i) / (n_half - 1) if n_half > 1 else 0

        # Apply blend control
        if blend == 0.0:
            weight = 0.0  # No center influence
        elif blend > 0.0:
            weight = raw_weight ** (1.0 / blend)
        else:
            # Negative blend: reverse the effect
            weight = 1.0 - (1.0 - raw_weight) ** (1.0 / abs(blend))

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
    blend: float | None = None,
    n: int = 256,
    reverse: bool = False,
    name: str | None = None,
) -> LinearSegmentedColormap:
    """Concatenate colormaps with equal space allocation.

    Each colormap gets equal space (e.g., 2 maps=50% each, 3 maps=33.33% each).

    Parameters
    ----------
    *colormaps : str or LinearSegmentedColormap
        Colormap names or objects (minimum 2)
    blend : float or None, default=None
        Blend fraction for smooth transitions (0.0-0.5).
        If None or 0.0, concatenate with sharp boundaries.
        If > 0.0, create smooth transitions using this fraction of total space,
        divided equally among all transitions.
        For example, blend=0.1 means 10% of total colors used for blending,
        split across all transitions (e.g., 2 maps = 10% for 1 transition,
        3 maps = 5% per transition for 2 transitions).
    n : int, default=256
        Number of colors
    reverse : bool, default=False
        Reverse the result
    name : str | None, optional
        Colormap name (auto-generated if None)

    Returns
    -------
    LinearSegmentedColormap
        Concatenated colormap

    Raises
    ------
    ValueError
        If less than 2 colormaps provided or blend out of range

    Examples
    --------
    >>> # Sharp boundaries (no blending)
    >>> cmap = peepomap.concat("viridis", "plasma")
    >>> cmap = peepomap.concat("blues", "reds", "greens")

    >>> # With smooth blending (10% of space)
    >>> cmap = peepomap.concat("viridis", "plasma", blend=0.1)
    >>> cmap = peepomap.concat("blues", "reds", blend=0.2)
    """
    if len(colormaps) < 2:
        raise ValueError("At least 2 colormaps are required")

    if blend is not None and not 0.0 <= blend <= 0.5:
        msg = f"blend must be between 0.0 and 0.5, got {blend}"
        raise ValueError(msg)

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

    n_maps = len(cmap_objects)

    if blend is None or blend == 0.0:
        # Simple concatenation: equal space for each colormap
        colors_per_map = n // n_maps
        remainder = n % n_maps

        all_colors: list[np.ndarray] = []

        for i, cmap_obj in enumerate(cmap_objects):
            # Calculate segment size
            segment_size = colors_per_map + (1 if i < remainder else 0)
            segment_size = max(1, segment_size)  # Ensure at least 1 color

            # Sample the colormap
            x_segment = np.linspace(0, 1, segment_size)
            colors_segment = cmap_obj(x_segment)
            all_colors.append(colors_segment)

        # Combine all segments
        final_colors = np.vstack(all_colors)

    else:
        # Blended concatenation: create smooth transitions between colormaps
        n_transitions = n_maps - 1

        # Calculate blend zone size per transition
        # blend represents the TOTAL fraction of space for ALL transitions
        # Ensure we have enough space for all colormaps and blends
        # Reserve at least 2 colors per colormap
        max_blend_space = n - (n_maps * 2)  # Reserve minimum 2 colors per map

        # Calculate total blend space requested as fraction of n
        total_blend_requested = int(n * blend)

        # Divide by number of transitions to get size per transition
        if n_transitions > 0:
            blend_zone_size_requested = total_blend_requested // n_transitions
        else:
            blend_zone_size_requested = 0
        blend_zone_size_requested = max(2, blend_zone_size_requested)

        # Cap at available space
        if n_transitions > 0:
            max_allowed_blend_size = max_blend_space // n_transitions
        else:
            max_allowed_blend_size = 0
        blend_zone_size = min(blend_zone_size_requested, max_allowed_blend_size)
        blend_zone_size = max(1, blend_zone_size)  # Ensure at least 1

        # Total space for blend zones
        total_blend_space = blend_zone_size * n_transitions

        # Remaining space for colormap segments
        remaining_space = n - total_blend_space
        colors_per_map = remaining_space // n_maps
        remainder = remaining_space % n_maps

        all_colors: list[np.ndarray] = []

        for i, cmap_obj in enumerate(cmap_objects):
            # Calculate segment size
            segment_size = colors_per_map + (1 if i < remainder else 0)
            segment_size = max(1, segment_size)

            # Sample the colormap
            x_segment = np.linspace(0, 1, segment_size)
            colors_segment = cmap_obj(x_segment)
            all_colors.append(colors_segment)

            # Add blend transition (except after last segment)
            if i < n_maps - 1:
                # Get the end color of current colormap and start color of next
                end_color = np.array(cmap_objects[i](1.0))
                start_color = np.array(cmap_objects[i + 1](0.0))

                # Create linear interpolation between the two colors
                blend_colors = np.zeros((blend_zone_size, 4))
                for j in range(blend_zone_size):
                    weight = j / (blend_zone_size - 1) if blend_zone_size > 1 else 0.5
                    blend_colors[j] = (1 - weight) * end_color + weight * start_color

                all_colors.append(blend_colors)

        # Combine all segments
        final_colors = np.vstack(all_colors)

    # Use custom name if provided, otherwise auto-generate
    if name is not None:
        combined_name = name
    else:
        combined_name = "_".join(cmap_names) + "_concat"
        if reverse:
            combined_name += "_r"

    cmap = LinearSegmentedColormap.from_list(combined_name, final_colors)

    if reverse:
        colors_reversed = cmap(np.linspace(0, 1, len(final_colors)))[::-1]
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


def export(
    cmap: LinearSegmentedColormap,
    n: int = 32,
    *,
    name: str | None = None,
    cmap_type: ColormapType = "sequential",
    description: str = "",
    output_file: str | None = None,
) -> ColormapInfo:
    """Export colormap as ColormapInfo object.

    Useful for adding custom colormaps to the registry or saving them persistently.

    Parameters
    ----------
    cmap : LinearSegmentedColormap
        Colormap to export
    n : int, default=32
        Number of colors to sample from the colormap
    name : str | None, optional
        Name for the colormap (uses cmap.name if None)
    cmap_type : ColormapType, default="sequential"
        Type of colormap (sequential, diverging, cyclic, multi-diverging)
    description : str, default=""
        Description of the colormap
    output_file : str | None, optional
        If provided, saves the Python code representation to this file

    Returns
    -------
    ColormapInfo
        Colormap metadata object ready to add to the registry

    Examples
    --------
    >>> # Create a custom colormap
    >>> cmap = peepomap.concat("viridis", "plasma", blend=0.1)
    >>> # Export as ColormapInfo
    >>> info = peepomap.export(
    ...     cmap,
    ...     n=32,
    ...     name="viridis_plasma",
    ...     cmap_type="sequential",
    ...     description="Viridis blended with plasma"
    ... )
    >>> # Save to file
    >>> peepomap.export(
    ...     cmap,
    ...     name="viridis_plasma",
    ...     cmap_type="sequential",
    ...     description="Viridis blended with plasma",
    ...     output_file="my_cmap.py"
    ... )
    """
    from pathlib import Path

    # Get colormap name
    cmap_name = name if name is not None else getattr(cmap, "name", "custom")

    # Sample the colormap
    x = np.linspace(0, 1, n)
    colors_array = cmap(x)[:, :3]  # Drop alpha channel

    # Convert to list of lists
    colors_list = colors_array.tolist()

    # Create ColormapInfo object
    info = ColormapInfo(
        name=cmap_name,
        colors=colors_list,
        cmap_type=cmap_type,
        description=description,
    )

    # Save to file if requested
    if output_file is not None:
        # Format colors as Python code
        colors_str = "[\n"
        for color in colors_list:
            colors_str += f"            [{color[0]}, {color[1]}, {color[2]}],\n"
        colors_str += "        ]"

        # Create the full code snippet
        code = f'''    "{cmap_name}": ColormapInfo(
        name="{cmap_name}",
        colors={colors_str},
        cmap_type="{cmap_type}",
        description="{description}",
    ),'''

        Path(output_file).write_text(code)

    return info


__all__ = [
    "adjust",
    "combine",
    "concat",
    "create_diverging",
    "create_linear",
    "export",
    "hex_to_decimal_rgb",
    "reverse",
    "rgb_to_lab_l",
    "set_special_colors",
    "shift",
    "truncate",
]
