"""ONS CMAP."""

from matplotlib.colors import LinearSegmentedColormap


_colors = [
    [0.10588235294117647, 0.25882352941176473, 0.2784313725490196],
    [0.03529411764705882, 0.592156862745098, 0.6078431372549019],
    [0.4588235294117647, 0.8470588235294118, 0.8352941176470589],
    [1.0, 0.7529411764705882, 0.796078431372549],
    [0.996078431372549, 0.4980392156862745, 0.615686274509804],
    [0.396078431372549, 0.19607843137254902, 0.24313725490196078],
]

cmap = LinearSegmentedColormap.from_list("plus", _colors)
cmap_r = LinearSegmentedColormap.from_list("plus", _colors[::-1])