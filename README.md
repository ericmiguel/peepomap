# ![PeepoPing_48x48](https://user-images.githubusercontent.com/12076399/201158312-96136d13-5a86-4aba-8a16-7cfc978b16dc.png) Peepomap

Just some extra Peepo-Powered Matplotlib colormaps.

## 📦 Installation

```bash
pip install peepomap
```

## 🎨 Colormaps

```python
import peepomap

peepomap.tools.display_colormaps(pepomap.cmaps)
```

![pepomap_colormaps_darkbg](samples/pepomap_colormaps_darkbg.png#gh-dark-mode-only)

![pepomap_colormaps_lightbg](samples/pepomap_colormaps_lightbg.png#gh-light-mode-only)

## 💻 How to use

Simple import and choose a colormap from the above list by it`s name.

```python
import peepomap

cmap = peepomap.cmaps["storm"]
```

## 🏗️ Development

Create the virtual env using [Poetry](https://github.com/python-poetry/poetry):

```bash
poetry install
```
