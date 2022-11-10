"""Just some extra Peepo-Powered Matplotlib colormaps."""

from peepomap import avanti
from peepomap import fdtb
from peepomap import jazz
from peepomap import ons
from peepomap import plus
from peepomap import storm
from peepomap import tok
from peepomap import tools
from peepomap import vapor


cmaps = {
    "storm": storm.cmap,
    "storm_r": storm.cmap_r,
    "ons": ons.cmap,
    "ons_r": ons.cmap_r,
    "tok": tok.cmap,
    "tok_r": tok.cmap_r,
    "vapor": vapor.cmap,
    "vapor_r": vapor.cmap_r,
    "avanti": avanti.cmap,
    "avanti_r": avanti.cmap_r,
    "jazz": jazz.cmap,
    "jazz_r": jazz.cmap_r,
    "plus": plus.cmap,
    "plus_r": plus.cmap_r,
    "fdtb": fdtb.cmap,
    "fdtb_r": fdtb.cmap_r,
}


__doc__ = """Just some extra Peepo-Powered Matplotlib colormaps."""

__all__ = ["cmaps", "tools"]
