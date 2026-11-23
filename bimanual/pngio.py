"""Plain-RGB PNG output for figures (no matplotlib style side effects on import).

Why: matplotlib writes RGBA PNGs with pHYs/tEXt metadata chunks, and some Markdown previewers
did not display those, while plain 8-bit RGB copies without metadata displayed fine.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def to_plain_rgb(im: Image.Image) -> Image.Image:
    """Flatten onto white (in case of transparency) and return 8-bit RGB."""
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def save_png_rgb(fig, path: Path, **savefig_kwargs) -> None:
    """Render a matplotlib figure and write it as a metadata-free RGB PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", **savefig_kwargs)
    buf.seek(0)
    to_plain_rgb(Image.open(buf)).save(path, format="PNG", optimize=True)  # no pnginfo -> no metadata


def convert_png_in_place(path: Path) -> bool:
    """Re-save an existing PNG as plain RGB without metadata; returns True if it changed."""
    with Image.open(path) as im:
        im.load()
        if im.mode == "RGB" and not im.info:
            return False
        out = to_plain_rgb(im)
    out.save(path, format="PNG", optimize=True)
    return True
