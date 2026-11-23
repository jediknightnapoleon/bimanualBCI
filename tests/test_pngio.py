"""Figure PNGs are written as plain RGB without metadata (some Markdown previewers need this)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from bimanual.pngio import convert_png_in_place, save_png_rgb


def _fig():
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    return fig


def test_save_png_rgb_is_plain_rgb(tmp_path):
    p = tmp_path / "a.png"
    fig = _fig()
    save_png_rgb(fig, p, dpi=50)
    plt.close(fig)
    with Image.open(p) as im:
        assert im.mode == "RGB"
        assert "dpi" not in im.info and not any(k for k in im.info if k.lower() in ("software", "dpi"))


def test_convert_in_place_flattens_transparency_onto_white(tmp_path):
    p = tmp_path / "b.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(p)  # fully transparent red
    assert convert_png_in_place(p) is True
    with Image.open(p) as im:
        assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)
    assert convert_png_in_place(p) is False  # already plain RGB: unchanged
