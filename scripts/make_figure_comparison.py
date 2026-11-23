"""Build the images used by docs/figure_comparison.md.

Our figures: the paper-layout versions made by scripts/paper_style_figures.py
(results/figures/paperstyle/ps_*.png), copied into docs/figure_comparison/ as ours_*.png. Paper figures: cropped from the published article and supplement (CC BY 4.0) into
paper_*.png, only if --paper-dir is given (the downloaded fig1..5.png and the supplement rendered to
supp_p1..p7.png at 300 dpi; these are not kept in the repo).

Every image is written as plain 8-bit RGB without metadata chunks. The matplotlib originals are RGBA
with pHYs/tEXt chunks, and some Markdown previewers did not display them, while re-saved RGB copies
displayed fine.

Run: pixi run python scripts/make_figure_comparison.py [--paper-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
OUT = ROOT / "docs" / "figure_comparison"

# ours_<name>.png: our results redrawn in the paper's layout by scripts/paper_style_figures.py
OURS = {name: (f"paperstyle/ps_{name}.png", None) for name in
        ("fig1c", "fig2bc", "fig2d", "fig3abc", "fig3de", "fig4d", "fig4e", "fig5", "supp1", "supp2", "supp4")}

# paper_<name>.png: (source in --paper-dir, crop box); boxes found by eye on the published images
PAPER = {
    "fig1c": ("fig1.png", (0, 1395, 1412, 2086)),
    "fig2bc": ("fig2.png", (0, 773, 2006, 1386)),
    "fig2d": ("fig2.png", (0, 1360, 2006, 1966)),
    "fig3ab": ("fig3.png", (0, 0, 1235, 590)),
    "fig3c": ("fig3.png", (0, 620, 1235, 1181)),
    "fig3de": ("fig3.png", (1240, 0, 2112, 1181)),
    "fig4d": ("fig4.png", (460, 355, 1235, 875)),
    "fig4e": ("fig4.png", (1235, 355, 2008, 875)),
    "fig5": ("fig5.png", None),
    "supp1cde": ("supp_p1.png", (300, 1430, 2220, 1790)),
    "supp2": ("supp_p2.png", (300, 300, 2250, 850)),
    "supp4a": ("supp_p4.png", (300, 300, 1260, 1090)),
}


def save_rgb(src: Path, box, dst: Path) -> None:
    """Flatten onto white (in case of transparency), crop, and save as metadata-free RGB PNG."""
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if box is not None:
        im = im.crop(box)
    im.save(dst, format="PNG", optimize=True)  # no pnginfo -> no tEXt/pHYs chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", type=Path, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, box) in OURS.items():
        save_rgb(FIGS / src, box, OUT / f"ours_{name}.png")
    if a.paper_dir is not None:
        for name, (src, box) in PAPER.items():
            save_rgb(a.paper_dir / src, box, OUT / f"paper_{name}.png")
    print(f"wrote {len(OURS)} ours_* {'and %d paper_* ' % len(PAPER) if a.paper_dir else ''}images to {OUT}")


if __name__ == "__main__":
    main()
