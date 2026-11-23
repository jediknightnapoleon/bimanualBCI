"""Build the images used by docs/figure_comparison.md.

Our figures: copied from results/figures/ (some cropped into panels) into docs/figure_comparison/ as
ours_*.png. Paper figures: cropped from the published article and supplement (CC BY 4.0) into
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

# ours_<name>.png: (source in results/figures, crop box (left, top, right, bottom) or None = whole)
OURS = {
    "fig1_tuning_curves": ("fig1_tuning_curves.png", None),
    "fig2bc_tuning_change_splithalf": ("fig2bc_tuning_change_splithalf.png", None),
    "fig2_pca": ("fig2_pca.png", None),
    "fig2_dpca": ("fig2_dpca.png", None),
    "fig3ab": ("fig3_decoding_direction.png", (0, 55, 2183, 665)),
    "fig3c": ("fig3_decoding_direction.png", (0, 680, 735, 1305)),
    "fig3_simulations": ("fig3_simulations.png", None),
    "fig4_rnn": ("fig4_rnn.png", None),
    "fig4e_online_speed": ("fig4e_online_speed.png", None),
    "fig4e_success": ("fig5_online.png", (1730, 760, 2584, 1482)),
    "fig5ab": ("fig5_online.png", (0, 60, 2584, 755)),
    "fig5c": ("fig5_online.png", (0, 760, 1730, 1482)),
    "supp_fig1_tuning_counts": ("supp_fig1_tuning_counts.png", None),
    "supp2ac": ("fig3_decoding_direction.png", (735, 680, 2183, 1305)),
    "supp_fig4_fsnr": ("supp_fig4_fsnr.png", None),
}

# paper_<name>.png: (source in --paper-dir, crop box); boxes found by eye on the published images
PAPER = {
    "fig1c": ("fig1.png", (0, 1395, 1412, 2086)),
    "fig2bc": ("fig2.png", (0, 773, 2006, 1386)),
    "fig2d_pca": ("fig2.png", (0, 1400, 490, 1966)),
    "fig2d_dpca": ("fig2.png", (1380, 1450, 2006, 1966)),
    "fig3ab": ("fig3.png", (0, 0, 1235, 590)),
    "fig3c": ("fig3.png", (0, 620, 1235, 1181)),
    "fig3de": ("fig3.png", (1240, 0, 2112, 1181)),
    "fig4b": ("fig4.png", (1365, 45, 2008, 350)),
    "fig4d": ("fig4.png", (460, 355, 1235, 875)),
    "fig4e": ("fig4.png", (1235, 355, 2008, 875)),
    "fig5a": ("fig5.png", (0, 0, 2009, 600)),
    "fig5b": ("fig5.png", (0, 615, 960, 1267)),
    "fig5c": ("fig5.png", (960, 615, 2009, 1267)),
    "supp1b": ("supp_p1.png", (920, 750, 2220, 1400)),
    "supp1cde": ("supp_p1.png", (300, 1430, 2220, 1790)),
    "supp2a": ("supp_p2.png", (300, 300, 960, 850)),
    "supp2c": ("supp_p2.png", (1580, 340, 2250, 850)),
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
