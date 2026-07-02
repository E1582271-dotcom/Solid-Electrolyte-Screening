"""
Nature-family figure styling for Project 1 plots (shared by 01_train_eval.py,
02_shap.py, 04_screen_mp.py).

Palette + rcParams adapted from the `nature-figure` skill: restrained publication
palette, Arial sans-serif, full 4-sided axes box, frameless legends, and editable-text
SVG export (svg.fonttype='none'). Import-only; call apply_publication_style() once
before creating figures, and finalize_figure() to save 600 dpi PNG (add "svg"/"pdf" to its
`formats` for editable vector export).

Self-contained on purpose: this repo ships standalone, so it does not import Project 2.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

# ── Nature-family palette ─────────────────────────────────────────────────────
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#FFD700",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

# CVD-safe qualitative palette (Okabe-Ito) for the screening families, so the
# categories stay distinguishable for colour-blind readers and in grayscale.
FAMILY_COLORS = {
    "argyrodites": "#0072B2",    # blue
    "LGPS": "#E69F00",           # orange
    "thio-LISICON": "#009E73",   # green
    "sulfides": "#56B4E9",       # sky
    "unknown": "#999999",        # gray
}

# ── Nature journal-final geometry (inches) & font sizes (pt at final print size) ──
# Nature column widths: single 89 mm, double 183 mm; max height 247 mm.
COL_SINGLE_IN = 3.50   # 89 mm
COL_DOUBLE_IN = 7.20   # 183 mm
MAX_H_IN = 9.72        # 247 mm
# Text must sit in the 5-7 pt band at final size (Nature spec).
FS_LABEL = 7           # axis labels
FS_TICK = 6            # tick labels
FS_LEGEND = 6          # legend text
FS_ANNOT = 6           # in-plot annotations
FS_PANEL = 8           # bold a/b/c panel letters


def apply_publication_style(font_size: int = FS_LABEL, axes_linewidth: float = 0.6) -> None:
    """Apply Nature journal-final rcParams. Call once before creating any figures.

    Defaults target a figure drawn at its physical column width (COL_*_IN): 5-7 pt text,
    0.6 pt spines. Text stays editable in both SVG (fonttype='none') and PDF (fonttype=42)."""
    # MANDATORY: editable vector text (keeps <text> nodes / TrueType, not bezier paths)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    # Layout & style
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = True   # full 4-sided box (match project 2);
    plt.rcParams["axes.spines.top"] = True     # ticks stay on bottom/left only
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["axes.labelsize"] = FS_LABEL
    plt.rcParams["axes.titlesize"] = FS_LABEL
    plt.rcParams["legend.fontsize"] = FS_LEGEND
    plt.rcParams["xtick.labelsize"] = FS_TICK
    plt.rcParams["ytick.labelsize"] = FS_TICK
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 2.5
    plt.rcParams["ytick.major.size"] = 2.5
    plt.rcParams["xtick.major.width"] = axes_linewidth
    plt.rcParams["ytick.major.width"] = axes_linewidth
    plt.rcParams["savefig.bbox"] = "tight"


def add_panel_label(ax, letter, x=-0.08, y=1.04, fontsize=FS_PANEL, color=None,
                    fontweight="bold"):
    """Bold lowercase panel letter near an axes' top-left corner (Nature convention)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight=fontweight,
            color=color or PALETTE["neutral_black"], ha="left", va="bottom")


def finalize_figure(fig, out_path: str, formats=("png",), dpi: int = 600,
                    pad: float = 0.6, close: bool = True):
    """tight_layout + save to png (600 dpi raster). The out_path suffix is ignored -- one file
    per entry in `formats` (add "svg"/"pdf" here for editable vector export). Returns paths."""
    fig.tight_layout(pad=pad)
    base = Path(out_path).with_suffix("")
    os.makedirs(base.parent, exist_ok=True)
    saved = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        fig.savefig(p, dpi=dpi)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved
