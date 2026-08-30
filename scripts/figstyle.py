# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""One visual language for every figure in this repo.

Dark ground (thermal imagery reads wrong on white), the inferno/magma family
for anything radiometric, and a single restrained cyan accent for annotations.
Import and call ``use()`` before creating any matplotlib figure.
"""

from __future__ import annotations

import matplotlib as mpl

BG = "#0b0e14"
PANEL = "#11151f"
INK = "#e8ecf3"
MUTED = "#9aa7b8"
GRID = "#232a38"
ACCENT = "#4cc9f0"       # cyan: annotations, the coupled arm
ACCENT2 = "#f4a259"      # amber: the one-way arm / warnings
TRUTH = "#7ee081"        # green: ground truth

CMAP_THERMAL = "inferno"   # rendered counts / radiance
CMAP_TEMP = "magma"        # temperature fields
CMAP_SOURCE = "inferno"    # heat-source maps
CMAP_DIVERGING = "RdBu_r"  # residuals


def use() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": INK,
        "legend.fontsize": 8.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def colorbar(fig, im, ax, label: str):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_edgecolor(GRID)
    return cb
