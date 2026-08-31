# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""One visual language for every figure in this repo.

Cinematic data, journal discipline: a deep near-black ground (thermal imagery
reads wrong on white), the inferno/magma family for anything radiometric, a
single restrained cyan accent for annotations and the coupled arm, and a warm
amber for the one-way arm and gradient flow.  Import and call ``use()`` before
creating any matplotlib figure.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe

BG = "#0b0a10"           # deep near-black, faintly violet
PANEL = "#14121c"        # panel ground, one step up from BG
INK = "#f2f0f7"          # primary text
MUTED = "#8f8ba0"        # secondary text / ticks
FAINT = "#5c5870"        # tertiary text
GRID = "#262336"         # hairlines, spines, grid
ACCENT = "#5cc8f5"       # cyan: annotations, the coupled arm
ACCENT2 = "#f5a45c"      # amber: the one-way arm / gradient flow / warnings
TRUTH = "#b9a7f5"        # soft violet: ground truth / reference lines
GLOW = "#ffb347"         # luminous warm core for the gradient path

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
        "axes.titlesize": 11.5,
        "axes.titleweight": "medium",
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
        "font.sans-serif": ["Helvetica Neue", "Avenir Next", "Arial",
                            "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 110,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.22,
    })


def colorbar(fig, im, ax, label: str):
    """Slim attached colorbar, journal weight."""
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
    cb.set_label(label, fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=2.5)
    cb.outline.set_edgecolor(GRID)
    cb.outline.set_linewidth(0.6)
    return cb


def inset_cbar(fig, im, ax, label: str, ticks=None):
    """Slim vertical colorbar living inside the panel column: keeps a row of
    image panels evenly spaced (an attached bar steals width from its axes)."""
    cax = ax.inset_axes([1.03, 0.02, 0.035, 0.96])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7.5, color=MUTED, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2, width=0.5)
    cb.outline.set_edgecolor(GRID)
    cb.outline.set_linewidth(0.5)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def panel_letter(ax, letter: str, x: float = -0.015, y: float = 1.06):
    """Bold journal-style panel letter, top-left, outside the frame."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", color=INK, ha="right", va="bottom")


def headline(fig, title: str, subtitle: str | None = None,
             x: float = 0.02, y: float = 0.985, sub_dy: float = 0.052,
             size: float = 15.0):
    """Finding-as-title plus one muted subtitle line, left-aligned."""
    fig.text(x, y, title, fontsize=size, fontweight="bold", color=INK,
             ha="left", va="top")
    if subtitle:
        fig.text(x, y - sub_dy, subtitle, fontsize=9.5, color=MUTED,
                 ha="left", va="top")


def stamp(ax, text: str, x: float = 0.04, y: float = 0.06, color=INK,
          edge=None, fontsize: float = 8.0):
    """A designed verdict stamp: small caps-feel annotation in a rounded chip."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            color=color, ha="left", va="bottom", fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.45", fc=BG, ec=edge or GRID,
                      lw=0.9, alpha=0.92))


def glow_line(ax, x, y, color=GLOW, lw: float = 1.8, **kw):
    """A line with a soft luminous halo — the gradient-path treatment."""
    ln, = ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", **kw)
    ln.set_path_effects([
        pe.Stroke(linewidth=lw + 5, foreground=color, alpha=0.12),
        pe.Stroke(linewidth=lw + 2.2, foreground=color, alpha=0.25),
        pe.Normal(),
    ])
    return ln
