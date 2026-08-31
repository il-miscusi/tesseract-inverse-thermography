# Copyright 2026 Tesseract Hackathon submission. SPDX-License-Identifier: Apache-2.0
"""Judge-facing figure for Experiment C's renderer-necessity result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle
from figstyle import ACCENT, ACCENT2, GRID, INK, MUTED, PANEL, colorbar


def clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def source_panel(ax, q, title, subtitle):
    band = max(4, int(round(0.375 * q.shape[1])))
    q_band = np.asarray(q)[:, :band]
    normalized = q_band / max(float(q_band.max()), 1e-300)
    im = ax.imshow(normalized.T, origin="lower", cmap=figstyle.CMAP_SOURCE,
                   vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title(title, pad=7)
    ax.text(0.03, 0.05, subtitle, transform=ax.transAxes, fontsize=7.2,
            color=INK, bbox=dict(boxstyle="round,pad=.28", fc=PANEL, ec=GRID))
    clean(ax)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="figures/experiment_c_renderer.json")
    ap.add_argument("--fields", default="figures/experiment_c_renderer_fields.npz")
    ap.add_argument("--out", default="figures/renderer_necessity.png")
    args = ap.parse_args()

    result = json.loads((ROOT / args.results).read_text())
    fields = np.load(ROOT / args.fields)
    full = result["results"]["full"]
    mismatch = result["results"]["calibration_mismatch"]
    delta_centroid = mismatch["centroid_shift_cells"] - full["centroid_shift_cells"]

    figstyle.use()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8),
                             gridspec_kw={"height_ratios": [1, 1.05],
                                          "hspace": 0.34, "wspace": 0.26,
                                          "top": 0.795, "bottom": 0.115,
                                          "left": 0.035, "right": 0.94})
    for i, ax in enumerate(axes.flat):
        figstyle.panel_letter(ax, "abcdef"[i], x=-0.01, y=1.05)
    source_panel(axes[0, 0], fields["q_true"], "coarse-grid source truth",
                 "analytic projection of 64×32 truth")
    source_panel(axes[0, 1], fields["q_full"], "recovery · calibrated renderer",
                 f"centroid {full['centroid_shift_cells']:.2f} cells · power {full['total_power_ratio']:.3f}")
    source_panel(axes[0, 2], fields["q_calibration_mismatch"],
                 "recovery · modest camera mismatch",
                 f"centroid {mismatch['centroid_shift_cells']:.2f} cells · power {mismatch['total_power_ratio']:.3f}")

    measured = np.asarray(fields["y_meas"])
    im = axes[1, 0].imshow(measured.T, origin="lower", cmap=figstyle.CMAP_THERMAL,
                           aspect="auto", interpolation="nearest")
    axes[1, 0].set_title("the same noisy 96×64 observation", pad=7)
    figstyle.inset_cbar(fig, im, axes[1, 0], "counts")
    clean(axes[1, 0])

    rf = np.asarray(fields["residual_full"])
    rm = np.asarray(fields["residual_calibration_mismatch"])
    lim = float(max(np.max(np.abs(rf)), np.max(np.abs(rm))))
    for ax, residual, title, rms in (
        (axes[1, 1], rf, "residual · calibrated renderer", full["pixel_rms_counts"]),
        (axes[1, 2], rm, "residual · modest camera mismatch", mismatch["pixel_rms_counts"]),
    ):
        im = ax.imshow(residual.T, origin="lower", cmap=figstyle.CMAP_DIVERGING,
                       vmin=-lim, vmax=lim, aspect="auto",
                       interpolation="nearest")
        ax.set_title(title, pad=7)
        figstyle.stamp(ax, f"pixel RMS {rms:.2f} counts", x=0.03, y=0.05,
                       fontsize=7.2)
        figstyle.inset_cbar(fig, im, ax, "counts")
        clean(ax)

    fig.text(0.5, 0.028,
             f"Calibration error moves the inferred hotspot an additional {delta_centroid:.2f} cells; "
             "both image fits remain above the 2-count noise scale.  "
             "Source panels are normalized individually; power ratios are printed.",
             ha="center", fontsize=8.8, color=ACCENT2)
    figstyle.headline(
        fig, "Calibration shifts the diagnosis",
        "the same observation, inverted through a calibrated renderer and a "
        "modestly mis-calibrated one, under independent-grid model discrepancy",
        x=0.035, y=0.975, sub_dy=0.042, size=15)
    out = ROOT / args.out
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
